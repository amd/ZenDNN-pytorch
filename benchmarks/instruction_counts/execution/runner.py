import collections
import dataclasses
# import math
import multiprocessing.dummy
import os
import random
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import typing

from torch.utils.benchmark._impl import common
from torch.utils.benchmark._impl.tasks.callgrind import CallgrindTask
from torch.utils.benchmark._impl.workers import callgrind_worker
from torch.utils.benchmark._impl.workers.subprocess_environment import EnvironmentSpec

from core.api import AutoLabels, Language, WorkSpec
from core.types import Label


CALLGRIND_N_ITER = 25


#   The first few times we run a benchmark in Python the results vary slightly
# as the interpreter "burns in". (e.g. various caches are populated.)
# Unfortunately the "stable" instruction count varies from process to process.
#   In order to minimize noise from this interpreter jitter, we require that
# the first five measurements for a particular benchmark are performed in
# different processes. After that, if there is still noise the subsequent
# replicates are free to schedule on whatever worker is available. (As we're
# unlikely to get a pristine result at that point.) For C++ we require two
# separate processes just to check determinism. This is enforced by
# `WorkUnit.excluded_workers`.
MIN_INDEPENDENT_PROCS = {
    Language.PYTHON: 7,
    Language.CPP: 1,
}

IN_PROCESS_REPEATS = {
    Language.PYTHON: 1,
    Language.CPP: 9,
}

PY_FOLLOW_UP_SIZE = 4
PY_MAX_RUNS = MIN_INDEPENDENT_PROCS[Language.PYTHON] + PY_FOLLOW_UP_SIZE * 4


class SourceToEnv:

    _lock = threading.Lock()
    _map: typing.Dict[str, EnvironmentSpec] = {}

    @classmethod
    def get(cls, source_cmd: typing.Optional[str] = None) -> EnvironmentSpec:
        if source_cmd is None:
            return EnvironmentSpec.default()

        with cls._lock:
            if source_cmd not in cls._map:
                cls._map[source_cmd] = EnvironmentSpec.from_source_cmd(source_cmd=source_cmd)

            return cls._map[source_cmd]


class PinnedCallgrindWorker(callgrind_worker.CallgrindWorker):

    _gomp_warned: bool = False

    def __init__(
        self,
        source_cmd: typing.Optional[str],
        affinity: typing.Tuple[int, ...],
        **kwargs: typing.Any,
    ) -> None:
        self._source_cmd = source_cmd
        self._affinity = affinity
        self._affinity_str = ",".join([str(i) for i in self._affinity])

        super().__init__(env_spec=SourceToEnv.get(source_cmd), **kwargs)

        # Allow us to enforce exclusive ownership.
        self.owned: bool = False

        # We free workers on an LRU basis.
        self.last_used = time.time()

    @property
    def args(self) -> typing.List[str]:
        return ["taskset", "--cpu-list", self._affinity_str] + super().args

    @property
    def source_cmd(self) -> typing.Optional[str]:
        return self._source_cmd

    @property
    def environ(self) -> typing.Dict[str, str]:
        environ = super().environ
        if environ.get("GOMP_CPU_AFFINITY", "") in environ and not self._gomp_warned:
            self._gomp_warned = True

            print(textwrap.dedent(f"""
                ============================================================
                == WARNING: GOMP_CPU_AFFINITY set ==========================
                ============================================================

                GOMP_CPU_AFFINITY={environ['GOMP_CPU_AFFINITY']}

                This config interacts very strangely with Callgrind, and
                causes I/O with os.pipe to serialize on a single core and
                prevents any meaningful scaling of the benchmark runner. It
                is STRONGLY advised that this flag be unset before running
                the suite.
            """).strip())

        environ.update({
            # Mitigate https://github.com/pytorch/pytorch/issues/37377
            "MKL_THREADING_LAYER": "GNU",
        })
        return environ


class BorrowedWorker:

    def __init__(
        self,
        worker: PinnedCallgrindWorker,
        is_new_worker: bool,
        num_threads: int,
        release_fn: typing.Callable[[PinnedCallgrindWorker, int], None],
    ) -> None:
        worker.owned, valid_worker = True, not worker.owned
        assert valid_worker, f"{worker} is already owned"
        self._worker: typing.Optional[PinnedCallgrindWorker] = worker
        self.is_new_worker = is_new_worker
        self._num_threads = num_threads
        self._release_fn = release_fn

    @property
    def worker(self) -> PinnedCallgrindWorker:
        worker = self._worker
        assert worker is not None
        return worker

    @property
    def num_threads(self) -> int:
        return self._num_threads

    def release(self) -> None:
        worker = self._worker
        if worker is None:
            return

        self._release_fn(worker, self.num_threads)
        self._worker = None

    def __del__(self) -> None:
        # Release in dtor to ensure we can't leak workers.
        self.release()


@dataclasses.dataclass(frozen=True)
class WorkOrder:
    label: Label
    autolabels: AutoLabels
    work_spec: WorkSpec
    source_cmd: typing.Optional[str]

    # NB: This field cannot be reassigned due to `frozen=True`, but the list may
    #     be mutated. This is deliberate. See `excluded_workers` for details.
    prior_run_worker_ids: typing.List[int]  # TODO: handle in __post_init__

    def begin(self, worker: PinnedCallgrindWorker) -> None:
        self.prior_run_worker_ids.append(id(worker))

    @property
    def excluded_workers(self) -> typing.Set[int]:
        # See the note above `MIN_INDEPENDENT_PROCS` for details.
        if len(self.prior_run_worker_ids) >= MIN_INDEPENDENT_PROCS[self.work_spec.language]:
            return set()
        return set(self.prior_run_worker_ids)

    def __hash__(self) -> int:
        return hash(id(self))


class WorkerPool:

    def __init__(
        self,
        affinity: typing.Optional[typing.List[int]] = None,
    ) -> None:
        self._affinity = tuple(affinity or os.sched_getaffinity(os.getpid()))

        # Our goal is to reuse workers to save on startup time. As a result, we
        # maintain a pool which is larger than the capacity that can be
        # simultaneously scheduled.
        self._max_scheduled_threads = len(self._affinity)
        self._currently_scheduled_threads = 0
        self._max_capacity = 2 * len(self._affinity)

        self._lock = threading.RLock()
        self._available_workers: typing.List[PinnedCallgrindWorker] = []
        self._outstanding_workers: int = 0

    def get(self, work_order: WorkOrder) -> typing.Optional[BorrowedWorker]:
        num_threads = work_order.work_spec.num_threads
        delete_queue: typing.List[PinnedCallgrindWorker] = []
        with self._lock:
            # Check if execution would put us over our concurrency limit.
            if num_threads + self._currently_scheduled_threads > self._max_scheduled_threads:
                return None

            self._currently_scheduled_threads += num_threads
            self._outstanding_workers += 1

            # Check if we can reuse a worker.
            excluded_worker_ids = work_order.excluded_workers
            worker_candidates = [
                w for w in self._available_workers
                if w.source_cmd == work_order.source_cmd
                and id(w) not in excluded_worker_ids
            ]

            # We found a worker that we can reuse.
            if worker_candidates:
                worker = random.choice(worker_candidates)
                self._available_workers = [
                    w for w in self._available_workers if w is not worker]
                return BorrowedWorker(
                    worker=worker,
                    is_new_worker=False,
                    num_threads=num_threads,
                    release_fn=self.release_fn,
                )

            while len(self._available_workers) >= self._max_capacity:
                worker_to_replace = min(self._available_workers, key=self.lru_key)
                self._available_workers = [
                    w for w in self._available_workers if w is not worker_to_replace]
                assert not worker_to_replace.owned
                delete_queue.append(worker_to_replace)

        for w in delete_queue:
            print(f"Deleting: {w}")
            del w

        # Worker creation is fairly expensive (>30 seconds) and we've already
        # declared the thread and worker count increases above, so there's no
        # need to hold the lock while we create our worker.
        worker = PinnedCallgrindWorker(
            source_cmd=work_order.source_cmd,
            affinity=self._affinity,
        )
        return BorrowedWorker(
            worker=worker,
            is_new_worker=True,
            num_threads=num_threads,
            release_fn=self.release_fn,
        )

    def release_fn(self, worker: PinnedCallgrindWorker, num_threads: int) -> None:
        with self._lock:
            worker.owned = False
            worker.last_used = time.time()
            self._available_workers.append(worker)
            self._currently_scheduled_threads -= num_threads
            assert self._currently_scheduled_threads >= 0

    @staticmethod
    def lru_key(w: PinnedCallgrindWorker) -> float:
        return w.last_used


class RunnerSchedule:

    def __init__(self, work_orders: typing.Tuple[WorkOrder, ...]) -> None:
        self._iter_called = False
        self._work_orders = work_orders
        self._worker_pool = WorkerPool()
        self._lock = threading.Lock()

        self._dir = tempfile.mkdtemp()
        self._out_index = 0

        self._num_in_progress = {work_order: 0 for work_order in work_orders}
        self._finished: typing.Dict[WorkOrder, typing.List[int]] = {work_order: [] for work_order in work_orders}
        self._finished_verbose: typing.Dict[WorkOrder, typing.List[common.FunctionCounts]] = {
            work_order: [] for work_order in work_orders}

        self._queue: typing.Deque[WorkOrder] = collections.deque()
        for i in range(max(MIN_INDEPENDENT_PROCS.values())):
            for work_order in work_orders:
                if i < MIN_INDEPENDENT_PROCS[work_order.work_spec.language]:
                    self._enque(work_order)

    def __del__(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)

    def __iter__(self) -> "RunnerSchedule":
        assert not self._iter_called, "RunnerSchedule is not reusable"
        self._iter_called = True
        return self

    def __next__(self) -> WorkOrder:
        try:
            return self._queue.popleft()
        except IndexError:
            pass

        raise StopIteration

        # while True:
        #     no_more_repeats = []
        #     for work_order, num_in_progress in self._num_in_progress.items():
        #         if num_in_progress:
        #             continue

        #         if work_order.work_spec.language == Language.CPP:
        #             no_more_repeats.append(work_order)
        #             continue

        #         stmt_repr = work_order.work_spec.stmt.replace("\n", "\\n")[:100]

        #         if len(self._finished[work_order]) >= PY_MAX_RUNS:
        #             no_more_repeats.append(work_order)
        #             print(f"Giving up: {stmt_repr}")
        #             continue

        #         finished = sorted(self._finished[work_order])
        #         k = math.ceil(len(finished) / 2) - 2
        #         mid_finished = finished[k:-k]
        #         assert len(mid_finished) in (3, 4)

        #         print(mid_finished, stmt_repr)
        #         if len(set(mid_finished)) == 1:
        #             # We have converged.
        #             no_more_repeats.append(work_order)
        #             print(f"Finished: {stmt_repr}")
        #         else:
        #             for _ in range(PY_FOLLOW_UP_SIZE):
        #                 self._enque(work_order)

        #     for work_order in no_more_repeats:
        #         self._num_in_progress.pop(work_order)

        #     if not self._num_in_progress:
        #         raise StopIteration

        #     try:
        #         return self._queue.popleft()
        #     except IndexError:
        #         time.sleep(1)

    def _enque(self, work_order: WorkOrder) -> None:
        self._queue.append(work_order)
        self._num_in_progress[work_order] += 1

    def map_fn(self, work_order: WorkOrder) -> None:
        try:
            return self._map_fn(work_order)

        except Exception as e:
            # Threading will swallow exceptions, so we have to catch and
            # manually print instead.
            with self._lock:
                print("\n")
                traceback.print_exception(
                    etype=type(e),
                    value=e,
                    tb=sys.exc_info()[2],
                )
                print()
            raise

        finally:
            self._num_in_progress[work_order] -= 1

    def _map_fn(self, work_order: WorkOrder) -> None:
        while True:
            borrowed_worker = self._worker_pool.get(work_order)
            if borrowed_worker is not None:
                break
            time.sleep(0.5)

        start_time = time.time()
        work_order.begin(borrowed_worker.worker)
        task = CallgrindTask(
            work_spec=work_order.work_spec,
            worker=borrowed_worker.worker,
        )

        begin_collect_time = time.time()
        raw_out_files = []
        for _ in range(IN_PROCESS_REPEATS[work_order.work_spec.language]):
            raw_out_files.append(task.collect(n_iter=CALLGRIND_N_ITER))

        out_files = []
        with self._lock:
            for raw_out_file in raw_out_files:
                dest = os.path.join(self._dir, f"callgrind.out.{self._out_index}")
                self._out_index += 1
                shutil.move(raw_out_file, dest)
                out_files.append(dest)

        borrowed_worker.release()

        begin_annotate_time = time.time()
        for out_file in out_files:
            counts = callgrind_worker.CallgrindWorker.annotate(fpath=out_file, inclusive=False)
            os.remove(out_file)
            total = int(counts.denoise().sum() // CALLGRIND_N_ITER)

            with self._lock:
                self._finished[work_order].append(total)
                self._finished_verbose[work_order].append(counts)
        done_time = time.time()

        stmt_repr = work_order.work_spec.stmt.replace("\n", "\\n")[:60]
        print(
            f"{begin_collect_time - start_time:6.2f}    "
            f"{begin_annotate_time - begin_collect_time:6.2f}    "
            f"{done_time - begin_annotate_time:6.2f}    "
            f"{work_order.work_spec.language:>3} "
            f"{repr(sorted(self._finished[work_order])):<20} "
            f"{stmt_repr}"
        )


class Runner:
    def __init__(
        self,
        work_orders: typing.Tuple[WorkOrder, ...],
    ) -> None:
        # Reduce scale for testing.
        work_orders = tuple([w for w in work_orders if w.work_spec.language == Language.CPP])
        work_orders = tuple([w for w in work_orders if "backward" not in w.work_spec.stmt])

        self._work_orders = work_orders

    def run(self) -> None:
        start_time = time.time()
        schedule = RunnerSchedule(self._work_orders)
        num_workers = len(schedule._worker_pool._affinity)
        with multiprocessing.dummy.Pool(num_workers) as pool:
            for _ in pool.imap(schedule.map_fn, schedule, 1):
                pass

        # for i, work_order in enumerate(schedule):
        #     schedule.map_fn(work_order)
        #     print(i)

        end_time = time.time()
        print(f"Time: {end_time - start_time:.1f} sec")

        # import pdb
        # pdb.set_trace()

        # for work_order, counts in schedule._finished_verbose.items():
        #     counts = [count.as_standardized().denoise() for count in counts]
        #     n_by_fn = {}
        #     for count in counts:
        #         for n, fn in count:
        #             _ = n_by_fn.setdefault(fn, [])
        #             n_by_fn[fn].append(int(n // CALLGRIND_N_ITER))

        #     finished = sorted(schedule._finished[work_order])
        #     k = math.ceil(len(finished) / 2) - 4
        #     print(work_order.work_spec.stmt, finished[k:-k] or finished)
        #     for fn, n_list in n_by_fn.items():
        #         if len(n_list) > 11:
        #             k = math.ceil(len(n_list) / 2) - 4
        #             n_list = n_list[k:-k]
        #         if len(set(n_list)) == 1:
        #             continue
        #         print(fn, sorted(n_list))
        #     print()
