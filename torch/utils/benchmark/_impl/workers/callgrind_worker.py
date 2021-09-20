import enum
import os
import re
import subprocess
import sys
import typing

from torch.utils.benchmark._impl import common
from torch.utils.benchmark._impl.workers import subprocess_worker


_OUT_FILE_PATTERN = re.compile("callgrind.out.([0-9]+)$")
_TOTAL_PATTERN = re.compile(r"^([0-9,]+)\s+PROGRAM TOTALS")
_BEGIN_PATTERN = re.compile(r"Ir\s+file:function")
_FUNCTION_PATTERN = re.compile(r"^\s*([0-9,]+)\s+(.+:.+)$")

def _index(fname: str) -> int:
    """Convert callgrind.out file name to numeric index for sorting."""
    match = _OUT_FILE_PATTERN.search(fname)
    return int(match.groups()[0]) + 1 if match else 0

class ScanState(enum.Enum):
    SCANNING_FOR_TOTAL = 0
    SCANNING_FOR_START = 1
    PARSING = 2


class CallgrindWorker(subprocess_worker.SubprocessWorker):

    env_check_passed: bool = False
    _bootstrap_timeout: int = 120  # seconds

    def __init__(self, **kwargs: typing.Any) -> None:
        self.check_environment()
        self._callgrind_out_dir = os.path.join(self.working_dir, "callgrind")
        self._out_prefix = os.path.join(self._callgrind_out_dir, "callgrind.out")
        os.mkdir(self._callgrind_out_dir)
        super().__init__(**kwargs)

    @classmethod
    def check_environment(cls) -> None:
        if cls.env_check_passed:
            return

        if sys.platform not in ("linux", "linux2"):
            raise OSError(f"Valgrind is Linux only, got {sys.platform}")

        for tool in ("valgrind", "callgrind_control", "callgrind_annotate"):
            if subprocess.run(["which", tool], stdout=subprocess.DEVNULL).returncode:
                raise OSError(f"Could not locate `{tool}`.")

        cls.env_check_passed = True

    @property
    def out_files(self) -> typing.List[str]:
        """Return all `callgrind.out` files in sorted order."""

        # NB: `_index` is VERY inexpensive compared to `os.listdir`, so the
        #     fact that we call it twice isn't a big deal.
        return sorted(
            [
                os.path.join(self._callgrind_out_dir, fname)
                for fname in os.listdir(self._callgrind_out_dir)
                if _index(fname)
            ], key=_index,
        )

    @classmethod
    def annotate(cls, fpath: str, inclusive: bool) -> common.FunctionCounts:
        """Extract data from `callgrind.out` file.

            Initializing a CallgrindWorker is very expensive (>30 seconds), in
        large part because of the complex initialization of `import torch`.
        Moreover, Callgrind becomes unstable when a very large number of
        workers are simultaneous active. For ad-hoc testing this is not
        particularly important, however when running a large number of
        microbenchmarks CallgrindWorkers become very precious, and we want
        them to always be active. (For small measurements, parsing is more
        expensive than collection.)

            To this end, we separate collection which MUST occur under
        Callgrind (See `CallgrindTask.collect(...)` for details) and
        `callgrind.out` parsing which can be run in a normal subprocess.
        This allows the microbenchmark suite to easily parse on a separate
        threadpool and maximize utilization of CallgrindWorkers.
        """
        cls.check_environment()
        result = subprocess.run(
            [
                "callgrind_annotate",
                f"--inclusive={'yes' if inclusive else 'no'}",
                "--threshold=100",
                "--show-percs=no",
                fpath,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )

        for line in result.stderr.splitlines(keepends=False):
            # `callgrind_annotate` has a bug in the line number extraction
            # where it sometimes fails to initialize values. In that case perl
            # will emit a warning, but the program still succeeds. We swallow
            # those warnings (but emit any other) to reduce logging chatter.
            if line.startswith("Use of uninitialized value"):
                continue
            print(line, file=sys.stderr)

        result.check_returncode()

        scan_state = ScanState.SCANNING_FOR_TOTAL
        fn_counts: typing.List[common.FunctionCount] = []
        for l in result.stdout.splitlines(keepends=False):
            if scan_state == ScanState.SCANNING_FOR_TOTAL:
                total_match = _TOTAL_PATTERN.match(l)
                if total_match:
                    program_totals = int(total_match.groups()[0].replace(",", ""))
                    scan_state = ScanState.SCANNING_FOR_START

            elif scan_state == ScanState.SCANNING_FOR_START:
                if _BEGIN_PATTERN.match(l):
                    scan_state = ScanState.PARSING

            else:
                assert scan_state == ScanState.PARSING
                fn_match = _FUNCTION_PATTERN.match(l)
                if fn_match:
                    ir_str, file_function = fn_match.groups()
                    ir = int(ir_str.replace(",", ""))
                    if ir == program_totals:
                        # Callgrind includes some top level red herring symbols when
                        # a program dumps multiple profiles.
                        continue
                    fn_counts.append(common.FunctionCount(ir, file_function))

                elif re.match(r"-+", l):
                    # Ignore heading separator lines.
                    continue

                else:
                    break

        assert scan_state == ScanState.PARSING, f"Failed to parse {fpath}"
        return common.FunctionCounts(tuple(sorted(fn_counts, reverse=True)), inclusive=inclusive)

    @property
    def args(self) -> typing.List[str]:
        return [
            "valgrind",
            "--tool=callgrind",
            f"--callgrind-out-file={self._out_prefix}",
            "--dump-line=yes",
            "--dump-instr=yes",
            "--instr-atstart=yes",
            "--collect-atstart=no",
        ] + super().args

    @property
    def environ(self) -> typing.Dict[str, str]:
        environ = super().environ or os.environ.copy()

        # https://github.com/python/cpython/blob/b0544ba77cf86074fb1adde00558c67ca75eeea1/Misc/README.valgrind
        environ["PYTHONMALLOC"] = "malloc"
        return environ
