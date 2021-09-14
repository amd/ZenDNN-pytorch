import os
import re
import subprocess
import sys
import typing

from torch.utils.benchmark._impl.workers import subprocess_worker


_OUT_FILE_PATTERN = re.compile("callgrind.out.([0-9]+)$")

def _index(fname: str) -> int:
    match = _OUT_FILE_PATTERN.search(fname)
    return int(match.groups()[0]) + 1 if match else 0


class CallgrindWorker(subprocess_worker.SubprocessWorker):

    env_check_passed: bool = False

    def __init__(self) -> None:
        self.check_environment()
        self._callgrind_out_dir = os.path.join(self.working_dir, "callgrind")
        self._out_prefix = os.path.join(self._callgrind_out_dir, "callgrind.out")
        os.mkdir(self._callgrind_out_dir)
        super().__init__()

    @classmethod
    def check_environment(cls) -> None:
        if cls.env_check_passed:
            return

        if sys.platform not in ("linux", "linux2"):
            raise OSError(f"Valgrind is Linux only, got {sys.platform}")

        if subprocess.call(["which", "valgrind"], stdout=subprocess.DEVNULL):
            raise OSError("Could not locate Valgrind.")

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
