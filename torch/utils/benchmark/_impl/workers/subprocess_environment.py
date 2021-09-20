import os
import dataclasses
import json
import subprocess
import sys
import typing

import torch


@dataclasses.dataclass(frozen=True)
class EnvironmentSpec:
    py_executable: str
    torch_path: str
    environ: typing.Optional[typing.Dict[str, str]]
    sys_path: typing.Tuple[str, ...]

    @staticmethod
    def default() -> "EnvironmentSpec":
        return EnvironmentSpec(
            py_executable=sys.executable,
            torch_path=torch.__file__,
            environ=os.environ.copy(),
            sys_path=tuple(sys.path),
        )

    @staticmethod
    def from_source_cmd(source_cmd: str) -> "EnvironmentSpec":
        shell_executable = os.getenv("SHELL")
        if shell_executable is None:
            raise OSError("`SHELL` variable is not set.")

        # Generally a source command is used in the following context:
        #   `. activate some_env && some_other_cmd`
        # However shells differ in how they handle this. `bash` will carry
        # `some_env` while `sh` will drop it. In order to ensure that we are
        # not accidentally A/A testing when we mean to A/B test, we require
        # that SHELL is bash when collecting state for `source_cmd`.
        bash_check = subprocess.run(
            "if [ -z ${BASH+_} ]; then exit 1; fi",
            shell=True,
            executable=shell_executable,
        )
        if bash_check.returncode:
            raise OSError(f"SHELL ({shell_executable}) must be bash.")

        # Make sure the command is valid before running extraction logic so we
        # don't confuse users with subsequent calls.
        subprocess.run(
            source_cmd,
            shell=True,
            executable=shell_executable,
            check=True,
        )

        source_prefix = f"{source_cmd} && printf '\n'"
        sys_exec_result = subprocess.run(
            f"{source_prefix} && python -c 'import sys;print(sys.executable)'",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            shell=True,
            executable=shell_executable,
            check=True,
        )
        py_executable = sys_exec_result.stdout.splitlines(keepends=False)[-1]
        assert os.path.exists(py_executable), f"Invalid Python exec: `{py_executable}`"

        # Often this will be run from the root of a PyTorch repo which is using
        # the `setup.py develop` workflow. In that case, `import torch` will
        # FIRST look in CWD and only after try `sys.path`. (Which is generally
        # where the environment specific site-packages can be found.) We side
        # step this by by moving to the folder of the python executable which
        # is a neutral location to run `import torch`. We can then note the
        # torch file path, and later when we launch the real subprocess we can
        # assert that `import torch` finds the appropriate path for `source_cmd`.
        cwd = os.path.split(py_executable)[0]

        path_info_result = subprocess.run(
            " ".join([
                source_prefix, "&&",
                py_executable, "-c",
                "'import json;"
                "import os;"
                "import sys;"
                "import torch;"
                "print(json.dumps([os.environ.copy(), sys.path, torch.__file__]))'",
            ]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            shell=True,
            executable=shell_executable,
            check=True,
            cwd=cwd,
        )

        path_info = json.loads(path_info_result.stdout.splitlines(keepends=False)[-1])
        environ, sys_path, torch_path = path_info

        return EnvironmentSpec(
            py_executable=py_executable,
            torch_path=torch_path,
            environ=environ,
            sys_path=tuple(sys_path),
        )
