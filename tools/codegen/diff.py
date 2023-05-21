from __future__ import annotations

import abc
import argparse
import contextlib
import dataclasses
import logging
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import typing
from collections.abc import Iterator, Sequence, Set
from typing import Any


IGNORED_TARGETS: Set[str] = {
    # This is not generated code, but instead a downloaded data set.
    "//:download_mnist",
}


@contextlib.contextmanager
def parent_commit(repo: Repository, /) -> Iterator[None]:
    commit = repo.whereami()
    repo.goto(CommitId("prev"))
    try:
        yield None
    finally:
        repo.goto(commit)


CommitId = typing.NewType("CommitId", str)


class Repository(abc.ABC):
    @abc.abstractmethod
    def whereami(self, /) -> CommitId:
        ...

    @abc.abstractmethod
    def goto(self, commit: CommitId, /) -> None:
        ...


class Sapling(Repository):
    def whereami(self, /) -> CommitId:
        return CommitId(self._run("whereami", stdout=subprocess.PIPE).stdout)

    def goto(self, commit: CommitId, /) -> None:
        self._run("goto", commit)

    def _run(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = ["sl"]
        cmd.extend(args)
        return subprocess.run(cmd, check=True, **kwargs)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument("--keep-files", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv[1:])

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    os.chdir(pathlib.Path(os.environ["BUILD_WORKING_DIRECTORY"]))
    with contextlib.ExitStack() as stack:
        if args.keep_files:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = stack.enter_context(tempfile.TemporaryDirectory())

        run(Sapling(), pathlib.Path(temp_dir))


@dataclasses.dataclass
class DiffContext:
    targets: Set[str]
    files: Set[pathlib.Path]


def stat_generated_files(temp_dir: pathlib.Path) -> DiffContext:
    bazel = Bazel()

    genrule_targets = set(bazel.query("kind(genrule, //...)"))
    genrule_targets -= IGNORED_TARGETS

    bazel.build(*genrule_targets)

    queries = []
    for target in genrule_targets:
        queries.append(f'labels("out", {target})')
        queries.append(f'labels("outs", {target})')

    files = set()
    outs = bazel.query(" union ".join(queries))
    for out in outs:
        path = label_to_path(out)
        src = "bazel-bin" / path
        dest = temp_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        files.add(src.rename(temp_dir / path))

    return DiffContext(targets=genrule_targets, files=files)


def run(repo: Repository, temp_dir: pathlib.Path) -> None:
    with parent_commit(repo):
        before = stat_generated_files(temp_dir / "before/")
    after = stat_generated_files(temp_dir / "after/")
    subprocess.run(
        ["diff", "--color", "--recursive", temp_dir / "before", temp_dir / "after"],
        check=True,
    )


def label_to_path(label: str) -> pathlib.Path:
    match = _LABEL.fullmatch(label)
    assert match is not None
    return pathlib.Path(match.group(1)) / match.group(2)


class Bazel:
    def __init__(self, path: pathlib.Path = pathlib.Path("bazelisk")) -> None:
        self.path = path

    def build(self, /, *targets: str) -> None:
        subprocess.run(self._make_cmd("build", *targets), check=True)

    def query(self, /, query: str) -> list[str]:
        return subprocess.run(
            self._make_cmd("query", query), stdout=subprocess.PIPE, text=True
        ).stdout.splitlines()

    def _make_cmd(self, /, *args: str) -> Sequence[str]:
        """Creates and logs the command to execute."""
        cmd = [os.fspath(self.path)]
        cmd.extend(args)
        _logger.debug("%s", shlex.join(cmd))
        return cmd


_logger = logging.getLogger(__name__)


_LABEL = re.compile("//([^:]*):(.*)")

if __name__ == "__main__":
    main(sys.argv)
