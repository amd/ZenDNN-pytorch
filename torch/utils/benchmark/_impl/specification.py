import dataclasses
import enum
import textwrap
import typing

from typing_extensions import Literal


class Language(enum.Enum):
    PYTHON = 0
    CPP = 1


@dataclasses.dataclass(init=False, repr=False, eq=True, frozen=True)
class WorkSpec:
    """Container to specifty execution parameters. (except globals)"""
    stmt: str
    setup: str
    teardown: str
    global_setup: str
    num_threads: int
    language: typing.Union[Language, Literal["py", "python", "c++", "cpp"]]

    def __init__(
        self,
        stmt: str,
        setup: str,
        teardown: str,
        global_setup: str,
        num_threads: int,
        language: typing.Union[Language, str]
    ) -> None:
        # timeit.Timer allows a callable, however due to the use of
        # subprocesses in some code paths we must be less permissive.
        if not isinstance(stmt, str):
            raise ValueError("Only a `str` stmt is supported.")

        if language in (Language.PYTHON, "py", "python"):
            language = Language.PYTHON
            if global_setup:
                raise ValueError(
                    f"global_setup is C++ only, got `{global_setup}`. Most "
                    "likely this code can simply be moved to `setup`."
                )

        elif language in (Language.CPP, "cpp", "c++"):
            language = Language.CPP

        else:
            raise ValueError(f"Invalid language `{language}`.")

        if language == Language.CPP and setup == "pass":
            setup = ""

        # Convenience adjustment so that multi-line code snippets defined in
        # functions do not IndentationError (Python) or look odd (C++). The
        # leading newline removal is for the initial newline that appears when
        # defining block strings. For instance:
        #   textwrap.dedent("""
        #     print("This is a stmt")
        #   """)
        # produces '\nprint("This is a stmt")\n'.
        #
        # Stripping this down to 'print("This is a stmt")' doesn't change
        # what gets executed, but it makes __repr__'s nicer.
        def tidy_str(s: str) -> str:
            s = textwrap.dedent(s)
            return (s[1:] if s and s[0] == "\n" else s).rstrip()

        object.__setattr__(self, "stmt", tidy_str(stmt))
        object.__setattr__(self, "setup", tidy_str(setup))
        object.__setattr__(self, "teardown", tidy_str(teardown))
        object.__setattr__(self, "global_setup", tidy_str(global_setup))
        object.__setattr__(self, "num_threads", num_threads)
        object.__setattr__(self, "language", language)


@dataclasses.dataclass(init=True, repr=False, eq=True, frozen=True)
class WorkMetadata:
    """Container for user provided metadata."""
    label: typing.Optional[str] = None
    sub_label: typing.Optional[str] = None
    description: typing.Optional[str] = None
    env: typing.Optional[str] = None


COMPILED_MODULE_NAME = "CompiledTimerModule"
