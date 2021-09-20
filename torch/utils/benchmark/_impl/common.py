import collections
import dataclasses
import re
import typing

import torch
from torch.utils.benchmark._impl import specification


class CommonStatistics:

    sorted_x: typing.Tuple[float, ...]
    median: float
    mean: float
    p25: float
    p75: float

    def __init__(self, x: typing.List[float]):
        self.sorted_x = tuple(sorted(x))
        _sorted_x = torch.tensor(self.sorted_x, dtype=torch.float64)
        self.median = _sorted_x.quantile(.5).item()
        self.mean = _sorted_x.mean().item()
        self.p25 = _sorted_x.quantile(.25).item()
        self.p75 = _sorted_x.quantile(.75).item()

    @property
    def iqr(self) -> float:
        return self.p75 - self.p25


@dataclasses.dataclass(init=True, repr=False)
class Measurement:
    """The result of a Timer measurement.
    NOTE: This is a placeholder. The full (existing) Measurement class will
          be ported in a later PR
    """
    number_per_run: int
    raw_times: typing.List[float]


FunctionCount = typing.NamedTuple("FunctionCount", [("count", int), ("function", str)])


@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class FunctionCounts(object):
    """Container for manipulating Callgrind results.
    It supports:
        1) Addition and subtraction to combine or diff results.
        2) Tuple-like indexing.
        3) A `denoise` function which strips CPython calls which are known to
           be non-deterministic and quite noisy.
        4) Two higher order methods (`filter` and `transform`) for custom
           manipulation.
    """
    _data: typing.Tuple[FunctionCount, ...]
    inclusive: bool
    truncate_rows: bool = True

    # For normal use, torch._tensor_str.PRINT_OPTS.linewidth determines
    # the print settings. This is simply to allow hermetic unit tests.
    _linewidth: typing.Optional[int] = None

    def __iter__(self) -> typing.Generator[FunctionCount, None, None]:
        for i in self._data:
            yield i

    def __len__(self) -> int:
        return len(self._data)

    # The Union return type follows the convention of indexable builtin data
    # types: e.g. given `x: Tuple[T]`, x[0] has type T but x[:1] has type `Tuple[T]`.
    def __getitem__(
        self,
        item: typing.Any
    ) -> typing.Union[FunctionCount, "FunctionCounts"]:
        data_i = self._data[item]
        if isinstance(data_i, tuple):
            return FunctionCounts(data_i, self.inclusive, truncate_rows=False)

        assert isinstance(data_i, FunctionCount)
        return data_i

    def __repr__(self) -> str:
        count_len = 0
        for c, _ in self:
            # Account for sign in string length.
            count_len = max(count_len, len(str(c)) + int(c < 0))

        lines = []
        linewidth = self._linewidth or torch._tensor_str.PRINT_OPTS.linewidth
        fn_str_len = max(linewidth - count_len - 4, 40)
        for c, fn in self:
            if len(fn) > fn_str_len:
                left_len = int((fn_str_len - 5) // 2)
                fn = fn[:left_len] + " ... " + fn[-(fn_str_len - left_len - 5):]
            lines.append(f"  {c:>{count_len}}  {fn}")

        if self.truncate_rows and len(lines) > 18:
            lines = lines[:9] + ["...".rjust(count_len + 2)] + lines[-9:]

        if not self.inclusive:
            lines.extend(["", f"Total: {self.sum()}"])

        return "\n".join([super().__repr__()] + lines)

    def __add__(
        self,
        other,  # type: FunctionCounts
    ) -> "FunctionCounts":
        return self._merge(other, lambda c: c)

    def __sub__(
        self,
        other,  # type: FunctionCounts
    ) -> "FunctionCounts":
        return self._merge(other, lambda c: -c)

    def __mul__(self, other: typing.Union[int, float]) -> "FunctionCounts":
        return self._from_dict({
            fn: int(c * other) for c, fn in self._data
        }, self.inclusive)

    def transform(self, map_fn: typing.Callable[[str], str]) -> "FunctionCounts":
        """Apply `map_fn` to all of the function names.

            This can be used to regularize function names (e.g. stripping
        irrelevant parts of the file path), coalesce entries by mapping
        multiple functions to the same name (in which case the counts are added
        together), etc.
        """
        counts: typing.DefaultDict[str, int] = collections.defaultdict(int)
        for c, fn in self._data:
            counts[map_fn(fn)] += c

        return self._from_dict(counts, self.inclusive)

    def substitute(self, before: str, after: str) -> "FunctionCounts":
        """Convenience function for regex substitution."""
        return self.transform(lambda fn: re.sub(before, after, fn))


    def filter(self, filter_fn: typing.Callable[[str], bool]) -> "FunctionCounts":
        """Keep only the elements where `filter_fn` applied to function name returns True."""
        return FunctionCounts(tuple(i for i in self if filter_fn(i.function)), self.inclusive)

    def sum(self) -> int:
        return sum(c for c, _ in self)

    def denoise(self) -> "FunctionCounts":
        """Remove known noisy instructions.

            Several instructions in the CPython interpreter are rather noisy.
        These instructions involve unicode to dictionary lookups which Python
        uses to map variable names. FunctionCounts is generally a content
        agnostic container, however this is sufficiently important for obtaining
        reliable results to warrant an exception."""
        return self.filter(lambda fn: "dictobject.c:lookdict_unicode" not in fn)

    def as_standardized(self) -> "FunctionCounts":
        """Strip library names and some prefixes from function strings.

            When comparing two different sets of instruction counts, one
        stumbling block can be path prefixes. Callgrind includes the full
        filepath when reporting a function (as it should). However, this can
        cause issues when diffing profiles. If a key component such as Python
        or PyTorch was built in separate locations in the two profiles, this
        can result in something resembling::

            23234231 /tmp/first_build_dir/thing.c:foo(...)
             9823794 /tmp/first_build_dir/thing.c:bar(...)
              ...
               53453 .../aten/src/Aten/...:function_that_actually_changed(...)
              ...
             -9823794 /tmp/second_build_dir/thing.c:bar(...)
            -23234231 /tmp/second_build_dir/thing.c:foo(...)

            Stripping prefixes can ameliorate this issue by regularizing the
        strings and causing better cancellation of equivilent call sites when
        diffing.
        """
        transforms = (
            # PyTorch may have been built in different locations.
            (r"^.+build/\.\./", "build/../"),
            (r"^.+/" + re.escape("build/aten/"), "build/aten/"),

            # "Python" and "Objects" come from CPython.
            (r"^.+/" + re.escape("Python/"), "Python/"),
            (r"^.+/" + re.escape("Objects/"), "Objects/"),

            # Strip library name. e.g. `libtorch.so`
            (r"\s\[.+\]$", ""),
        )

        result = self
        for before, after in transforms:
            result = result.substitute(before=before, after=after)

        return result

    def _merge(
        self,
        second,   # type: FunctionCounts
        merge_fn: typing.Callable[[int], int]
    ) -> "FunctionCounts":
        assert self.inclusive == second.inclusive, "Cannot merge inclusive and exclusive counts."
        counts: typing.DefaultDict[str, int] = collections.defaultdict(int)
        for c, fn in self:
            counts[fn] += c

        for c, fn in second:
            counts[fn] += merge_fn(c)

        return self._from_dict(counts, self.inclusive)

    @staticmethod
    def _from_dict(counts: typing.Dict[str, int], inclusive: bool) -> "FunctionCounts":
        flat_counts = (FunctionCount(c, fn) for fn, c in counts.items() if c)
        return FunctionCounts(tuple(sorted(flat_counts, reverse=True)), inclusive)


@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class CallgrindStats(object):
    """Top level container for Callgrind results collected by Timer.

        Manipulation is generally done using the FunctionCounts class, which is
    obtained by calling `CallgrindStats.stats(...)`. Several convenience
    methods are provided as well; the most significant is
    `CallgrindStats.as_standardized()`.
    """
    work_spec: specification.WorkSpec
    work_metadata: specification.WorkMetadata
    number_per_run: int
    built_with_debug_symbols: bool
    stmt_inclusive_stats: FunctionCounts
    stmt_exclusive_stats: FunctionCounts
    stmt_callgrind_out: typing.Optional[str]

    def __repr__(self) -> str:
        # TODO: __repr__
        return super().__repr__()

    def stats(self, inclusive: bool = False) -> FunctionCounts:
        """Returns detailed function counts.

            Conceptually, the FunctionCounts returned can be thought of as a
        tuple of (count, path_and_function_name) tuples.

            `inclusive` matches the semantics of callgrind. If True, the counts
        include instructions executed by children. `inclusive=True` is useful
        for identifying hot spots in code; `inclusive=False` is useful for
        reducing noise when diffing counts from two different runs. (See
        `CallgrindStats.delta(...)` for more details)
        """
        return self.stmt_inclusive_stats if inclusive else self.stmt_exclusive_stats

    def counts(self, *, denoise: bool = False) -> int:
        """Returns the total number of instructions executed.

        See `FunctionCounts.denoise()` for an explation of the `denoise` arg.
        """
        stats = self.stmt_exclusive_stats
        return (stats.denoise() if denoise else stats).sum()

    # FIXME: Once 3.7 is the minimum version, type annotate `other` per PEP 563
    def delta(
        self,
        other,  # type: CallgrindStats
        inclusive: bool = False,
    ) -> FunctionCounts:
        """Diff two sets of counts.

            One common reason to collect instruction counts is to determine the
        the effect that a particular change will have on the number of instructions
        needed to perform some unit of work. If a change increases that number, the
        next logical question is "why". This generally involves looking at what part
        if the code increased in instruction count. This function automates that
        process so that one can easily diff counts on both an inclusive and
        exclusive basis.
        """
        return self.stats(inclusive=inclusive) - other.stats(inclusive=inclusive)

    def as_standardized(self) -> "CallgrindStats":
        return CallgrindStats(
            work_spec=self.work_spec,
            work_metadata=self.work_metadata,
            number_per_run=self.number_per_run,
            built_with_debug_symbols=self.built_with_debug_symbols,
            stmt_inclusive_stats=self.stmt_inclusive_stats.as_standardized(),
            stmt_exclusive_stats=self.stmt_exclusive_stats.as_standardized(),

            # `as_standardized` will change symbol names, so the contents will
            # no longer map directly to `callgrind.out`
            stmt_callgrind_out=None,
        )
