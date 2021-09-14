import dataclasses
import typing

import torch


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
