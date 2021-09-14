""" Python template for Timer methods.

This template will replace:
    `SETUP_TEMPLATE_LOCATION`
    `STMT_TEMPLATE_LOCATION`
      and
    `TEARDOWN_TEMPLATE_LOCATION`
sections with user provided statements.
"""
import timeit
import typing

import torch


# Note: The name of this class (PythonTemplate) is a magic word in compile.py
class PythonTemplate:

    @staticmethod
    def call(n_iter: int) -> None:
        # SETUP_TEMPLATE_LOCATION

        for _ in range(n_iter):
            # STMT_TEMPLATE_LOCATION
            pass

        # TEARDOWN_TEMPLATE_LOCATION

    @staticmethod
    def measure_wall_time(
        n_iter: int,
        n_warmup_iter: int,
        cuda_sync: bool,
        timer: typing.Callable[[], float] = timeit.default_timer,
    ) -> float:
        # SETUP_TEMPLATE_LOCATION

        for _ in range(n_warmup_iter):
            # STMT_TEMPLATE_LOCATION
            pass

        if cuda_sync:
            torch.cuda.synchronize()
        start_time = timer()

        for _ in range(n_iter):
            # STMT_TEMPLATE_LOCATION
            pass

        if cuda_sync:
            torch.cuda.synchronize()

        result = timer() - start_time

        # TEARDOWN_TEMPLATE_LOCATION

        return result

    @staticmethod
    def collect_callgrind(n_iter: int, n_warmup_iter: int) -> None:
        # SETUP_TEMPLATE_LOCATION

        for _ in range(n_warmup_iter):
            # STMT_TEMPLATE_LOCATION
            pass

        torch._C._valgrind_toggle()
        for _ in range(n_iter):
            # STMT_TEMPLATE_LOCATION
            pass

        torch._C._valgrind_toggle_and_dump_stats()

        # TEARDOWN_TEMPLATE_LOCATION
