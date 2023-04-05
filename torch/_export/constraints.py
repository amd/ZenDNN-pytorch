from typing import Optional
import warnings

import torch._dynamo
from torch._dynamo.comptime import comptime
from torch.fx.experimental.symbolic_shapes import constrain_range


def add_inline_constraint(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Add min/max constraint on the intermediate symbol at tracing time
    """

    if torch._dynamo.is_compiling():
        @comptime
        def _(ctx):
            node = ctx.get_local("symbol").as_proxy().node
            min = ctx.get_local("min").as_python_constant()
            max = ctx.get_local("max").as_python_constant()
            constrain_range(node.meta["example_value"], min=min, max=max)
            ctx.graph().call_function(constrain_range, (node,), {"min": min, "max": max})
    else:
        constrain_range(symbol, min=min, max=max)

    return symbol

def add_inline_size_constraint(symbol, min: Optional[int] = 2, max: Optional[int] = None):
    """
    Add min/max constraint on the intermediate symbol which will be used as a size
    """

    if min is not None and min <= 2:
        if not torch._dynamo.is_compiling():
            warnings.warn("Unable to set min size to be <= 2 because we specialize on 0/1 sizes.")
        min = 2

    return add_inline_constraint(symbol, min, max)
