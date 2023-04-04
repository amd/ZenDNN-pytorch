import warnings

from torch._dynamo.comptime import comptime
from torch.fx.experimental.symbolic_shapes import constrain_range

def add_inline_constraint(symbol, min, max):
    """
    Add min/max constraint on the intermediate symbol at tracing time
    """

    @comptime
    def _(ctx): 
        if str(symbol) not in ctx.locals():
            raise RuntimeError(f"symbol {symbol} could not be found")

        node = ctx.get_local(str(symbol)).as_proxy().node
        constrain_range(node.meta["example_value"], min=min, max=max)
        ctx.graph().call_function(constrain_range, (node,), {"min": min, "max": max})

def add_inline_size_constraint(symbol, min, max):
    """
    Add min/max constraint on the intermediate symbol which will be used as a size
    """

    if min <= 2:
        warnings.warn("Trying to set min size as <= 2 which is not allowed "
        + "because we specialize on sizes of 0/1. Setting min to 2.")
        min = 2

    return add_inline_constraint(symbol, min, max)
