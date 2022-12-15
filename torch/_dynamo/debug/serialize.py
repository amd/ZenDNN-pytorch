import textwrap

import torch

from .. import config


def codegen_graph_module(gm):
    """
    Codegen a nn Module from a Fx graph module
    """
    from torch.nn.modules.module import _addindent

    tab = " " * 4

    model_str = textwrap.dedent(
        """
        from torch.nn import *
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
        """
    )

    for module_name, module in gm.named_children():
        module_str = f"{module.__repr__()}"
        # module should be a core torch.nn.Module, so all parameters
        # should be on the same device.
        example_param = next(module.parameters(), None)
        if example_param is not None and example_param.is_cuda:
            module_str = f"{module_str}.cuda()"
        model_str += f"{tab*2}self.{module_name} = {module_str}\n"

    for buffer_name, buffer in gm._buffers.items():
        if buffer is None:
            continue
        if torch.is_floating_point(buffer):
            tensor_str = f"torch.randn({list(buffer.shape)}, dtype={buffer.dtype})"
        else:
            tensor_str = (
                f"torch.randint(1, size={list(buffer.shape)}, dtype={buffer.dtype})"
            )
        if buffer.is_cuda:
            tensor_str = f"{tensor_str}.cuda()"
        model_str += f"{tab*2}self.register_buffer('{buffer_name}', {tensor_str})\n"

    for param_name, param in gm._parameters.items():
        if param is None:
            continue
        tensor_str = (
            f"torch.nn.Parameter(torch.randn({list(param.shape)}, dtype={param.dtype}))"
        )
        if param.is_cuda:
            tensor_str = f"{tensor_str}.cuda()"
        model_str += f"{tab*2}self.{param_name} = {tensor_str}\n"

    model_str += f"{_addindent(gm.code, 4)}\n"
    return model_str


def codegen_inputs(args):
    s = f"args = {[(tuple(a.shape), tuple(a.stride()), a.dtype, a.device.type, a.requires_grad) for a in args]!r}\n"
    s += "args = [rand_strided(sh, st, dt, dev).requires_grad_(grad) for (sh, st, dt, dev, grad) in args]\n"
    return s


INDUCTOR_IMPORT = f"""
from {config.inductor_import}.compile_fx import compile_fx_inner
"""


def codegen_imports():
    return textwrap.dedent(
        f"""
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx
import {config.dynamo_import}
from {config.dynamo_import}.testing import rand_strided
# from {config.dynamo_import}.debug_utils import run_fwd_maybe_bwd
# from {config.dynamo_import}.debug_utils import same_two_models
{INDUCTOR_IMPORT}
"""
    )


class ReproCodegen:
    pass


class ReproInductorCodegen(ReproCodegen):
    def codegen_module_instantiation(self):
        return "mod = make_fx(Repro())(*args)\n"

    def codegen_opt_mod_call(self):
        return textwrap.dedent(
            f"""
    compiled = compile_fx_inner(mod, args)
    ref = compiled(args)
    """
        )

    def codegen_call():
        return self.codegen_module_instantiation() + self.codegen_opt_mod_call()
