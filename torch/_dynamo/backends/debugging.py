import functools
from importlib import import_module

from functorch.compile import min_cut_rematerialization_partition

import torch
from torch._functorch.compilers import ts_compile
from .common import aot_autograd
from .registry import register_debug_backend as register_backend

"""
This file contains TorchDynamo backends intended for debugging uses.
"""


@register_backend
def eager(gm, fake_tensor_inputs):
    return gm


@register_backend
def eager_debug(gm, fake_tensor_inputs):
    from torch._subclasses.schema_check_mode import SchemaCheckMode

    # We could add more debugging bits here.
    # Right now, this backend can be used to check for and error on
    # custom dispatcher ops that have incorrect schemas.
    def inner(*args):
        with SchemaCheckMode():
            return torch.fx.Interpreter(gm).run(*args)

    return inner


@register_backend(name="ts")
def torchscript(gm, fake_tensor_inputs):
    return torch.jit.script(gm)


# used boxed call to discard inputs when they are no longer needed
def boxed_nop(fx_g, example_inputs):
    def run(args):
        return torch.fx.Interpreter(fx_g).boxed_run(args)

    run._boxed_call = True
    return run


# Useful for debugging purpose
# aot_eager uses AOT Autograd backend with nop compiler. It is helpful in debugging.
aot_eager = aot_autograd(fw_compiler=boxed_nop)
register_backend(name="aot_eager", compiler_fn=aot_eager)


# Uses TorchInductor AOT Autograd decomps and partitioner to isolate aot vs
# inductor problems.
# aot_eager_decomp_partition just replaces the inductor compiler with nop to help
# isolate inductor vs aot_eager errors
decomp_table = lambda: import_module("torch._inductor.compile_fx").select_decomp_table()
decomp_table = decomp_table()

aten = torch.ops.aten

# These are all decomps triggered in AlbertForQuestionAnswering. So, I copied
# the decomps here and deleted every decomp but one. The goal is to find the one
# decomp that causes accuracy error. _softmax, native_layer_norm seems to be culprit.
deleted = [
    # aten._softmax.default,
    aten.native_layer_norm.default,

    # Following dont seem to be relevant
    aten._log_softmax.default,
    aten._log_softmax_backward_data.default,
    aten._softmax_backward_data.default,
    aten._to_copy.default,
    aten._unsafe_view.default,
    aten.cat.default,
    aten.clamp.default,
    aten.conj_physical.default,
    aten.embedding_dense_backward.default,
    aten.full_like.default,
    aten.masked_fill.Scalar,
    aten.native_layer_norm_backward.default,
    aten.new_zeros.default,
    aten.nll_loss_backward.default,
    aten.nll_loss_forward.default,
    aten.ones.default,
    aten.rsub.Scalar,
    aten.t.default,
    aten.tanh_backward.default,
    aten.transpose.int,
    aten.zeros_like.default,
]
for k in deleted:
    decomp_table.pop(k)

aot_eager_decomp_partition = aot_autograd(
    # these are taken from memory_efficient_fusion()
    fw_compiler=boxed_nop,
    bw_compiler=boxed_nop,
    # NB: lambda here is to delay import of inductor
    decompositions=decomp_table,
    partition_fn=functools.partial(
        min_cut_rematerialization_partition, compiler="inductor"
    ),
)
register_backend(
    name="aot_eager_decomp_partition", compiler_fn=aot_eager_decomp_partition
)

# AOT Autograd with torchscript backend. Default partitioner.
# aot_ts uses torchscript backend. We can use this with both nnc and nvfuser
# by using the relevant fuser with torch.jit.fuser(...)
aot_ts = aot_autograd(fw_compiler=ts_compile)
register_backend(name="aot_ts", compiler_fn=aot_ts)

# These buggy backends are used for inducing bugs so that we can test
# our repro extraction / minifier scripts


class ReluCompileError(Exception):
    pass


class TestingOnlyCompileError(Exception):
    pass


@register_backend
def relu_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            raise ReluCompileError()
    return gm


@register_backend
def relu_runtime_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch._assert
            node.args = (False, "ReluRuntimeError")
    gm.recompile()
    return gm


@register_backend
def relu_accuracy_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch.add
            node.args = (node.args[0], 1)
    gm.recompile()

    return gm


@register_backend
def non_leaf_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    # Require at least one non-trivial thing in the graph,
    # see https://github.com/pytorch/pytorch/issues/102898
    for node in gm.graph.nodes:
        if node.op == "call_function":
            break
    else:
        return gm
    for t in example_inputs:
        if not t.is_leaf:
            raise TestingOnlyCompileError()
    return gm
