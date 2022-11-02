# Owner(s): ["module: dynamo"]
import copy
import functools
import os
import unittest

import torch
from functorch.compile import aot_module_simplified
from torch._dynamo import disable

has_torch_xla = True
try:
    import torch._dynamo.optimizations.torchxla_integration as integration
except ImportError:
    has_torch_xla = False

import torch.utils._pytree as pytree
from torch import fx, nn
from functorch.compile import make_boxed_compiler


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def forward(self, x, y):
        return x + y

    def get_random_inputs(self):
        return (torch.randn(10), torch.randn(10))


class MatmulModule(nn.Module):
    def __init__(self):
        super(MatmulModule, self).__init__()

    def forward(self, x, y):
        return x @ y

    def get_random_inputs(self):
        return (torch.randn(5, 100), torch.randn(100, 5))


class LinearModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def get_random_inputs(self):
        return (torch.randn(2, 10),)

class MaxPoolModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)

    def get_random_inputs(self):
        return (torch.randn(2, 3, 10, 10),)


class ModuleInplaceUpdate(nn.Module):
    def __init__(self):
        super(ModuleInplaceUpdate, self).__init__()

    def forward(self, a, b):
        a.sub_(b)
        return b - 1, b + 1

    def get_random_inputs(self):
        return (torch.randn(10), torch.randn(10))


def allclose(expected, actual):
    def unwrap(cont):
        if isinstance(cont, (list, tuple)) and len(cont) == 1:
            return cont[0]
        return cont

    expected = unwrap(expected)
    actual = unwrap(actual)

    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        return torch.allclose(expected, actual)
    elif isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        return len(expected) == len(actual) and all(
            torch.allclose(a, b) for a, b in zip(expected, actual)
        )
    else:
        raise RuntimeError("Unexpected types")


@functools.lru_cache(None)
def should_run_torchxla_tests():
    """
    Run the tests if torch_xla is available and number of gpu devices is specified.
    """
    gpu_device_specified = int(os.environ.get("GPU_NUM_DEVICES", "0")) > 0
    return has_torch_xla and gpu_device_specified

def maybe_skip(f):
    return unittest.skipIf(
        not should_run_torchxla_tests(),
        "Skip the tests since torch_xla is not available or XLA devices are not specified",
    )(f)

def make_reuse_graph_test(module_class, niter=100):
    @maybe_skip
    def test_wrapper(self):
        import torch_xla.core.xla_model as xm

        xla_dev = xm.xla_device()
        mod = module_class()
        xla_module = copy.deepcopy(mod).to(device=xla_dev)
        inputs = mod.get_random_inputs()
        optimized_mod = integration.extract_compiled_graph(
            fx.symbolic_trace(mod), inputs
        )

        for i in range(niter):
            rand_args = mod.get_random_inputs()
            orig_dev = rand_args[0].device
            rand_args_copy = copy.deepcopy(rand_args)

            # Can not simply call
            #   expected = mod(*rand_args)
            # Since we need use xla to calculate expected results
            xla_inputs = tuple(
                copy.deepcopy(inp).to(device=xla_dev) for inp in rand_args
            )
            xla_out = xla_module(*xla_inputs)
            # copy xla_inputs back to rand_args since the model may inplace update
            # the arguments
            rand_args = tuple(inp.to(device=orig_dev) for inp in xla_inputs)
            expected = pytree.tree_map(lambda o: o.to(device=orig_dev), xla_out)

            actual = optimized_mod(*rand_args_copy)

            if not allclose(expected, actual):
                print(
                    f"Incorrect results at iter {i}. expected\n{expected}, actual\n{actual}"
                )
                self.assertTrue(False)

            # make sure arguments match after calling the model forward method
            # to handle inplace updates.
            if not allclose(rand_args, rand_args_copy):
                print(
                    f"Incorrect updated arguments at iter {i}. expected\n{rand_args}, actual\n{rand_args_copy}"
                )
                self.assertTrue(False)

    return test_wrapper

already_on_xla = True
check_inplace_update = os.environ.get("check_inplace_update", "0") == "1"

def training_compiler(gm, example_inputs):
    @make_boxed_compiler
    @disable
    def fw_compiler(graph, inputs, *args, **kwargs):
        return integration.extract_compiled_graph(graph, inputs, already_on_xla=already_on_xla, check_inplace_update=check_inplace_update)

    @make_boxed_compiler
    @disable
    def bw_compiler(graph, inputs, *args, **kwargs):
        return integration.extract_compiled_graph(graph, inputs, already_on_xla=already_on_xla, check_inplace_update=check_inplace_update)

    return aot_module_simplified(gm, fw_compiler=fw_compiler, bw_compiler=bw_compiler)


def model_iter_fn_train(mod, inputs, collect_outputs=True):
    outputs = mod(*inputs)
    loss = outputs.mean()
    loss.backward()

def make_training_test(model_cls):
    @maybe_skip
    def test_wrapper(self):
        import torch_xla.core.xla_model as xm
        import torch_xla
        xla_dev = xm.xla_device()
        model = model_cls()
        inputs = model.get_random_inputs()

        if already_on_xla:
            model = model.to(device=xla_dev)
            inputs = tuple(inp.to(device=xla_dev) for inp in inputs)

            # do baseline
            baseline_model = copy.deepcopy(model)
            baseline_inputs = copy.deepcopy(inputs)
            model_iter_fn_train(baseline_model, baseline_inputs)

        output = model(*inputs)
        if not isinstance(output, (tuple, list)):
            output = (output,)

        if already_on_xla:
            print(f"XLA IR Text: {torch_xla._XLAC._get_xla_tensors_text(output)}")
        compiler = training_compiler
        optimize_ctx = torch._dynamo.optimize(compiler, nopython=False)
        model_iter_fn = model_iter_fn_train
        optimized_model_iter_fn = optimize_ctx(model_iter_fn)

        actual_output = optimized_model_iter_fn(model, inputs)

    return test_wrapper

class TorchXLAReuseGraphTest(unittest.TestCase):
    test_basic = make_reuse_graph_test(BasicModule)
    test_matmul = make_reuse_graph_test(MatmulModule)
    test_linear = make_reuse_graph_test(LinearModule)
    test_inplace_update = make_reuse_graph_test(ModuleInplaceUpdate)

    test_training_linear = make_training_test(LinearModule)
    test_training_maxpool = make_training_test(MaxPoolModule)
