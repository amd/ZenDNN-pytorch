import torch.library
from torch import Tensor
from torch.autograd import Function

_test_lib_def = torch.library.Library("_inductor_test", "DEF")
_test_lib_def.define("realize(Tensor self) -> Tensor")
_test_lib_def.define("bad_clone(Tensor self) -> Tensor")
_test_lib_def.define("inaccurate_clone(Tensor self) -> Tensor")

_test_lib_impl = torch.library.Library("_inductor_test", "IMPL")
for dispatch_key in ("CPU", "CUDA", "Meta"):
    _test_lib_impl.impl("realize", lambda x: x.clone(), dispatch_key)
    _test_lib_impl.impl("bad_clone", lambda x: x.clone(), dispatch_key)
    _test_lib_impl.impl("inaccurate_clone", lambda x: x.clone(), dispatch_key)


class Realize(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ops._inductor_test.realize(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BadClone(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ops._inductor_test.bad_clone(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class InaccurateClone(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ops._inductor_test.inaccurate_clone(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def realize(x: Tensor) -> Tensor:
    return Realize.apply(x)


def bad_clone(x: Tensor) -> Tensor:
    return BadClone.apply(x)


def inaccurate_clone(x: Tensor) -> Tensor:
    return InaccurateClone.apply(x)
