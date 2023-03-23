import torch
import torch.library as library
import torch._C as _C
from torch.fx.experimental.proxy_tensor import make_fx

USER_LIBS = {}

def get_library(ns):
    if ns not in USER_LIBS:
        # TODO: doesn't make sense to me you can only create a library once
        USER_LIBS[ns] = library.Library(ns, "DEF")
    return USER_LIBS[ns]

def def_impl(ns, opname):
    def inner(f):
        lib = get_library(ns)
        library.impl(lib, opname, 'CompositeExplicitAutograd')(f)
        return f
    return inner

def def_meta(ns, opname):
    def inner(f):
        lib = get_library(ns)
        library.impl(lib, opname, 'Meta')(f)
        return f
    return inner

def old_def_autograd(ns, opname):
    def inner(f):
        # TODO: check is autograd.Function
        # assert isinstance(f, torch.autograd.Function)
        def even_inner(*args, **kwargs):
            guard = _C._AutoDispatchBelowAutograd()
            return f(*args, **kwargs)

        lib = get_library(ns)
        library.impl(lib, opname, 'Autograd')(even_inner)
        return f
    return inner

class custom_op:
    def __init__(self, schema):
        # TODO: schema parsing validation
        ns, rest = schema.split('::')
        opname, *_ = rest.split('(')

        lib = get_library(ns)
        lib.define(rest)

        self.namespace = ns
        self.opname = opname
        op_ns = getattr(torch.ops, ns)
        op = getattr(op_ns, opname)
        self.dispatcher_op = op

    def __call__(self, *args, **kwargs):
        result = self.dispatcher_op(*args, **kwargs)
        return result

    def def_impl(self, f):
        return def_impl(self.namespace, self.opname)(f)

    def def_autograd(self, save_fn, call_fn):
        # TODO: what else do people specify?
        # Make sure the autograd.Function options are available...
        # - mark_dirty
        # - mark_non_differentiable
        # - set_materialize_grads
        class Generated(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs):
                guard = _C._AutoDispatchBelowAutograd()
                output = self(*inputs)
                del guard
                saved = save_fn(inputs, output)
                # TODO: enabling saving more than just tensors
                ctx.save_for_backward(*saved)
                return output

            @staticmethod
            def backward(ctx, *grads):
                return call_fn(grads, ctx.saved_tensors)

        def even_inner(*args, **kwargs):
            return Generated.apply(*args, **kwargs)

        lib = get_library(self.namespace)
        library.impl(lib, self.opname, 'Autograd')(even_inner)

    def def_meta(self, f):
        return def_meta(self.namespace, self.opname)(f)

# ==========================
# Sample code
# ==========================

import numpy as np

# ================================================================
# Create a custom op and register an implementation for it.
# Also, you need a meta formula to actually do symbolic tracing,
# so we'll register that as well.

foo = custom_op('my_new_lib::foo(Tensor x) -> Tensor')

def foo_impl(x):
    res = np.sin(x.cpu().numpy())
    return torch.from_numpy(res).to(x.device)

def foo_meta(x):
    return x.sin()

foo.def_impl(foo_impl)
foo.def_meta(foo_meta)

# ================================================================
# Let's assume the user has a custom backward pass that is also some
# sort of fused kernel.
#
# Here we construct the backward operator.
foo_backward = custom_op('my_new_lib::foo_backward(Tensor gy, Tensor x) -> Tensor')

def foo_backward_impl(gy, x):
    x_np = x.cpu().numpy()
    grad_out_np = grad_out.cpu().numpy()
    grad_x_np = grad_out_np * np.cos(x_np)
    return torch.from_numpy(grad_out_np).to(x.device)

def foo_backward_meta(gy, x):
    return gy * x.cos()

foo_backward.def_impl(foo_backward_impl)
foo_backward.def_meta(foo_backward_meta)

# ================================================================
# To stitch together the forward and backward, we need to specify
# how to actually call the backward operator.

def foo_save_for_backward(inputs, output):
    x, = inputs
    return [x]

def foo_call_backward(grads, saved):
    grad_output, = grads
    x, = saved
    return foo_backward(grad_output, x)

foo.def_autograd(foo_save_for_backward, foo_call_backward)

# ================================================================
# Check that foo, foo_backward get traced out.

def f(x, gy):
    y = foo(x)
    gx, = torch.autograd.grad(y, x, gy)
    return gy

x = torch.randn([1], requires_grad=True)
gy = torch.randn([1])
gm = make_fx(f, tracing_mode='symbolic')(x, gy)
print(gm.code)

"""
def forward(self, x_1, gy_1):
    foo = torch.ops.my_new_lib.foo.default(x_1)
    is_same_size = torch.ops.aten.is_same_size.default(foo, gy_1);  foo = None
    foo_backward = torch.ops.my_new_lib.foo_backward.default(gy_1, x_1);  x_1 = None
    return gy_1
"""
