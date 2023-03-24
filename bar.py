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

def def_impl(ns, opname, devices=('cpu', 'cuda')):
    for device in devices:
        # Only ones supported for now.
        # In theory we could support all.
        assert device in ('cpu', 'cuda', 'meta')

    backend_to_key = {
        'cpu': 'CPU',
        'cuda': 'CUDA',
        'meta': 'Meta',
    }

    def inner(f):
        lib = get_library(ns)
        for device in devices:
            key = backend_to_key[device]
            library.impl(lib, opname, key)(f)
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

def parse_schema(schema):
    # TODO: schema parsing validation
    ns, rest = schema.split('::')
    opname, *_ = rest.split('(')
    return ns, rest, opname

class custom_op:
    def __init__(self, schema):
        # TODO: schema parsing validation
        ns, rest, opname = parse_schema(schema)

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

    def def_autograd_old(self, save_fn, call_fn):
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

    def def_autograd(self, dct):
        # TODO(rzou): this is just for the demo, lol
        class Generated(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs):
                guard = _C._AutoDispatchBelowAutograd()
                output = self(*inputs)
                del guard
                ctx.save_for_backward(inputs[0])
                return output

            @staticmethod
            def backward(ctx, *grads):
                return dct['x'](grads[0], ctx.saved_tensors[0])

        lib = get_library(self.namespace)
        library.impl(lib, self.opname, 'Autograd')(Generated.apply)

    def def_meta(self, f):
        return def_meta(self.namespace, self.opname)(f)

def preserve_ops(schema):
    def inner(f):
        op = custom_op(schema)
        op.def_impl(f)
        op.def_meta(f)
        return op
    return inner

# # Needs FRAGMENT library.
# def custom_op_from_cpp(schema):
#     ns, rest, opname = parse_schema(schema)

# ==========================
# Sample code
# ==========================

import numpy as np

# ================================================================
# The new custom operator API is a way to create custom operators
# (of class custom_op) and register interactions with various
# high-level subsystems, like autograd, shape propagation
# (meta/fake/dynamic shapes), and functionalization, in a safe
# and blessed manner.
#
# There are two entry points into the new custom operator API:
# 1. You have an operator that consists of only PyTorch operations.
# 2. You have an operator that calls some custom C++/CUDA/etc code.

# ================================================================
# For the first use case:
# if you want to preserve a set of PyTorch ops through
# torch.compile or export, then use the @preserve_ops decorator.
# This will create a custom operator wrapping your sequence
# of PyTorch ops.
@preserve_ops('my_new_lib::bar(Tensor x) -> Tensor')
def bar(x):
    return x.sin().sin()

def f(x):
    return bar(x) * 2

x = torch.randn(3)
gm = make_fx(f, tracing_mode='symbolic')(x)
print(gm.code)

# def forward(self, x_1):
#     bar = torch.ops.my_new_lib.bar.default(x_1);  x_1 = None
#     mul = torch.ops.aten.mul.Tensor(bar, 2);  bar = None
#     return mul

# ================================================================
# Need additional things, like a backwards formula?
# The preserve_ops transforms the function into a `custom_op`
# class, where you can register additional things as needed.
#
# TODO: we could theortically generate the autograd formula
# for the user under certain conditions... but I'm not convinced
# this will be useful, and it may have implications for the schema.
# Shout if you have a concrete use case.

# Example, but keep reading to see how this works.
# bar.def_autograd(...)

# ================================================================
# For the second use case, use the custom_op decorator to
# directly create a custom op and register implementations for it.
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
# how to actually call the backward operator. Our API for doing it
# is similar to derivatives.yaml.

# TODO: this isn't good enough for saving things like x.shape,
# unless we make it automagical...
foo.def_autograd({
    'x': lambda grad, x: foo_backward(grad, x),
})

# NB: This registers an autograd formula for ALL Autograd* keys, which
# may not be desired, but is what most users want. If the user wants to do
# something wild, then they are able to use the low-level torch.library API
# directly.

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

# def forward(self, x_1, gy_1):
#     foo = torch.ops.my_new_lib.foo.default(x_1)
#     is_same_size = torch.ops.aten.is_same_size.default(foo, gy_1);  foo = None
#     foo_backward = torch.ops.my_new_lib.foo_backward.default(gy_1, x_1);  x_1 = None
#     return gy_1
