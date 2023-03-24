from functools import partial

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional, _wrap_all_tensors_to_functional, functionalize
from torch._functorch.aot_autograd import create_joint
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
    unwrap_proxy,
)
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)
from torch.utils._pytree import tree_flatten
from ._cond import _has_potential_branch_input_alias, _has_potential_branch_input_mutation, UnsupportedAliasMutationException


map = HigherOrderOperator("map")

class MapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fw_graph, joint_graph, args_spec, *flat_args):
        ctx.save_for_backward(*flat_args)
        ctx._joint_graph = joint_graph
        ctx._args_spec = args_spec
        _ = torch._C._AutoDispatchBelowAutograd()
        return (*map(fw_graph, *flat_args), )
    
    @staticmethod
    def backward(ctx, *flat_grads):
        xs, args = pytree.tree_unflatten(ctx.saved_tensors, ctx._args_spec)
        _ = torch._C._AutoDispatchBelowAutograd()
        def bw_fn(xs, *args):
            xs_primals, xs_grads = xs
            flat_xs, _ = pytree.tree_flatten(xs_primals)
            flat_args, _ = pytree.tree_flatten((flat_xs + list(args), xs_grads))
            _, grads = ctx._joint_graph(*flat_args)
            return grads
        grads = map(bw_fn, (xs, [grad for grad in flat_grads if grad is not None]), *args)
        return None, None, None, *grads

def trace_map(proxy_mode, func_overload, f, xs, *args):
    if not all(isinstance(o, torch.Tensor) for o in args):
        raise ValueError("map() positional args must be a list of tensors")
    
    def check_tensor(arg):
        if not isinstance(arg, torch.Tensor):
            raise ValueError(f"map() operands must be a list of tensors got {arg}")
        if len(arg.shape) == 0 or arg.shape[0] == 0:
            raise ValueError(f"map() cannot be traced with scalar tensors or zero dimension tensors got {arg.shape}")
        
    pytree.tree_map(check_tensor, xs)
    flat_xs, _ = pytree.tree_flatten(xs)
    leading_dim_size = flat_xs[0].shape[0]

    xs_pytrees = _unstack_pytree(xs)
    # Note: f is lowered to a fx.GraphModule in post_autograd make_fx tracing
    if not isinstance(f, torch.fx.GraphModule):
        with disable_proxy_modes_tracing():
            body_graph = make_fx(f)(xs_pytrees[0], *args)
    else:
        body_graph = f

    next_name = None
    i = 0
    while not next_name:
        candidate = f"body_graph_{i}"
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate

    proxy_mode.tracer.root.register_module(next_name, body_graph)
    node_args = (body_graph, xs, *args)
    proxy_args = pytree.tree_map(partial(unwrap_proxy, proxy_mode), node_args)
    out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {},
                                               name="map")
    example_outs = body_graph(xs_pytrees[0], *args)
    expanded_outs = pytree.tree_map(lambda t: t.expand(leading_dim_size, *t.shape) if t is not None else t, example_outs)
    # # Implementation notes: we need to use new_empty() + copy_() here instead of stack() directly
    # # because stack([...]) takes a fixed size list which will specialize dynamic shape here.
    # # Meanwhile we want to preserve the looped over dimension as symbolic shape, such that:
    # # ys: Tensor[s0, ...] = map(xs: Tensor[s0, ...], *args)
    # out = outs[0].new_empty([xs.shape[0], *outs[0].shape])
    # out.copy_(torch.stack(outs))
    return track_tensor_tree(expanded_outs, out_proxy, constant=None, tracer=proxy_mode.tracer)

def _unstack_pytree(xs):
    flat_xs, inspec = pytree.tree_flatten(xs)
    assert all([isinstance(xs, torch.Tensor) for xs in flat_xs]), f"Leaves of xs must be Tensor {flat_xs}"
    assert all([xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs]), f"Leaves of xs must have same leading dimension size {flat_xs}"
    a = list(zip(*flat_xs))
    pytrees = []
    for tuple in a:
        pytrees.append(pytree.tree_unflatten(tuple, inspec))
    return pytrees

def _stack_pytree(pytrees):
    flat_out = []
    out_spec = None
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        flat_out.append(flat_pt)
    b = list(zip(*flat_out))
    stacked_out = []
    for leaves in b:
        if all([leave is not None for leave in leaves]):
            stacked_out.append(torch.stack(leaves))
        else:
            stacked_out.append(None)
    return pytree.tree_unflatten(stacked_out, out_spec)

@map.py_impl(DispatchKey.CUDA)
@map.py_impl(DispatchKey.CPU)
def map_impl(f, xs, *args):
    mode = _get_current_dispatch_mode()
    assert (mode is None), "Mode should never be enabled for CPU/CUDA keyOne of the differentiated Tensors"
    pytrees = []
    for inp in _unstack_pytree(xs):
        pytrees.append(f(inp, *args))
    return _stack_pytree(pytrees)

# Create a GraphModule for fn, which takes flattend input and produces flattend output.
# The tree specs of original functions's output is also returned.
def _make_flattend_graph(fn, *args, **kwargs):
    flat_in, in_spec = pytree.tree_flatten((args, kwargs))
    out_spec = [None]
    def flat_fn(*flat_args):
        args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
        unflattened_out = fn(*args, **kwargs)
        flat_out, tmp_out_spec = pytree.tree_flatten(unflattened_out)
        out_spec[0] = tmp_out_spec
        return flat_out
    flat_graph = make_fx(flat_fn)(*flat_in)
    return flat_graph, out_spec[0]

def _make_flattend_joint_graph(flattend_fw_graph, flat_args, flat_grads):
    def fw_with_masks(*args):
        flat_out = flattend_fw_graph(*args)
        return flat_out, [True if isinstance(ret, torch.Tensor) and ret.requires_grad else False for ret in flat_out]
    
    joint_fn = create_joint(fw_with_masks)
    flat_graph, out_spec = _make_flattend_graph(joint_fn, flat_args, flat_grads)
    return flat_graph, out_spec


@map.py_impl(DispatchKey.Autograd)
def map_autograd(f, xs, *args):

    with disable_proxy_modes_tracing():
        example_xs = _unstack_pytree(xs)[0]
        example_flat_out, _ = pytree.tree_flatten(f(example_xs, *args))
        example_grad = [torch.ones_like(out) for out in example_flat_out if out is not None and out.requires_grad]


    fw_graph, fw_out_spec = _make_flattend_graph(f, example_xs, *args)
    fw_graph.print_readable()

    # Run tests to verify correctness of following lines
    bw_graph, out_spec = _make_flattend_joint_graph(fw_graph, example_xs, *args)
    bw_graph.print_readable()
    print(out_spec)

    flat_args, args_spec = pytree.tree_flatten((xs, *args))
    flat_out = MapAutogradOp.apply(fw_graph, bw_graph, args_spec, *flat_args)
    return pytree.tree_unflatten(flat_out, fw_out_spec)


@map.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(f, xs, *args):
    print("map forward proxy torch dispatch")
    mode = _get_current_dispatch_mode()
    assert (mode is not None), "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        res = trace_map(mode, map, f, xs, *args)
    return res


@map.py_impl(FakeTensorMode)
def map_fake_tensor_mode(f, xs, *args):
    print("fake tensor mode")
    leading_dims = pytree.tree_map(lambda t: t.shape[0], xs)
    xs_pytree = _unstack_pytree(xs)
    example_out = f(xs_pytree[0], *args)
    return pytree.tree_map(lambda t: t.expand(leading_dims[0], *t.shape), example_out)

@map.py_impl(torch._C._functorch.TransformType.Functionalize)
def map_functionalize(interpreter, f, xs, *args):
    print("map_functionalize")
    """
    Functionalization implementation for torch.map. Currently:
      1. We don't allow any input mutation inside the map function
      2. Our check for above condition is not exhaustive
    """
    reapply_views = interpreter.functionalize_add_back_views()
    mode = 'mutations_and_views' if reapply_views else 'mutations'
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    unwrapped_args = _unwrap_all_tensors_from_functional(args, reapply_views=reapply_views)

    functional_map_fn = functionalize(f, remove=mode)

    with interpreter.lower():
        inputs = (unwrapped_xs,) + unwrapped_args
        if _has_potential_branch_input_mutation(functional_map_fn, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is mutating the input!"
            )

        if _has_potential_branch_input_alias(functional_map_fn, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is aliasing the input!"
            )

        map_return = map(functional_map_fn, unwrapped_xs, *unwrapped_args)
        return _wrap_all_tensors_to_functional(map_return, level=interpreter.level())

# TODO(voz) Make this automatic for keys, this is very ugly atm
map.fallthrough(DispatchKey.PythonDispatcher)
map.fallthrough(DispatchKey.PythonTLSSnapshot)
map.fallthrough(DispatchKey.ADInplaceOrView)
map.fallthrough(DispatchKey.BackendSelect)
map.fallthrough(DispatchKey.AutocastCPU)
