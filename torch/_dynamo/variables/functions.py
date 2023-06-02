import abc
import enum
import functools
import inspect
import itertools
import types
from typing import Dict, List

import torch

from .. import variables
from ..allowed_functions import is_allowed, is_builtin_callable
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import istensor, istype, make_cell
from .base import typestr, VariableTracker


def wrap_bound_arg(tx, val, options, source=None):
    # Source propagation is best effort since not every object we encounter has a source to begin with.
    assert (
        "source" not in options
    ), "Source needs to be separate from options due to recursive calls for lists/dicts"

    if isinstance(val, dict):
        return variables.ConstDictVariable(
            {
                k: wrap_bound_arg(tx, v, options, source=getattr(v, "source", None))
                for k, v in val.items()
            },
            dict,
            **options,
        )
    elif isinstance(val, (tuple, list)):
        cls = variables.BaseListVariable.cls_for(type(val))
        return cls(
            [
                wrap_bound_arg(tx, x, options, source=getattr(x, "source", None))
                for x in val
            ],
            **options,
        )

    if variables.ConstantVariable.is_literal(val) or istype(
        val, (torch.Size, torch.device, torch.dtype)
    ):
        return variables.ConstantVariable(val, **options)
    elif is_builtin_callable(val):
        return variables.BuiltinVariable(val, source=source, **options)
    elif is_allowed(val):
        return variables.TorchVariable(val, source=source, **options)
    elif isinstance(val, types.FunctionType):
        return variables.UserFunctionVariable(val, source=source, **options)
    elif isinstance(val, enum.Enum):
        return variables.EnumVariable(val, source=source, **options)
    elif isinstance(val, (type, abc.ABCMeta)):
        return variables.UserDefinedClassVariable(val, source=source, **options)
    elif istensor(val):
        from torch._dynamo.variables.builder import VariableBuilder

        return VariableBuilder(tx, source=source)(val)
    else:
        assert isinstance(val, VariableTracker), typestr(val)
        return val


def wrap_args_kwargs(tx, result, options):
    for k, v in list(result.items()):
        if isinstance(v, (tuple, dict)):
            # args/kwargs
            result[k] = wrap_bound_arg(tx, v, options)


def init_cellvars(parent, result, code):
    closure_cells = dict()
    side_effects = parent.output.side_effects

    for name in code.co_cellvars:
        closure_cells[name] = side_effects.track_cell_new()
        if name in result:
            side_effects.store_cell(closure_cells[name], result.pop(name))

    return closure_cells


def _create_nested_fn(
    code, f_globals, name, defaults, closure, kwdefaults, annotations
):
    from types import FunctionType

    func = FunctionType(code, f_globals, name, defaults, closure)
    func.__kwdefaults__ = kwdefaults

    if isinstance(annotations, tuple):
        from itertools import pairwise

        annotations = dict(pairwise(annotations))

    # TypeError: __annotations__ must be set to a dict object
    assert annotations is None or isinstance(annotations, dict)
    func.__annotations__ = annotations

    return func


class BaseUserFunctionVariable(VariableTracker):
    def get_filename(self):
        return self.get_code().co_filename

    def get_name(self):
        return self.get_code().co_name

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        return tx.inline_user_function_return(
            self, list(self.self_args()) + list(args), kwargs
        )

    def num_parameters(self):
        return len(inspect.signature(self.get_function()).parameters)

    def closure_vars(self, tx):
        return {}


class UserFunctionVariable(BaseUserFunctionVariable):
    """Some unsupported user-defined global function"""

    def __init__(self, fn, is_constant=False, **kwargs):
        super().__init__(**kwargs)
        if getattr(fn, "_dynamo_marked_constant", False):
            print("MARKED CONSTANT", fn)
            # This method should be treated as a constant for the purposes of compilation
            self.is_constant = True
        else:
            # if "needs_unshard" in str(fn):
                # self.is_constant = True
            # else:
            self.is_constant = False

        assert isinstance(
            fn, (types.FunctionType, torch.jit.ScriptFunction)
        ), f"expected FunctionType found {typestr(fn)} {fn}"
        # unpack @torch._dynamo.optimize()(fn) wrapped function
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        # unpack torch.jit.script_if_tracing
        if inspect.getattr_static(fn, "__script_if_tracing_wrapper", False):
            fn = inspect.getattr_static(fn, "__original_fn", fn)
        self.fn: types.FunctionType = fn

    def self_args(self):
        return []

    def get_function(self):
        return self.fn

    def get_code(self):
        return self.fn.__code__

    def python_type(self):
        return types.FunctionType

    def has_self(self):
        return getattr(self.fn, "__self__", None) is not None

    def get_globals(self):
        return self.fn.__globals__

    def get_default_args(self):
        return self.fn.__defaults__
    
    def get_default_kwargs(self):
        return self.fn.__kwdefaults__

    def get_name(self):
        return self.fn.__name__

    def get_closure(self):
        return self.fn.__closure__

    def bind_args(self, parent, args, kwargs):
        assert not self.is_constant
        options = VariableTracker.propagate([self])
        tx = parent.output.root_tx
        wrap = functools.partial(wrap_bound_arg, tx=tx, options=options)

        defaults = self.get_default_args() or []
        defaults_sources = [
            None if self.source is None else DefaultsSource(self.source, idx)
            for idx, _ in enumerate(defaults)
        ]
        print("SOURCES", defaults_sources)
        fake_func = types.FunctionType(
            self.get_code(),
            self.get_globals(),
            self.get_name(),
            tuple(
                [
                    wrap(val=arg, source=source)
                    for arg, source in zip(defaults, defaults_sources)
                ]
            ),
            self.get_closure(),
        )
        if self.get_default_kwargs():
            kwdefaults_sources = {
                k: None
                if self.source is None
                else DefaultsSource(self.source, k, is_kw=True)
                for k in self.get_default_kwargs()
            }
            fake_func.__kwdefaults__ = {
                k: wrap(val=v, source=kwdefaults_sources[k])
                for k, v in self.get_default_kwargs().items()
            }

        print("BINDING?", fake_func, args, kwargs)
        bound = inspect.signature(fake_func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())
        print("BOUND ARGS RESULT", result)
        wrap_args_kwargs(tx, result, options)
        closure_cells = init_cellvars(parent, result, self.get_code())
        closure = self.fn.__closure__ or ()
        assert len(closure) == len(self.fn.__code__.co_freevars)
        for idx, name, cell in zip(
            itertools.count(), self.fn.__code__.co_freevars, closure
        ):
            if name == "__class__":
                source = AttrSource(self.source, "__class__") if self.source else None
                result[name] = variables.UserDefinedClassVariable(
                    cell.cell_contents,
                    source=source,
                )
            else:
                var = tx.match_nested_cell(name, cell)
                if var is not None:
                    # optimization for cleaner codegen
                    result[name] = var
                elif self.source:
                    from .builder import VariableBuilder

                    side_effects = parent.output.side_effects
                    if cell in side_effects:
                        out = side_effects[cell]
                    else:
                        closure_cell = GetItemSource(
                            AttrSource(self.source, "__closure__"), idx
                        )
                        closure_cell_contents = AttrSource(
                            closure_cell, "cell_contents"
                        )
                        contents_var = VariableBuilder(parent, closure_cell_contents)(
                            cell.cell_contents
                        )

                        if (
                            closure_cell_contents.name()
                            not in tx.mutated_closure_cell_contents
                        ):
                            # Optimistically don't allocate the cell, to
                            # reduce the number of side effects.  This is
                            # important for cond, as without it, any accesses
                            # to closures create side effects and cond doesn't
                            # support side effects.  If we're wrong and this
                            # closure cell gets written to, we will restart
                            # the analysis with this cell's name in the
                            # mutated list here
                            result[name] = contents_var
                            continue

                        # cells are written to with "cell_contents",
                        # so the source should just be the closure_cell, not its contents
                        out = side_effects.track_cell_existing(closure_cell, cell)
                        side_effects.store_cell(
                            out,
                            contents_var,
                        )

                    result[name] = out

                else:
                    unimplemented("inline with __closure__")

        return result, closure_cells

    def export_freevars(self, parent, child):
        pass

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        if self.is_constant:
            options = VariableTracker.propagate(self, args, kwargs.values())
            result = invoke_and_store_as_constant(
                tx, self.fn, self.get_name(), options, args, kwargs
            )
            return result

        return super().call_function(tx, args, kwargs)


class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(self, fn, obj, **kwargs):
        super().__init__(fn=fn, **kwargs)
        self.obj = obj

    def __str__(self):
        return f"{self.__class__.__name__}({self.fn}, {self.obj})"

    def self_args(self):
        return [self.obj]

    def python_type(self):
        return types.MethodType

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        if self.is_constant or getattr(self.obj, "_dynamo_marked_constant", False):
            return super().call_function(tx, [self.obj] + [*args], kwargs)
        
        # For nn.Module methods, redirecting to NNModuleVariable.call_method for optimized solution
        # rather than simple inlining. E.g, putting `call_method` op in FX graph for `forward` method
        # since we ensure `forward` of allowed modules can be traced by AOT safely.
        # Note this is not only for allowed modules, as user customized modules can extend from
        # allowed modules but using parent's `forward` method, which is also covered by this branch.

        # If we are tracing the higher order op, we want Dynamo to step inside
        # the module call so that Dynamo can see the underlying parameters and
        # buffers and raise them as inputs to the graph. The is_root_tracer
        # check bypasses the if condition for non-root tracers and directly
        # calls the super().call_function at the end, which is basically
        # equivalent of inlining the method.
        if tx.output.is_root_tracer() and isinstance(
            self.obj, variables.NNModuleVariable
        ):
            module_attr = getattr(self.fn, "__module__", "")
            if (
                module_attr is not None
                and module_attr.startswith("torch.nn.")
                or self.is_constant
            ):
                return self.obj.call_method(
                    tx, self.fn.__name__, args, kwargs, constant=self.is_constant
                ).add_options(self)
        return super().call_function(tx, args, kwargs)

    def num_parameters(self):
        return super().num_parameters() - 1

class PartialUserFunctionVariable(BaseUserFunctionVariable):
    def __init__(self, fn, is_constant=False, **kwargs):
        inner_fn = kwargs.pop("inner_fn", None)
        if not inner_fn:
            inner_fn = UserFunctionVariable(fn.func, source=AttrSource(kwargs["source"], "func"), is_constant=is_constant)
        assert isinstance(fn, functools.partial)
        super().__init__(**kwargs)
        self.fn = fn
        self.is_constant = is_constant
        self.inner_fn = inner_fn

    def bind_args(self, parent, args, kwargs):
        if self.fn.args:
            unimplemented(f"NYI - partials w/ non-keyword args")
        result, closure_cells = self.inner_fn.bind_args(parent, args, kwargs)
        if closure_cells:
            unimplemented("NYI - partials w/ closure_cells")
        wrapped_args =  dict(self.fn.keywords)
        options = VariableTracker.propagate([self])
        defaults_sources = [
            None if self.source is None else DefaultsSource(AttrSource(self.source, "func"), idx)
            for idx, _ in enumerate(wrapped_args)
        ]
        wrap_args_kwargs(parent.output.root_tx, wrapped_args, VariableTracker.propagate(self))
        wrap = functools.partial(wrap_bound_arg, tx=parent.output.root_tx, options=options)
        for i, key in enumerate(wrapped_args.keys()):
            value = wrapped_args[key]
            value = wrap(val=value, source=defaults_sources[i])
            result[key] = value
        return result, closure_cells

    def self_args(self):
        return []

    def has_self(self):
        return False
    
    def get_code(self):
        return self.inner_fn.get_code()
    
    def get_function(self):
        return self.inner_fn.get_function()

    def get_globals(self):
        return self.inner_fn.get_globals()
    
    def export_freevars(self, parent, child):
        return self.inner_fn.export_freevars(parent, child)


class WrappedUserMethodVariable(UserMethodVariable):
    def __init__(self, wrapped, context, **kwargs):
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        super().__init__(wrapped.fn, wrapped.obj, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result


class WrappedUserFunctionVariable(UserFunctionVariable):
    def __init__(self, wrapped, context, **kwargs):
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        super().__init__(wrapped.fn, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result


def invoke_and_store_as_constant(tx, fn, name, options, args, kwargs):
    def convert(x):
        if isinstance(x, variables.TensorVariable):
            return x.get_real_value()
        if isinstance(x, variables.UserDefinedObjectVariable):
            return x.value
        if isinstance(x, variables.GetAttrVariable):
            if isinstance(x.obj, variables.TensorVariable):
                return getattr(x.obj.get_real_value(), x.name)
            # YOLO
            return getattr(x.obj, x.name)
        return x.as_python_constant()

        
    args = [convert(x) for x in args]
    kwargs = {k: convert(v) for k, v in kwargs.items()}
    res = fn(*args, **kwargs)
    return tx.output.register_attr_or_module(
        res,
        name,
        source=ConstantSource(name),
        **options,
    )


class NestedUserFunctionVariable(BaseUserFunctionVariable):
    def __init__(
        self,
        fn_name,
        code,
        f_globals,
        defaults,
        kwdefaults,
        annotations,
        closure,
        closure_scope,
        wraps_source=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(fn_name.as_python_constant(), str)
        assert isinstance(code.as_python_constant(), types.CodeType)
        assert isinstance(f_globals, dict)
        self.fn_name = fn_name
        self.code = code
        self.f_globals = f_globals
        self.defaults = defaults
        self.kwdefaults = kwdefaults
        self.annotations = annotations
        self.closure = closure
        if closure is None:
            closure_scope = None
        self.closure_scope = closure_scope
        self.wraps_source = wraps_source

    def self_args(self):
        return []

    def get_code(self):
        return self.code.as_python_constant()

    def get_function(self):
        if self.closure:
            raise NotImplementedError()
        func = types.FunctionType(
            self.code.as_python_constant(),
            self.f_globals,
            self.fn_name.as_python_constant(),
        )
        if self.defaults:
            func.__defaults__ = self.defaults.as_python_constant()
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.as_python_constant()
        if self.annotations:
            annotations = self.annotations.as_python_constant()
            if isinstance(annotations, tuple):
                from itertools import pairwise

                annotations = dict(pairwise(annotations))

            # TypeError: __annotations__ must be set to a dict object
            assert isinstance(annotations, dict)
            func.__annotations__ = annotations
        return func

    def has_closure(self):
        return self.closure is not None

    def has_self(self):
        return False

    def get_globals(self):
        return self.f_globals

    def bind_args(self, parent, args, kwargs):
        code = self.get_code()
        func = types.FunctionType(
            code,
            self.f_globals,
            self.fn_name.as_python_constant(),
            tuple(self.defaults.items) if self.defaults else None,
            tuple(make_cell(None) for _ in range(len(self.get_code().co_freevars))),
        )
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.items
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())
        wrap_args_kwargs(parent.output.root_tx, result, VariableTracker.propagate(self))
        closure_cells = init_cellvars(parent, result, code)

        for idx, name in enumerate(code.co_freevars):
            if not getattr(self.closure.items[idx], name, name) == name:
                unimplemented(f"{getattr(self.closure.items[idx], name, name)} MISMATCH {name}")
            assert name not in result
            closure_cells[name] = self.closure.items[idx]

        return result, closure_cells

    def export_freevars(self, parent, child):
        code = self.get_code()
        for var in code.co_freevars:
            if var in child.symbolic_locals:
                parent.symbolic_locals[var] = child.symbolic_locals[var]

    def reconstruct(self, codegen):
        codegen.load_import_from(__name__, "_create_nested_fn")
        codegen(self.code)
        codegen.extend_output([codegen._create_load_const(self.f_globals)])
        codegen(self.fn_name)

        if self.defaults:
            codegen(self.defaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.closure:
            codegen(self.closure)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.kwdefaults:
            codegen(self.kwdefaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.annotations:
            try:
                if isinstance(self.annotations, variables.ConstDictVariable):
                    annotations = {
                        k: v.as_python_constant()
                        for k, v in self.annotations.items.items()
                    }
                else:
                    annotations = tuple(
                        [v.as_python_constant() for v in self.annotations.items]
                    )
                codegen.extend_output([codegen._create_load_const(annotations)])
            except NotImplementedError:
                codegen(self.annotations)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        codegen.extend_output(create_call_function(7, push_null=True))

        if self.wraps_source:
            codegen.load_import_from("functools", "wraps")
            codegen(self.wraps_source)
            codegen.extend_output(create_call_function(1, True))
            codegen.extend_output(create_rot_n(2))
            codegen.extend_output(create_call_function(1, True))

        return []
