from typing import Dict, List

import torch
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name
from ..guards import GuardBuilder

from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .lists import ListVariable
from .misc import GetAttrVariable
from .user_defined import UserDefinedObjectVariable
from copy import copy


class ArgMappingException(Exception):
    pass

class OptimizerStepVariable(VariableTracker):
    def reconstruct(self, cg):
        return self.source.base.reconstruct(cg)

class OptimizerVariable(UserDefinedObjectVariable):
    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        """This is an optimization to avoid tracing the very slow intialization of the optimizer"""
        import torch.optim

        if name == "_init_group":
            try:
                py_args, py_kwargs = self.get_python_args(*args, **kwargs)
                self.value._init_group(*py_args, **py_kwargs)
                self.install_guards(tx)
                self.update_list_args(tx, args, kwargs, py_args, py_kwargs)
                if isinstance(self.value, torch.optim.RMSprop):
                    group = args[0]
                    self.update_step(tx, group)

                return ConstantVariable(None)
            except ArgMappingException:
                # trace normally if we can't map args
                pass

        return super().call_method(tx, name, args, kwargs)

    def param_group_source(self, group_ind):
        return GetItemSource(AttrSource(self.source, "param_groups"), group_ind)

    def param_state_source(self, p_source):
        return GetItemSource(AttrSource(self.source, "state"), p_source)

    def params_with_sources(self):
        for g_ind, group in enumerate(self.value.param_groups):
            group_source = self.param_group_source(g_ind)
            for p_ind, p in enumerate(group["params"]):
                if p in self.value.state:
                    yield p, GetItemSource(GetItemSource(group_source, "params"), p_ind)

    def update_step(self, tx, group):
        for p_ind, param in enumerate(self.value.param_groups[group.source.index]["params"]):
            group_source = self.param_group_source(group.source.index)
            p_source = GetItemSource(GetItemSource(group_source, "params"), p_ind)
            source = GetItemSource(self.param_state_source(p_source), 'step')
            var = OptimizerStepVariable(source=source)
            tracked_var = tx.output.side_effects._track_obj(source, self.value.state[param]['step'], var)
            # register the mutation
            newvar = OptimizerStepVariable(source=source)
            tx.replace_all(tracked_var, newvar)

    def map_grads_to_sources(self):
        """Map the optimizer's grads to their sources"""
        self.grad_to_source = {}
        for param, p_source in self.params_with_sources():
            self.grad_to_source[param.grad] = AttrSource(p_source, "grad")

    def var_getattr(self, tx, name):
        if name == "_init_group":
            return GetAttrVariable(self, name)

        return super().var_getattr(tx, name)

    def get_python_args(self, *args, **kwargs):
        """Get python values equivalent to the variable tracker args"""

        def map_arg(arg):
            if isinstance(arg, ConstantVariable):
                return arg.as_python_constant()
            elif isinstance(arg, ListVariable) and not arg.items:
                return []
            elif (
                isinstance(arg, ConstDictVariable)
                and isinstance(arg.source, GetItemSource)
                and isinstance(arg.source.base, AttrSource)
                and arg.source.base.member == "param_groups"
            ):
                return self.value.param_groups[arg.source.index]

            raise ArgMappingException()

        new_args = [map_arg(arg) for arg in args]
        new_kwargs = {k: map_arg(v) for k, v in kwargs.items()}

        return new_args, new_kwargs

    def install_guards(self, tx):
        from .builder import VariableBuilder

        tx.output.guards.update([self.source.make_guard(GuardBuilder.ID_MATCH)])

        if isinstance(self.value, torch.optim.RMSprop):
            state_dict = copy(self.value.state)
            for p in state_dict.keys():
                p_state = copy(state_dict[p])
                p_state.pop("step", None)
                state_dict[p] = p_state

        state_dict_var = VariableBuilder(tx, AttrSource(self.source, "state"))(
            state_dict
        )
        tx.output.guards.update(state_dict_var.guards)

        group_guards = VariableBuilder(tx, AttrSource(self.source, "param_groups"))(
            self.value.param_groups
        )
        tx.output.guards.update(group_guards.guards)

    def wrap_tensor(self, tx, tensor_value):
        """Wrap state tensor in a TensorVariable"""
        from .builder import VariableBuilder

        # don't add weakref guards for grads, they will possibly change on
        # each iteration
        if tensor_value in self.grad_to_source:
            return VariableBuilder(tx, self.grad_to_source[tensor_value])(tensor_value)
        else:
            tx.store_dict_key(global_key_name(tensor_value), tensor_value)
            return VariableBuilder(
                tx, GlobalWeakRefSource(global_key_name(tensor_value))
            )(tensor_value)

    def update_list_args(self, tx, args, kwargs, py_args, py_kwargs):
        """Update the args and kwargs to the traced optimizer call"""
        self.map_grads_to_sources()
        for arg, py_arg in zip(args, py_args):
            if isinstance(arg, ListVariable) and all(
                isinstance(t, torch.Tensor) for t in py_arg
            ):
                tensor_vars = ListVariable(
                    [self.wrap_tensor(tx, t) for t in py_arg],
                    mutable_local=MutableLocal(),
                )
                arg.call_method(tx, "extend", (tensor_vars,), {})
