from __future__ import annotations

import inspect
import operator
import re
import types
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
    graph_building as onnxscript_graph_building,
)

import torch
from torch._subclasses import fake_tensor
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
    diagnostics,
    onnxfunction_dispatcher,
    op_validation,
    type_utils as fx_type_utils,
)
from torch.utils import _pytree


@_beartype.beartype
def _fx_node_to_onnx_message_formatter(
    fn: Callable,
    self,
    node: torch.fx.Node,
    *args,
    **kwargs,
) -> str:
    return f"FX Node: {node.op}:{node.target}[name={node.name}]. "


def _location_from_fx_stack_trace(
    node_stack_trace: str,
) -> Optional[diagnostics.infra.Location]:
    """Extract location from FX node stack trace.

    TODO(bowbao): Create fx utils module and move this function there.

    Args:
        node_stack_trace: The stack trace of the FX node. Example:

            File "path/file.py", line 311, in <function>
                <code>
            |   File "path/file2.py", line 389, in <function>
                <code>

    Returns:
        location: The location of the FX node.
    """
    if "File" not in node_stack_trace:
        return None

    lines = node_stack_trace.strip().split("\n")
    idx = 0
    while idx < len(lines) and "File" not in lines[idx]:
        idx += 1
    if idx + 1 >= len(lines):
        return None

    pattern = re.compile(r"^File \"(.+)\", line (\d+), in (.+)$")
    matches = pattern.match(lines[idx].strip())
    if matches:
        uri = matches.group(1)
        line_number = int(matches.group(2))
        snippet = lines[idx + 1].strip()
        return diagnostics.infra.Location(uri=uri, line=line_number, snippet=snippet)
    return None


@_beartype.beartype
def _retrieve_or_adapt_input_to_graph_set(
    fx_node_arg: fx_type_utils.Argument,
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            onnxscript_graph_building.TorchScriptTensor,
            Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
        ],
    ],
    tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
):
    """Map FX value to TorchScript value.

    When creating TorchScript graph from FX graph, we need a mapping from FX variable
    to TorchScript variable. This function maps FX variable, fx_node_arg, to torch.jit.Value.
    """

    onnx_tensor = fx_node_arg
    if isinstance(onnx_tensor, torch.fx.Node):
        # 1. fx_node_arg is a torch.fx.Node, which means
        #    fx_node_arg stands for the output of that torch.fx.Node.
        # 2. fx_node_arg (variable in torch.fx.Graph) is be mapped to
        #    torch.jit.Value, fx_name_to_onnxscript_value[fx_node_arg.name],
        #    in TorchScript graph.
        return fx_name_to_onnxscript_value[onnx_tensor.name]
    if isinstance(onnx_tensor, (tuple, list)) and any(
        isinstance(node, torch.fx.Node) and isinstance(node.meta["val"], torch.SymInt)
        for node in onnx_tensor
    ):
        # This intends to handle dynamic axes. for example, if the input size of op.Expand
        # is dynamic, each dimension would be variable (i.e., sym variable in Pytorch
        # FX graph. Note that sym variable is mapped to tensor in ONNX Script world)
        # calculated by other operators.
        sequence_mixed_elements: List[
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                List[int],
            ]
        ] = []
        for tensor in onnx_tensor:
            if isinstance(tensor, torch.fx.Node) and isinstance(
                tensor.meta["val"], torch.SymInt
            ):
                sequence_mixed_elements.append(fx_name_to_onnxscript_value[tensor.name])
            elif isinstance(tensor, int):
                # NOTE: op.Concat doesn't support scalar, so we need to wrap it with
                # dim, and onnx-script will promote it to tensot(int64)
                sequence_mixed_elements.append([tensor])
        # Concat all the elements in the sequence.
        # shapes are mapped to tensors in ONNX graph (TorchScriptGraph),
        # so list of sym_ints is concatenated to a tensor before calling ONNX op.

        # For example:
        #    inputs: [[2], [4], fx.Node(SymIntA), [1], fx.Node(SymIntB)]
        #    outputs: op.Concat([op.Constant(2), op.Constant(4), TorchScriptTensor(A), op.Constant(1), TorchScriptTensor(B)])

        # onnx-script auto wraps python number with op.Constants,
        # so we don't need to specifically process them.
        with onnxscript.evaluator.default_as(tracer):
            output = onnxscript.opset18.Concat(*sequence_mixed_elements, axis=0)
        output.dtype = torch.int64
        output.shape = [len(sequence_mixed_elements)]
        return output
    elif isinstance(onnx_tensor, (tuple, list)) and all(
        isinstance(node, torch.fx.Node) for node in onnx_tensor
    ):
        sequence_elements: List[
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[
                    onnxscript_graph_building.TorchScriptTensor,
                    ...,
                ],
            ]
        ] = []
        for tensor in onnx_tensor:
            sequence_elements.append(fx_name_to_onnxscript_value[tensor.name])
        return sequence_elements
    if isinstance(onnx_tensor, torch.dtype):
        onnx_tensor = int(
            jit_type_utils.JitScalarType.from_dtype(onnx_tensor).onnx_type()
        )

    # all other cases, we do nothing.
    return onnx_tensor


def filter_incompatible_and_dtype_convert_kwargs(kwargs):
    """Filter out kwargs that are not supported by onnxscript."""
    filtered = {}
    for key, value in kwargs.items():
        if key in {
            "layout",
            "device",
            "requires_grad",
            "pin_memory",
            "memory_format",
            "implicit",
        }:
            continue
        if key == "dtype":
            if value is None:
                # We omit if dtype is not provided, because onnxscript handles the
                # default case.
                continue
            else:
                filtered["dtype"] = int(
                    jit_type_utils.JitScalarType.from_dtype(value).onnx_type()
                )
            continue
        filtered[key] = value
    return filtered


@_beartype.beartype
def _fill_tensor_shape_type(
    onnxscript_values: Union[
        onnxscript_graph_building.TorchScriptTensor,
        Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
    ],
    name: str,
    expected_values: Union[
        fake_tensor.FakeTensor,
        torch.SymInt,
        torch.SymFloat,
        List[fake_tensor.FakeTensor],
        Tuple[fake_tensor.FakeTensor, ...],
    ],
):
    """Fill the meta information of onnxscript_values with that from the fx FakeTensor."""

    if isinstance(expected_values, (list, tuple)) and not isinstance(
        onnxscript_values, (list, tuple)
    ):
        # ex: aten::split - in onnx_dtype: seq(tensor)
        # onnxscript_values is a single tensor, but expected_values is a list of tensors.
        return

    flat_onnxscript_values, _ = _pytree.tree_flatten(onnxscript_values)
    flat_expected_values, _ = _pytree.tree_flatten(expected_values)
    for i, (onnxscript_value, expected_value) in enumerate(
        zip(flat_onnxscript_values, flat_expected_values)
    ):
        # aten::sym_size output is a int, not a tensor, which stands
        # for the size of one dim. We treat it as 0-D tensor.
        # TODO(titaiwang): set shape?
        if isinstance(expected_value, torch.SymInt):
            onnxscript_value.dtype = torch.int64
        elif isinstance(expected_value, torch.SymFloat):
            onnxscript_value.dtype = torch.float32
        elif fx_type_utils.is_torch_complex_dtype(expected_value.dtype):
            # Like torch.view_as_real, we flatten complex tensors to real tensors with
            # additional last dimension of 2
            onnxscript_value.shape = (
                *[
                    dim if isinstance(dim, int) else None
                    for dim in expected_value.size()
                ],
                2,
            )
            # complex64 -> float32, complex128 -> float64, etc.
            onnxscript_value.dtype = fx_type_utils.from_complex_to_float(
                expected_value.dtype
            )
            # Dispatcher needs to know the value is complex
            onnxscript_value.is_complex = True
        else:
            # We set node output sizes to be dynamic to continue the model conversion,
            # and inputs are also set to be dynamic in add_input().
            onnxscript_value.shape = tuple(
                [dim if isinstance(dim, int) else None for dim in expected_value.size()]
            )
            onnxscript_value.dtype = expected_value.dtype
        # naming
        if i > 0:
            onnxscript_value.name = f"{name}_{i}"
        else:
            onnxscript_value.name = name


@_beartype.beartype
def _fill_in_default_kwargs(
    node: torch.fx.Node,
) -> Tuple[List[fx_type_utils.Argument], Dict[str, fx_type_utils.Argument]]:
    """Find and Fill in the not provided kwargs with default values."""

    # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
    # overloadpacket for some reasons.
    # https://github.com/pytorch/pytorch/issues/97201
    # We manually assigned overload for aten::sym_size.
    if hasattr(node.target, "_schema"):
        node_schema = node.target._schema  # type: ignore[union-attr]
    else:
        node_schema = torch.ops.aten.sym_size.int._schema  # type: ignore[union-attr]

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    complete_args: List[fx_type_utils.Argument] = []
    complete_kwargs: Dict[str, fx_type_utils.Argument] = {}

    if inspect.isbuiltin(node.target):
        complete_args = list(node.args)
    else:
        for i, expected_arg in enumerate(node_schema.arguments):
            if i < len(node.args):
                complete_args.append(node.args[i])
            elif expected_arg.name in node.kwargs:
                complete_kwargs[expected_arg.name] = node.kwargs[expected_arg.name]
            else:
                # Get default from schema.
                complete_kwargs[expected_arg.name] = expected_arg.default_value

    return complete_args, complete_kwargs


@_beartype.beartype
def _wrap_fx_args_as_onnxscript_args(
    complete_args: List[fx_type_utils.Argument],
    complete_kwargs: Dict[str, fx_type_utils.Argument],
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            onnxscript_graph_building.TorchScriptTensor,
            Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
        ],
    ],
    tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
) -> Tuple[
    Sequence[
        Optional[
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                str,
                int,
                float,
                bool,
                list,
            ]
        ]
    ],
    Dict[str, fx_type_utils.Argument],
]:
    """Map all FX arguments of a node to arguments in TorchScript graph."""

    onnxscript_args = tuple(
        _retrieve_or_adapt_input_to_graph_set(arg, fx_name_to_onnxscript_value, tracer)
        for arg in complete_args
    )
    onnxscript_kwargs = filter_incompatible_and_dtype_convert_kwargs(complete_kwargs)

    return onnxscript_args, onnxscript_kwargs


class FxOnnxInterpreter:
    """Stateless class to process FX graph Nodes and translate them into their ONNX counterparts.

    All FX nodes described by [FX Graph](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) are supported.
    Similarly to [FX Interpreter pattern](https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter), each FX node
    must be implemented on its own method in this class.

    Each operator's implementation returns either an `onnxscript.OnnxFunction` or
    `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm. They can
    also raise RuntimeError: If there are no overloaded functions available for the given FX node.

    TODO: Convert methods to @staticmethod when the diagnostic system supports it
          DO NOT ADD NEW ATTRIBUTES TO THIS CLASS!
    """

    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
    ):
        # THIS SHOULD BE THE ONLY STATE IN THIS CLASS (constraint from diagnosticS API)
        # TODO: Diagnostics API should be revised to get rid of this attribute.
        # DO NOT add other class-level attributes.
        self.diagnostic_context = diagnostic_context

    @_beartype.beartype
    @diagnostics.diagnose_call(
        diagnostics.rules.fx_node_to_onnx,
        diagnostic_message_formatter=_fx_node_to_onnx_message_formatter,
    )
    def run_node(
        self,
        node,
        fx_graph_module: torch.fx.GraphModule,
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        onnxscript_tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
    ):
        """Execute a single FX node to produce its ONNX counterpart.

        Args:
            node: The FX node to be translated.
            fx_graph_module: The FX graph module containing the node.
            onnxfunction_dispatcher: The dispatcher to find the best matched ONNX op.
            op_level_debug (bool): Whether to enable op level debug.
            onnxscript_graph: The ONNX graph to be populated.
            onnxscript_tracer: The tracer to trace the ONNX graph.
            fx_name_to_onnxscript_value: The mapping from FX node name to ONNX Script value.

        Raises:
            RuntimeError: When a node.op is not supported.
        """
        # Record stack trace of node in diagnostic.
        node_stack_trace = node.stack_trace
        if node_stack_trace:
            diagnostic = self.diagnostic_context.inflight_diagnostic(
                rule=diagnostics.rules.fx_node_to_onnx
            )
            diagnostic.with_additional_message(
                f"### PyTorch source information\n```\n{node_stack_trace}\n```"
            )
            location = _location_from_fx_stack_trace(node_stack_trace)
            if location is not None:
                diagnostic.with_location(location)

        if node.op == "placeholder":
            self.placeholder(node, onnxscript_graph, fx_name_to_onnxscript_value)
        elif node.op == "get_attr":
            self.get_attr(
                node,
                onnxscript_graph,
                fx_name_to_onnxscript_value,
                fx_graph_module,
            )
        elif node.op == "call_function":
            self.call_function(
                node,
                onnxscript_tracer,
                fx_name_to_onnxscript_value,
                onnxfunction_dispatcher,
                op_level_debug,
            )
        elif node.op == "call_method":
            self.call_method(node)
        elif node.op == "call_module":
            self.call_module(node)
        elif node.op == "output":
            self.output(node, onnxscript_graph, fx_name_to_onnxscript_value)
        else:
            raise RuntimeError(f"Found node type not defined in torch.fx: {node.op}")

    @_beartype.beartype
    @diagnostics.diagnose_call(diagnostics.rules.atenlib_fx_to_onnx)
    def run(
        self,
        fx_graph_module: torch.fx.GraphModule,
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
    ) -> onnxscript_graph_building.TorchScriptGraph:
        """Analyze all FX nodes and trigger their ONNX translation.

        Args:
            fx_graph_module: FX graph module to be translated.
            onnxfunction_dispatcher: ONNX function dispatcher.
            op_level_debug: Whether to enable op-level debug.
        """
        onnxscript_graph = onnxscript_graph_building.TorchScriptGraph()
        onnxscript_tracer = onnxscript_graph_building.TorchScriptTracingEvaluator(
            onnxscript_graph
        )
        # In the following loop, a TorchScript graph is created to
        # represent the input FX graph with ONNX symbols (e.g., onnx::add).
        # To connect the values to nodes in the TorchScript graph, we maintian
        # fx_name_to_onnxscript_value. Basically, we want to translate
        #   fx_tensor_x (type: torch.fx.Node) -> fx_node_1 -> fx_tensor_y (type: torch.fx.Node)
        # to
        #   fx_name_to_onnxscript_value[fx_tensor_x.name] -> onnx_node_1 -> fx_name_to_onnxscript_value[fx_tensor_y.name]
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ] = {}

        # TODO: Fix FakeTensorMode limitation asap
        # We want to pass list of ints and floats to TorchScript graph correctly
        # in _export_fx_to_ts, so we must disable FakeTensorMode. Otherwise, graph may
        # receive FakeTensor and results runtime error. In addition, TorchScript-based
        # ONNX exporter used in _ts_graph_to_onnx_model_in_protobuf is not compatible
        # with FakeTensorMode.
        with torch.utils._mode_utils.no_dispatch():
            # node_fixed_shape is only used on op_level_debug purpose.
            for node in fx_graph_module.graph.nodes:
                self.run_node(
                    node,
                    fx_graph_module,
                    onnxfunction_dispatcher,
                    op_level_debug,
                    onnxscript_graph,
                    onnxscript_tracer,
                    fx_name_to_onnxscript_value,
                )

        return onnxscript_graph

    @_beartype.beartype
    def placeholder(
        self,
        node: torch.fx.Node,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
    ):
        # Input of graph.
        # The node.meta["val"] is generated by FakeTensorProp.
        # NOTE: add_input() intends to create nodes with shape/type
        fake_tensor = node.meta.get("val", None)
        if fake_tensor is None:
            output = onnxscript_graph.add_input(
                input_name=None,
            )
        else:
            output = onnxscript_graph.add_input(
                input_name=node.name,
                shape=fake_tensor.shape,
                dtype=fake_tensor.dtype,
            )
        assert (
            output is not None
        ), f"Node creates None with target={node.target} and name={node.name}"

        assert isinstance(output, onnxscript_graph_building.TorchScriptTensor)
        assert isinstance(output, onnxscript.tensor.Tensor)

        fx_name_to_onnxscript_value[node.name] = output

    @_beartype.beartype
    def call_function(
        self,
        node: torch.fx.Node,
        onnxscript_tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
    ):
        # aten ops and other stateless functions.
        if node.target == operator.getitem and isinstance(
            fx_name_to_onnxscript_value[node.args[0].name], tuple  # type: ignore[union-attr,index]
        ):
            onnx_tensor_tuple = fx_name_to_onnxscript_value[node.args[0].name]  # type: ignore[union-attr,index]
            index = node.args[1]
            output = onnx_tensor_tuple[index]  # type: ignore[index]
            assert (
                output is not None
            ), f"Node creates None with target={node.target} and name={node.name}"
            assert isinstance(
                output, (onnxscript_graph_building.TorchScriptTensor, tuple)
            ), type(output)

            fx_name_to_onnxscript_value[node.name] = output
            return

        # Map FX inputs to ONNX inputs and fill optional inputs with default values.
        # torch_args and torch_kwargs are for op-level validation
        complete_args, complete_kwargs = _fill_in_default_kwargs(node)
        onnx_args, onnx_kwargs = _wrap_fx_args_as_onnxscript_args(
            complete_args,
            complete_kwargs,
            fx_name_to_onnxscript_value,
            onnxscript_tracer,
        )

        # Dispatch to ONNX op through OpShema. The input argument dtypes are compared to
        # function signature in OpSchema, and find the best matched overload.
        # TODO(titaiwang): diagnostic rules.
        symbolic_fn = onnxfunction_dispatcher.dispatch(
            node=node,
            onnx_args=onnx_args,
            onnx_kwargs=onnx_kwargs,
            diagnostic_context=self.diagnostic_context,
        )

        with onnxscript.evaluator.default_as(onnxscript_tracer):
            output: Union[  # type: ignore[no-redef]
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ] = symbolic_fn(*onnx_args, **onnx_kwargs)
        assert (
            output is not None
        ), f"Node creates None with target={node.target}, name={node.name}, args={onnx_args}, kwargs={onnx_kwargs}"
        # Assign type and shape from fx graph.
        _fill_tensor_shape_type(output, node.name, node.meta["val"])
        # One fx node could produce multiple outputs (e.g., tuple of tensors); in
        # that case, v is a tuple of TorchScriptTensors.
        assert isinstance(
            output, (onnxscript_graph_building.TorchScriptTensor, tuple)
        ), type(output)
        # NOTE(titaiwang): We bypass two kinds of ops as it's not meaningful to
        # validate them with op level debug.
        # 1. aten::sym_size: The op is simply get item from a list of tensors.
        # 2. BuiltinFunction: It doesn't supported tensor
        if (
            op_level_debug
            and node.target != torch.ops.aten.sym_size
            and not isinstance(node.target, types.BuiltinFunctionType)
        ):
            (
                node_with_fixed_shape_args,
                node_with_fixed_shape_kwargs,
            ) = _fill_in_default_kwargs(node)
            try:
                torch_args, torch_kwargs = op_validation.wrap_fx_args_as_torch_args(
                    node_with_fixed_shape_args, node_with_fixed_shape_kwargs
                )
            except ValueError as value_error:
                warnings.warn(
                    f"\nFound unsupported input types on PyTorch Op {node.target} with "
                    f"ValueError: \n{value_error}.\n"
                )
                diagnostic = self.diagnostic_context.inflight_diagnostic()
                diagnostic.with_additional_message(
                    f"### Op level debug fails due to unsupported input types\n"
                    f"{diagnostics.decorator.format_exception_in_markdown(value_error)}"
                )
                diagnostic.level = diagnostics.levels.ERROR
            else:
                op_validation.validate_op_between_ort_torch(
                    self.diagnostic_context,
                    node,
                    symbolic_fn,
                    torch_args,
                    torch_kwargs,
                )
        fx_name_to_onnxscript_value[node.name] = output

    @_beartype.beartype
    def output(
        self,
        node: torch.fx.Node,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
    ):
        if isinstance(node.args[0], torch.fx.Node):
            onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[node.args[0].name]
            onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)
        else:
            # ONNX can't represent collection types (e.g., dictionary, tuple of tuple of
            # tensor, etc), we flatten the collection and register each element as output.
            flat_args, _ = _pytree.tree_flatten(node.args[0])
            for arg in flat_args:
                assert isinstance(
                    arg, torch.fx.Node
                ), f"arg must be a torch.fx.Node, not {type(arg)}"
                onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[arg.name]
                onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)

    @_beartype.beartype
    def call_method(self, node: torch.fx.Node):
        # TODO(wechi): Support call_method.
        raise RuntimeError("call_method is not supported yet.")

    @_beartype.beartype
    def call_module(self, node: torch.fx.Node):
        # TODO(wechi): Support call_module.
        raise RuntimeError("call_module is not supported yet.")

    @_beartype.beartype
    def get_attr(
        self,
        node: torch.fx.Node,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
        fx_graph_module: torch.fx.GraphModule,
    ):
        current_attr = fx_graph_module
        sub_attr_names = node.target.split(".")  # type: ignore[union-attr]
        # If node.targe is "conv.weight", the following loop first
        # assigns fx_module_with_metadata.conv to current_attr, and then
        # fx_module_with_metadata.conv.weight to current_attr.
        while sub_attr_names:
            sub_attr_name = sub_attr_names.pop(0)
            if not hasattr(current_attr, sub_attr_name):
                raise AttributeError(
                    f"Attribute {sub_attr_name} is not found in {current_attr}."
                )
            current_attr = getattr(current_attr, sub_attr_name)

        input_ = onnxscript_graph.add_initializer(node.name, current_attr)

        assert isinstance(input_, onnxscript_graph_building.TorchScriptTensor)
        assert isinstance(input_, onnxscript.tensor.Tensor)
        fx_name_to_onnxscript_value[node.name] = input_
        # FIXME: Refactor logic getting 'current_attr'.
        assert isinstance(current_attr, torch.Tensor)
