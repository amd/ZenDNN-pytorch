import functools
from functools import reduce

import torch

from torch.fx.experimental.symbolic_shapes import free_symbols

from .. import ir

from ..lowering import lowerings as L
from ..pattern_matcher import Arg, CallFunction, filter_nodes, get_arg_value, KeywordArg
from ..virtualized import ops
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern


if torch._C._has_mkldnn:
    aten = torch.ops.aten
    mkldnn = torch.ops.mkldnn
    prims = torch.ops.prims

    _conv_args = [Arg() for _ in range(10)]
    _linear_args = [Arg() for _ in range(6)]
    _conv_transpose_args = [Arg() for _ in range(11)]

    def _conv_call(users=1):
        return CallFunction(
            mkldnn._convolution_pointwise.default, *_conv_args, _users=users
        )

    def _linear_call(users=1):
        return CallFunction(
            mkldnn._linear_pointwise.default, *_linear_args, _users=users
        )

    def _conv_transpose_call(users=1):
        return CallFunction(
            mkldnn._convolution_transpose_pointwise.default,
            *_conv_transpose_args,
            _users=users,
        )

    def _to_float(input_call, users=1):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_float"),
            _users=users,
        )

    def _to_bf16(input_call):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_bf16"),
            _users=1,
        )

    def _unary_fusion_pattern(unary_fusion, call_fn, users, is_bf16):
        # only insert to_dtype if is_bf16 is True
        computation_call = (
            _to_float(call_fn(), users=users) if is_bf16 else call_fn(users=users)
        )
        out = unary_fusion(computation_call)
        return _to_bf16(out) if is_bf16 else out

    def _gelu_fusion_1(computation_call):
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.erf,
                    CallFunction(aten.mul, computation_call, 0.7071067811865476),
                ),
                1,
            ),
        )

    def _gelu_fusion_2(computation_call):
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.tanh,
                    CallFunction(
                        aten.mul,
                        CallFunction(
                            aten.add,
                            computation_call,
                            CallFunction(
                                aten.mul,
                                CallFunction(
                                    aten.mul,
                                    CallFunction(
                                        aten.mul, computation_call, computation_call
                                    ),
                                    computation_call,
                                ),
                                0.044715,
                            ),
                        ),
                        0.7978845608028654,
                    ),
                ),
                1,
            ),
        )

    def _hardswish_fusion(computation_call):
        return CallFunction(
            aten.div,
            CallFunction(
                aten.mul,
                computation_call,
                CallFunction(
                    aten.clamp_max,
                    CallFunction(
                        aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                    ),
                    6,
                ),
            ),
            6,
        )

    def _silu_fusion(computation_call):
        return CallFunction(
            aten.mul, computation_call, CallFunction(aten.sigmoid, computation_call)
        )

    def _hardsigmoid_fusion(computation_call):
        return CallFunction(
            aten.div,
            CallFunction(
                aten.clamp_max,
                CallFunction(
                    aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                ),
                6,
            ),
            6,
        )

    def _leaky_relu_fusion(computation_call):
        return CallFunction(
            aten.where,
            CallFunction(aten.gt, computation_call, 0),
            computation_call,
            CallFunction(aten.mul, computation_call, KeywordArg("negative_slope")),
        )

    def _hardtanh_fusion(computation_call):
        return CallFunction(
            aten.clamp_max,
            CallFunction(aten.clamp_min, computation_call, KeywordArg("min_value")),
            KeywordArg("max_value"),
        )

    def _combined_fusion(computation_call, elementwise_op):
        return CallFunction(elementwise_op, computation_call)

    # binary_op(other, computation_op)
    def _binary_fusion_v1(computation_call, binary_fn):
        return CallFunction(binary_fn, KeywordArg("other"), computation_call)

    # binary_op(computation_op, other)
    def _binary_fusion_v2(computation_call, binary_fn):
        return CallFunction(binary_fn, computation_call, KeywordArg("other"))

    def _is_single_computation_op(computation_op):
        def fn(match):
            computation_nodes = filter_nodes(match.nodes, computation_op)
            if len(computation_nodes) < 1:
                return False
            if any(n.args[-3] != "none" for n in computation_nodes):
                return False
            return True

        return fn

    def _is_valid_computation_unary_fusion(computation_op, is_bf16=False):
        def fn(match):
            matched = _is_single_computation_op(computation_op)(match)
            computation_node = filter_nodes(match.nodes, computation_op)[0]
            if is_bf16:
                conversion_dtype_nodes = filter_nodes(
                    match.nodes, prims.convert_element_type.default
                )
                if len(conversion_dtype_nodes) != 2:
                    return False
                # fusion pattern is always in the form of computation_op + to_float32 + unary_op + to_bfloat16
                if computation_node == conversion_dtype_nodes[0].args[0]:
                    to_float = conversion_dtype_nodes[0].args[1]
                    to_bf16 = conversion_dtype_nodes[1].args[1]
                else:
                    to_float = conversion_dtype_nodes[1].args[1]
                    to_bf16 = conversion_dtype_nodes[0].args[1]
                matched = (
                    matched and to_float == torch.float and to_bf16 == torch.bfloat16
                )
            return matched

        return fn

    def _register_unary_fusion_lowering(
        pattern, unary_attr, computation_op, is_bf16=False
    ):
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_unary_fusion(computation_op, is_bf16),
        )
        def fn(match, *args, **kwargs):
            computation_args = list(args)[:-3] + [
                unary_attr.op_name,
                unary_attr.scalars_attr,
                unary_attr.algorithm_attr,
            ]
            return L[computation_op](*computation_args)

        return fn

    def _register_leaky_relu_fusion_lowering(pattern, computation_op, is_bf16=False):
        @register_lowering_pattern(
            pattern, extra_check=_is_single_computation_op(computation_op)
        )
        def fn(match, *args, **kwargs):
            negative_slope = kwargs.get("negative_slope")
            if isinstance(negative_slope, ir.TensorBox):
                matched = False
            else:  # inp is a Number
                matched = True
            if is_bf16:
                dtype1 = kwargs.get("to_float")
                dtype2 = kwargs.get("to_bf16")
                matched = matched and dtype1 == torch.float and dtype2 == torch.bfloat16
            computation_args = list(args)
            if matched:
                computation_args = computation_args[:-3] + [
                    "leaky_relu",
                    [negative_slope],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                # computation_args += ["none", [], ""]
                out = L[computation_op](*computation_args)
                if is_bf16:
                    out = L[prims.convert_element_type.default](out, dtype=torch.float)
                out = L[aten.where](
                    L[aten.gt](out, 0),
                    out,
                    L[aten.mul](out, negative_slope),
                )
                if is_bf16:
                    out = L[prims.convert_element_type.default](
                        out, dtype=torch.bfloat16
                    )
                return out

        return fn

    def _register_hardtanh_fusion_lowering(pattern, computation_op, is_bf16=False):
        @register_lowering_pattern(
            pattern, extra_check=_is_single_computation_op(computation_op)
        )
        def fn(match, *args, **kwargs):
            min_value = kwargs.get("min_value")
            max_value = kwargs.get("max_value")
            if isinstance(min_value, ir.TensorBox) or isinstance(
                max_value, ir.TensorBox
            ):
                matched = False
            else:  # inp is a Number
                matched = min_value <= max_value
            if is_bf16:
                dtype1 = kwargs.get("to_float")
                dtype2 = kwargs.get("to_bf16")
                matched = matched and dtype1 == torch.float and dtype2 == torch.bfloat16
            computation_args = list(args)
            if matched:
                computation_args = computation_args[:-3] + [
                    "hardtanh",
                    [min_value, max_value],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                out = L[computation_op](*computation_args)
                if is_bf16:
                    out = L[prims.convert_element_type.default](out, dtype=torch.float)
                out = L[aten.clamp_max](L[aten.clamp_min](out, min_value), max_value)
                if is_bf16:
                    out = L[prims.convert_element_type.default](
                        out, dtype=torch.bfloat16
                    )
                return out

        return fn

    _binary_attr = {
        aten.add: "add",
        ops.add: "add",
        aten.sub: "sub",
        ops.sub: "sub",
    }

    def _is_valid_binary(match, fn):
        binary_nodes = filter_nodes(match.nodes, fn)
        if len(binary_nodes) < 1:
            return False
        if any(
            not (
                hasattr(n.args[0], "meta")
                and isinstance(n.args[0].meta.get("val", None), torch.Tensor)
            )
            or not (
                hasattr(n.args[1], "meta")
                and isinstance(n.args[1].meta.get("val", None), torch.Tensor)
            )
            for n in binary_nodes
        ):
            return False
        # check alpha is one.
        if any(
            get_arg_value(n, 2, kwarg_name="alpha") != 1.0
            and get_arg_value(n, 2, kwarg_name="alpha") is not None
            for n in binary_nodes
        ):
            return False
        if any(
            n.args[0].meta["val"].size() != n.args[1].meta["val"].size()
            or n.args[0].meta["val"].device != n.args[1].meta["val"].device
            or n.args[0].meta["val"].dtype != n.args[1].meta["val"].dtype
            for n in binary_nodes
        ):
            return False
        # check args[0] and args[1] is not same
        if any(n.args[0] == n.args[1] for n in binary_nodes):
            return False
        return True

    def _is_valid_computation_binary(computation_op, binary_op, other_index=None):
        def fn(match):
            if not _is_single_computation_op(computation_op)(match):
                return False
            if not _is_valid_binary(match, binary_op):
                return False
            return True

        return fn

    def _is_valid_computation_binary_inplace(computation_op, binary_op, other_index):
        def fn(match):
            if not _is_valid_computation_binary(computation_op, binary_op)(match):
                return False
            binary_nodes = filter_nodes(match.nodes, binary_op)
            if any(len(n.args[other_index].users) > 1 for n in binary_nodes):
                return False
            if any(
                n.args[other_index].op in ["placeholder", "output"]
                for n in binary_nodes
            ):
                return False
            return True

        return fn

    def _register_binary_unary_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        fusion_op,
        unary_attr=None,
    ):
        @register_lowering_pattern(
            pattern, extra_check=_is_valid_computation_binary(computation_op, binary_op)
        )
        def fn(match, *args, **kwargs):
            other = kwargs.get("other")
            assert isinstance(other, ir.TensorBox)
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            return L[fusion_op](*computation_args)

        return fn

    def _register_binary_unary_maybe_inplace_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        inplace_fusion_op,
        outplace_fusion_op,
        unary_attr=None,
        other_index=None,
    ):
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_binary_inplace(
                computation_op, binary_op, other_index
            ),
        )
        def fn(match, *args, **kwargs):
            other = kwargs.get("other")
            assert isinstance(other, ir.TensorBox)
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            # Make sure the other is not an alias or mutation(fx side doesn't has such info).
            other.realize()
            can_be_inplace = not (
                isinstance(other.data, ir.ReinterpretView)
                or isinstance(other.get_layout(), (ir.MutationLayout, ir.AliasedLayout))
            )
            if not can_be_inplace:
                return L[outplace_fusion_op](*computation_args)
            return L[inplace_fusion_op](*computation_args)

        return fn

    computation_ops = [
        mkldnn._convolution_pointwise.default,
        mkldnn._linear_pointwise.default,
        mkldnn._convolution_transpose_pointwise.default,
    ]

    class UnaryAttr:
        def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
            self.op_name = op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""

    def _register_unary_fusion():
        computation_call_fns = [_conv_call, _linear_call, _conv_transpose_call]

        def _unary_fusion_patterns(is_bf16):
            replacement_unary_fusion_patterns = {
                UnaryAttr("gelu", algorithm_attr="tanh"): [
                    _unary_fusion_pattern(_gelu_fusion_2, call_fn, 4, is_bf16)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("gelu", algorithm_attr="none"): [
                    _unary_fusion_pattern(_gelu_fusion_1, call_fn, 2, is_bf16)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("hardswish"): [
                    _unary_fusion_pattern(_hardswish_fusion, call_fn, 2, is_bf16)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("hardsigmoid"): [
                    _unary_fusion_pattern(_hardsigmoid_fusion, call_fn, 1, is_bf16)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("swish"): [
                    _unary_fusion_pattern(_silu_fusion, call_fn, 2, is_bf16)
                    for call_fn in computation_call_fns
                ],
            }
            if not is_bf16:
                call_user1 = [call_fn(users=1) for call_fn in computation_call_fns]
                replacement_unary_fusion_patterns.update(
                    {
                        UnaryAttr("relu"): [
                            _combined_fusion(u, aten.relu) for u in call_user1
                        ],
                        UnaryAttr("sigmoid"): [
                            _combined_fusion(u, aten.sigmoid) for u in call_user1
                        ],
                        UnaryAttr("tanh"): [
                            _combined_fusion(u, aten.tanh) for u in call_user1
                        ],
                    }
                )

            return replacement_unary_fusion_patterns

        for is_bf16 in [True, False]:
            replace_patterns = _unary_fusion_patterns(is_bf16)
            for unary_attr, patterns in replace_patterns.items():
                _register_unary_fusion_lowering(
                    patterns[0], unary_attr, computation_ops[0], is_bf16
                )
                _register_unary_fusion_lowering(
                    patterns[1], unary_attr, computation_ops[1], is_bf16
                )
                _register_unary_fusion_lowering(
                    patterns[2], unary_attr, computation_ops[2], is_bf16
                )
            _leaky_relu_patterns = [
                _unary_fusion_pattern(_leaky_relu_fusion, call_fn, 3, is_bf16)
                for call_fn in computation_call_fns
            ]
            for pattern, computation_op in zip(_leaky_relu_patterns, computation_ops):
                _register_leaky_relu_fusion_lowering(pattern, computation_op, is_bf16)
            hardtanh_patterns = [
                _unary_fusion_pattern(_hardtanh_fusion, call_fn, 1, is_bf16)
                for call_fn in computation_call_fns
            ]
            for pattern, computation_op in zip(hardtanh_patterns, computation_ops):
                _register_hardtanh_fusion_lowering(pattern, computation_op, is_bf16)

    def _register_inplace_fusion():
        binary_ops = [aten.add, ops.add]
        inplace_fusion_op = mkldnn._convolution_pointwise_.binary
        outplace_fusion_op = mkldnn._convolution_pointwise.binary
        conv_call = _conv_call(users=1)
        conv_op = computation_ops[0]
        for binary_op in binary_ops:
            binary_v1 = _binary_fusion_v1(conv_call, binary_op)
            binary_unary_v1 = _combined_fusion(binary_v1, aten.relu)
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v1,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
                unary_attr=UnaryAttr("relu"),
            )
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v1,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
            )
            binary_v2 = _binary_fusion_v2(conv_call, binary_op)
            binary_unary_v2 = _combined_fusion(binary_v2, aten.relu)
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v2,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
                unary_attr=UnaryAttr("relu"),
            )
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v2,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
            )

    def _register_binary_fusion():
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [
            mkldnn._convolution_pointwise.binary,
            mkldnn._linear_pointwise.binary,
        ]
        _computation_user_1 = [_conv_call(users=1), _linear_call(users=1)]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern = _binary_fusion_v2(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

            for binary_op in [aten.add, ops.add]:
                pattern = _binary_fusion_v1(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

    def _register_binary_unary_fusion():
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [mkldnn._convolution_pointwise.binary]
        _computation_user_1 = [_conv_call(users=1)]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern_v1 = _combined_fusion(
                    _binary_fusion_v2(computation_call, binary_op), aten.relu
                )
                _register_binary_unary_fusion_lowering(
                    pattern_v1,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )
            for binary_op in [aten.add, ops.add]:
                pattern_v2 = _combined_fusion(
                    _binary_fusion_v1(computation_call, binary_op), aten.relu
                )
                _register_binary_unary_fusion_lowering(
                    pattern_v2,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )

    def _recover_linear():
        # convert reshape+linear+reshape to a single linear for applying fusion path.
        @register_freezing_graph_pattern(
            CallFunction(
                aten.reshape.default,
                CallFunction(
                    mkldnn._linear_pointwise.default,
                    CallFunction(aten.reshape.default, Arg(), KeywordArg("reshape_1")),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                ),
                KeywordArg("reshape_2"),
            ),
            pass_number=1,
        )
        def reshape_linear_reshape_pattern(match, *args, **kwargs):
            reshape_1 = kwargs.get("reshape_1")
            reshape_2 = kwargs.get("reshape_1")
            graph = match.graph
            node = match.output_node()
            if reshape_1[0] == reduce(lambda x, y: x * y, reshape_2[:-1]):
                repl = graph.call_function(mkldnn._linear_pointwise.default, args)
                repl.meta.update(node.meta)
                node.replace_all_uses_with(repl)
                match.erase_nodes(graph)

        def is_linear_add_bias(match):
            add_node = match.output_node()
            linear_node = add_node.args[0]
            weight_meta = linear_node.args[1].meta.get("val")
            bias_meta = add_node.args[1].meta.get("val")
            if weight_meta is None or bias_meta is None:
                return False
            return (
                linear_node.args[2] is None
                and bias_meta.dim() == 1
                and bias_meta.size(0) == weight_meta.size(0)
            )

        # convert linear+bias to a single linear for applying fusion path.
        @register_freezing_graph_pattern(
            CallFunction(
                aten.add.Tensor,
                CallFunction(mkldnn._linear_pointwise.default, *_linear_args),
                Arg(),
            ),
            pass_number=1,
            extra_check=is_linear_add_bias,
        )
        def linear_bias_pattern(match, *args):
            graph = match.graph
            add_node = match.output_node()
            linear_node = add_node.args[0]
            new_args = list(linear_node.args)
            new_args[2] = add_node.args[1]
            repl = graph.call_function(
                mkldnn._linear_pointwise.default, tuple(new_args)
            )
            repl.meta.update(add_node.meta)
            add_node.replace_all_uses_with(repl)
            match.erase_nodes(graph)

    def _is_packable_convolution(match):
        """
        Check if the node is supported for MKLDNN convolution.
        """
        conv_node = match.output_node()
        input_meta_value = conv_node.args[0].meta.get("val")
        weight_meta_value = conv_node.args[1].meta.get("val")
        if input_meta_value is None or weight_meta_value is None:
            return False
        input_size = input_meta_value.shape
        # TODO: support dynamic input size.
        if free_symbols(input_size) or conv_node.args[1].op != "get_attr":
            return False
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type != "cpu"
                or meta_value.dim() != 4
            ):
                return False
        if (
            input_meta_value.dtype == torch.bfloat16
            or weight_meta_value.dtype == torch.bfloat16
        ):
            if not mkldnn._is_mkldnn_bf16_supported():
                return False
        is_transposed = conv_node.args[-3]
        if is_transposed:
            groups = conv_node.args[-1]
            in_channels = weight_meta_value.size(0)
            # doesn't support group_depthwise_conv_transpose.
            if groups > 1 and groups == in_channels:
                return False
            # Port from: aten/src/ATen/native/Convolution.cpp:is_output_padding_big
            output_paddings = conv_node.args[-2]
            strides = conv_node.args[3]
            if any(
                output_padding >= stride
                for output_padding, stride in zip(output_paddings, strides)
            ):
                return False
        return True

    def _is_packable_linear(match):
        """
        Check if the node is supported for MKLDNN linear.
        """
        linear_node = match.output_node()
        # weight_idx is 1 for aten.mm and is 2 for aten.addmm
        weight_idx = 2 if linear_node.target == aten.addmm.default else 1
        if linear_node.args[weight_idx].op != "get_attr":
            return False
        input_meta_value = linear_node.args[weight_idx - 1].meta.get("val")
        weight_meta_value = linear_node.args[weight_idx].meta.get("val")
        if input_meta_value is None or weight_meta_value is None:
            return False
        batch_size = input_meta_value.shape[0]
        is_bf16_weight = weight_meta_value.dtype == torch.bfloat16
        # for fp32, batch_size should not be a free symbol.
        if not is_bf16_weight and free_symbols(batch_size):
            return False
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type != "cpu"
                or meta_value.dim() != 2
            ):
                return False
        if weight_idx == 2:
            bias_meta_value = linear_node.args[0].meta.get("val")
            if (
                bias_meta_value is None
                or meta_value.device.type != "cpu"
                or bias_meta_value.dim() != 1
                or bias_meta_value.size(0) != weight_meta_value.size(1)
            ):
                return False

        if (
            input_meta_value.dtype == torch.bfloat16
            or weight_meta_value.dtype == torch.bfloat16
        ):
            if not mkldnn._is_mkldnn_bf16_supported():
                return False
            if free_symbols(batch_size):
                # TODO: support symbolic input size for bfloat16 path
                return False
        return True

    _aten_conv_args = (
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        KeywordArg("is_transposed"),
        Arg(),
        Arg(),
    )

    def _register_weight_pack_pass():
        @register_freezing_graph_pattern(
            CallFunction(aten.convolution.default, *_aten_conv_args),
            extra_check=_is_packable_convolution,
        )
        def convolution(match, *args, **kwargs):
            is_transposed = kwargs.get("is_transposed")
            assert isinstance(is_transposed, bool)
            graph = match.graph
            conv_node = match.output_node()
            input_size = conv_node.args[0].meta.get("val").shape
            with graph.inserting_before(conv_node):
                mkldnn_tensor_node = graph.create_node(
                    "call_method", "to_mkldnn", (args[1],)
                )
                constant_args = [args[4], args[3], args[5], args[-1]]
                packed_weight_op = torch._C._nn.mkldnn_reorder_conv2d_weight
                packed_conv_op = mkldnn._convolution_pointwise.default
                if is_transposed:
                    constant_args.insert(1, args[-2])  # output_padding
                    packed_weight_op = mkldnn._reorder_convolution_transpose_weight
                    packed_conv_op = mkldnn._convolution_transpose_pointwise.default

                packed_weight_inputs = (
                    (mkldnn_tensor_node,) + tuple(constant_args) + (input_size,)
                )
                packed_weight_node = graph.create_node(
                    "call_function", packed_weight_op, args=packed_weight_inputs
                )
                packed_conv_inputs = (
                    (args[0], packed_weight_node, args[2])
                    + tuple(constant_args)
                    + ("none", [], "")
                )
                packed_conv_node = graph.create_node(
                    "call_function", packed_conv_op, tuple(packed_conv_inputs)
                )
                conv_node.replace_all_uses_with(packed_conv_node)
                packed_conv_node.meta.update(conv_node.meta)
                graph.erase_node(conv_node)

        @register_freezing_graph_pattern(
            CallFunction(aten.addmm.default, Arg(), Arg(), Arg()),
            extra_check=_is_packable_linear,
        )
        @register_freezing_graph_pattern(
            CallFunction(aten.mm.default, Arg(), Arg()),
            extra_check=_is_packable_linear,
        )
        def linear(match, *args, **kwargs):
            graph = match.graph
            linear_node = match.output_node()
            input = args[0] if linear_node.target == aten.mm.default else args[1]
            bias = None if linear_node.target == aten.mm.default else args[0]
            weight = args[1] if linear_node.target == aten.mm.default else args[2]
            with graph.inserting_before(linear_node):
                transpose_weight_node = graph.create_node(
                    "call_function", aten.permute.default, (weight, (1, 0))
                )
                mkldnn_tensor_node = graph.create_node(
                    "call_method", "to_mkldnn", (transpose_weight_node,)
                )
                is_bf16_weight = weight.meta.get("val").dtype == torch.bfloat16
                batch_size = input.meta.get("val").shape[0]
                packed_weight_inputs = (mkldnn_tensor_node, batch_size)
                packed_weight_op = (
                    mkldnn._reorder_linear_weight
                    if is_bf16_weight
                    else torch.ops.mkl._mkl_reorder_linear_weight
                )
                packed_weight_node = graph.create_node(
                    "call_function", packed_weight_op, args=packed_weight_inputs
                )
                packed_linear_inputs = (input, packed_weight_node)
                if is_bf16_weight:
                    packed_linear_inputs += (bias, "none", [], "")
                    packed_linear_op = mkldnn._linear_pointwise.default
                else:
                    packed_linear_inputs += (transpose_weight_node, bias, batch_size)
                    packed_linear_op = torch.ops.mkl._mkl_linear
                packed_linear_node = graph.create_node(
                    "call_function", packed_linear_op, packed_linear_inputs
                )
                linear_node.replace_all_uses_with(packed_linear_node)
                packed_linear_node.meta.update(linear_node.meta)
                graph.erase_node(linear_node)

    def _eliminate_duplicate_packed_nodes(gm):
        """
        Combine packed weight nodes with the same inputs to reduce memory usage.
        for example:
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32, bias=True)

            def forward(self, x):
                return self.linear(self.linear(x))

        the above's packed weight nodes are duplicate if two linear calls have same input size.
        """
        if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
            return gm
        for node in gm.graph.nodes:
            if node.target == "to_mkldnn" and len(node.args[0].users) > 1:
                for user_node in list(node.args[0].users.keys()):
                    if user_node.target == "to_mkldnn" and user_node != node:
                        user_node.replace_all_uses_with(node)
                        gm.graph.erase_node(user_node)
        packed_weight_ops = [
            torch._C._nn.mkldnn_reorder_conv2d_weight,
            mkldnn._reorder_convolution_transpose_weight,
            mkldnn._reorder_linear_weight,
            torch.ops.mkl._mkl_reorder_linear_weight,
        ]
        for node in gm.graph.nodes:
            if node.target in packed_weight_ops and len(node.args[0].users) > 1:
                for user_node in list(node.args[0].users.keys()):
                    if (
                        user_node.target == node.target
                        and user_node != node
                        and user_node.args == node.args
                    ):
                        user_node.replace_all_uses_with(node)
                        gm.graph.erase_node(user_node)

    @functools.lru_cache(None)
    def _mkldnn_fusion_init():
        if torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available():
            _register_unary_fusion()
            _register_inplace_fusion()
            _register_binary_unary_fusion()
            _register_binary_fusion()

    @functools.lru_cache(None)
    def _mkldnn_weight_pack_init():
        if torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available():
            _register_weight_pack_pass()
            _recover_linear()
