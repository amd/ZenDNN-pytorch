#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************

import torch
from torch.nn.common_types import _size_2_t
from typing import Optional, Union, Tuple, List


class _ZendnnConvNd(torch.jit.ScriptModule):
    """Common base of ZendnnConv1d and ZendnnConv2d"""
    __constants__ = ['stride', 'padding', 'dilation', 'groups']

    def __init__(self, dense_module):
        super(_ZendnnConvNd, self).__init__()

        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups

        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_zendnn())
        else:
            # TODO: Remove this once ScriptModule supports registering None buffer
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_zendnn())

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)

    @torch.jit.script_method
    def forward(self, x):
        return torch.zendnn_convolution(
            x,
            self.weight,
            self.bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups)


class ZendnnConv1d(_ZendnnConvNd):
    def __init__(self, dense_module, dtype):
        super(ZendnnConv1d, self).__init__(dense_module)

        self.register_buffer('weight', dense_module.weight.to_zendnn(dtype))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_zendnn()
        self.bias = state[1].to_zendnn()
        self.training = state[2]


class ZendnnConv2d(_ZendnnConvNd):
    def __init__(self, dense_module, dtype):
        super(ZendnnConv2d, self).__init__(dense_module)

        self.register_buffer('weight', torch._C._nn.zendnn_reorder_conv2d_weight(
            dense_module.weight.to_zendnn(dtype),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = torch._C._nn.zendnn_reorder_conv2d_weight(
            state[0].to_zendnn(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_zendnn()
        self.training = state[2]

class ZendnnConv3d(_ZendnnConvNd):
    def __init__(self, dense_module, dtype):
        super(ZendnnConv3d, self).__init__(dense_module)

        self.register_buffer('weight', torch._C._nn.zendnn_reorder_conv3d_weight(
            dense_module.weight.to_zendnn(dtype),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = torch._C._nn.zendnn_reorder_conv3d_weight(
            state[0].to_zendnn(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_zendnn()
        self.training = state[2]

def to_zendnn(module, dtype=torch.float):
    assert dtype in [torch.float], \
        "ZENDNN only support float path now"

    def m_fn(m, d):

        if isinstance(m, torch.nn.Conv1d):
            return ZendnnConv1d(m, d)
        elif isinstance(m, torch.nn.Conv2d):
            return ZendnnConv2d(m, d)
        elif isinstance(m, torch.nn.Conv3d):
            return ZendnnConv3d(m, d)
        else:
            return m

    def m_fn_rec(m, d):
        new_m = m_fn(m, d)
        for name, sub_m in m.named_children():
            setattr(new_m, name, m_fn_rec(sub_m, d))
        return new_m

    return m_fn_rec(module, dtype)


####################################################################################
#                              Support for ZenVitisAI                              #
####################################################################################


class ZenVitisAIAdd(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, other: torch.Tensor, alpha: int):
        return torch.add(input, other, alpha=alpha)


class ZenVitisAIAvgPool(torch.nn.AvgPool2d):

    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None) -> None:
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


class ZenVitisAIAdaptiveAvgPool(torch.nn.AdaptiveAvgPool2d):

    def __init__(self, output_size: _size_2_t):
        super().__init__(output_size)


class ZenVitisAIConv2D(torch.nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1,
                 quant: bool = True, dequant: bool = False, fuse_relu: bool = False,
                 i_scale: int = -1, f_scale: int = -1, o_scale: int = -1, add_scale: int = -1, add_out_scale: int = -1,
                 bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         groups, bias, padding_mode, device, dtype)
        self.quant = quant
        self.dequant = dequant
        self.fuse_relu = fuse_relu
        self.i_scale = i_scale
        self.f_scale = f_scale
        self.o_scale = o_scale
        self.add_scale = add_scale
        self.add_out_scale = add_out_scale

    def forward(self, input: torch.Tensor, add_input: Optional[torch.Tensor] = None):
        return torch._C._nn.zendnn_vitisai_convolution(
            input, self.weight, self.bias, add_input, self.padding,
            self.stride, self.dilation, self.groups, self.dequant, self.fuse_relu,
            self.i_scale, self.f_scale, self.o_scale, self.add_scale, self.add_out_scale)

    def __repr__(self):
        original_str = super().__repr__()
        new_str = original_str[:-1]
        new_str += ", quant={}".format(self.quant)
        if self.quant:
            new_str += ", dequant={}".format(self.dequant)
            new_str += ", i_scale={}, f_scale={}, o_scale={}".format(self.i_scale, self.f_scale, self.o_scale)
            if self.add_scale and self.add_out_scale != -1:
                new_str += ", add_scale={}, add_out_scale={}".format(self.add_scale, self.add_out_scale)
        new_str += ")"
        return new_str

class ZenVitisAIConcat(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, dim: int, tensors: List[torch.Tensor]):
        return torch.cat(tensors, dim)

class ZenVitisAIInput(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor):
        return input


class ZenVitisAILinear(torch.nn.Linear):

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None,
                 dequant: bool = False, fuse_relu: bool = False,
                 i_scale: int = -1, f_scale: int = -1, o_scale: int = -1) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.dequant = dequant
        self.fuse_relu = fuse_relu
        self.i_scale = i_scale
        self.f_scale = f_scale
        self.o_scale = o_scale

    def forward(self, input: torch.Tensor):
        return torch._C._nn.zendnn_vitisai_linear(
            input, self.weight, self.bias,
            self.dequant, self.fuse_relu,
            self.i_scale, self.f_scale, self.o_scale)


class ZenVitisAIMaxPool(torch.nn.MaxPool2d):

    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)


class ZenVitisAIRelu(torch.nn.ReLU):

    def __init__(self, inplace=False) -> None:
        super().__init__(inplace)


class ZenVitisAIReshape(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, shape: Tuple[torch.Tensor, int]):
        return torch.reshape(input, tuple(shape))


class ZenVitisAIReshapeOpt(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor):
        batch_size = int(list(input.size())[0])
        return torch.reshape(input, (batch_size, -1))


class ZenVitisAISize(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, dim: int):
        # This provides the batch size to the next node
        return torch.tensor(int(list(input.size())[dim]))


# Taken from https://github.com/Xilinx/Vitis-AI/blob/master/src/Vitis-AI-Quantizer/vai_q_pytorch/pytorch_binding/pytorch_nndct/nn/modules/prim_ops.py#L168
# This implementation might be replaced with another proper implementation for June 2023 release
# Implements slicing tensors with start, end and steps
# For example, if the start, end and step are single elements, then
# forward returns input[start:end:step] -> from start to end, use step to index
# If start, end and step are multiple elements, the forward returns
# input[start1:end1:step1, start2:end2:step2,...] -> start to end, use steps, repeated
class ZenVisitAIStridedSlice(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, dim: list,
                start: list, end: list, step: list):
        size = input.size()
        break_symbol = ':'
        symbols = ""
        start_symbol = []
        end_symbol = []
        step_symbol = []
        for i in range(dim[0]):
            start_symbol.append(str(0))
            end_symbol.append(str(int(size[i])))
            step_symbol.append(str(1))

        for i in range(len(start)):
            start_symbol.append(str(int(start[i])))
            end_symbol.append(str(int(end[i])))
            step_symbol.append(str(int(step[i])))

        for i in range(len(start_symbol)):
            slice_symbol = break_symbol.join([start_symbol[i], end_symbol[i], step_symbol[i]])
            if i > 0:
                symbols += "," + slice_symbol
            else:
                symbols = slice_symbol

        eval_str = f"input[{symbols}]"
        output = eval(eval_str)
        return output


# Taken from https://github.com/Xilinx/Vitis-AI/blob/master/src/Vitis-AI-Quantizer/vai_q_pytorch/pytorch_binding/pytorch_nndct/nn/modules/prim_ops.py#L188
# This implementation might be replaced with another proper implementation for June 2023 release
# If the index is a single element, return input[index]
# If the index is a list, then return input[indexes]
# If the input contains None values add slice points to create complex patterns
# like these input[:,index1, index2] and such
class ZenVisiAIIndex(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, index: list):
        if isinstance(index, (list, tuple)):
            break_symbol = ':'
            symbols = ""
            for i in range(len(index)):
                if index[i] is None:
                    slice_symbol = break_symbol
                else:
                    slice_symbol = "index[" + str(i) + "]"
                if i > 0:
                    symbols += "," + slice_symbol
                else:
                    symbols = slice_symbol
            eval_str = f"input[{symbols}]"
            output = eval(eval_str)
        else:
            output = input[index]
        return output


class ZenVitisAIConst(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: torch.Tensor, dtype=torch.int64):
        return torch.tensor(data, dtype=torch.int64)


class ZenVitisAITranspose(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, dim0: int, dim1: int):
        return input.transpose(dim0, dim1)

class ZenVitisAIBMM(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, mat2: torch.Tensor):
        return torch.bmm(input, mat2)
