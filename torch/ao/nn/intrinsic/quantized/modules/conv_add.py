import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.nn.functional as F
import torch.ao.nn.quantized as nnq

from torch.nn.utils import fuse_conv_bn_weights

_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding

class Conv2dAdd(nnq.Conv2d):
    r"""
    A Conv2dAdd module is a fused module of Conv2d and Add

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.Conv2dAdd  # type: ignore[assignment]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        super(Conv2dAdd, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

    def forward(self, input, extra_input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != 'zeros':
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        return torch.ops.quantized.conv2d_add(
            input, extra_input, self._packed_params, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedConv2dAdd'

    @classmethod
    def from_float(cls, mod):
        print("mod is: {}".format(mod))
        return super(Conv2dAdd, cls).from_float(mod)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        print("ref_qconv is: {}".format(ref_qconv))
        assert type(ref_qconv) != torch.nn.intrinsic.ConvBnReLU2d, \
            "BatchNorm2d should be fused into Conv2d before converting to reference module"
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)
