#******************************************************************************
# Modifications Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#******************************************************************************

#******************************************************************************
# Steps for Usage:
# 1. To execute all unit tests in this file, run the following command:
#    python test_zendnn.py
# 2. To execute a particular unit test in this file, run the following command:
#    python test_zendnn.py <test_name>
#    e.g., python test_zendnn.py TestZENDNN.test_relu
#******************************************************************************

from torch.testing._internal.common_utils import TestCase, \
    run_tests, gradcheck, gradgradcheck, set_default_dtype
from torch.testing import FileCheck
from torch.utils import zendnn as zendnn_utils
import torch.backends.zendnn
import torch.jit
import torch.nn.functional as F
import torch
from collections import OrderedDict
import itertools
import functools
import unittest

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


# batched grad doesn't support zendnn
gradcheck = functools.partial(gradcheck, check_batched_grad=False)
gradgradcheck = functools.partial(gradgradcheck, check_batched_grad=False)

# For ZenDNN bf16 path, ZenDNN requires the cpu which has amd avx512 with
# avx512bf16 at least. So we will skip the test case if one processor
# is not meet the requirement.
@functools.lru_cache(maxsize=None)
def has_bf16_support():
    import sys
    if sys.platform != 'linux':
        return False
    with open("/proc/cpuinfo", encoding="ascii") as f:
        lines = f.read()
    return all(word in lines for word in ["avx512_bf16"])

types = [torch.float, torch.bfloat16]

# Comment the line below to find out the CI machines having ZENDNN build disabled


@unittest.skipIf(not torch._C.has_zendnn, "ZENDNN build is disabled")
class TestZENDNN(TestCase):
    def test_conversion(self):
        for cpu_tensor in [torch.randn((1, 2, 3, 4),
                                       dtype=torch.float, device=torch.device('cpu')),
                           torch.randn((1, 2, 3, 4, 5),
                                       dtype=torch.float, device=torch.device('cpu'))[:, :, :, :, 1]]:
            cpu_tensor.requires_grad_()
            # float cpu tensor to zendnn float tensor
            for dtype1 in types:
                zendnn_tensor = cpu_tensor.to_zendnn(dtype1)
                self.assertEqual(zendnn_tensor.dtype, dtype1)
                cpu_tensor_1 = zendnn_tensor.to_dense()
                # not given dtype for to_dense, zendnn tensor has same dtype with cpu tensor
                self.assertEqual(zendnn_tensor.dtype, cpu_tensor_1.dtype)
                # zendnn float tensor to cpu float
                for dtype2 in types:
                    cpu_tensor_2 = zendnn_tensor.to_dense(dtype2)
                    self.assertEqual(cpu_tensor_2.dtype, dtype2)
                    atol = 1e-5 if dtype1 == torch.float and dtype2 == torch.float else 1e-2
                    self.assertEqual(
                        cpu_tensor, cpu_tensor_2.float(), atol=atol, rtol=0)

                self.assertEqual(zendnn_tensor.device, torch.device('cpu'))
                self.assertEqual(zendnn_tensor.size(),
                                 torch.Size([1, 2, 3, 4]))
                self.assertEqual(zendnn_tensor.numel(), cpu_tensor.numel())
                if dtype1 == torch.float:
                    self.assertEqual(zendnn_tensor.element_size(),
                                     cpu_tensor.element_size())
                else:
                    self.assertEqual(zendnn_tensor.element_size(),
                                     cpu_tensor.element_size() / 2)
                self.assertRaisesRegex(RuntimeError,
                                       "Cannot access data pointer of Tensor that doesn't have storage",
                                       lambda: zendnn_tensor.data_ptr() != 0)

    def test_copy(self):
        x = torch.randn(4, 5, dtype=torch.float32)
        zendnn_x = x.to_zendnn()
        zendnn_y = torch.randn(4, 5, dtype=torch.float32).to_zendnn()
        zendnn_z = torch.randn(4, 10, dtype=torch.float32).to_zendnn()
        zendnn_y.copy_(zendnn_x)
        self.assertEqual(x, zendnn_y.to_dense())
        self.assertRaisesRegex(RuntimeError,
                               "copy_zendnn_: only support same size tensor.",
                               lambda: zendnn_z.copy_(zendnn_x))
        self.assertRaisesRegex(RuntimeError,
                               "copy_zendnn_: between zendnn layout and dense Tensors is not implemented! "
                               "Found self type = torch.FloatTensor and src type = Zendnntorch.FloatTensor",
                               lambda: x.copy_(zendnn_x))
        self.assertRaisesRegex(RuntimeError,
                               "copy_zendnn_: between zendnn layout and dense Tensors is not implemented! "
                               "Found self type = Zendnntorch.FloatTensor and src type = torch.FloatTensor",
                               lambda: zendnn_x.copy_(x))

    def test_unsupported(self):
        # unsupported types
        for dtype in [torch.double, torch.half, torch.uint8, torch.int8,
                      torch.short, torch.int, torch.long]:
            with self.assertRaises(RuntimeError) as context:
                torch.randn(1, 2, 3, 4, dtype=dtype,
                            device=torch.device('cpu')).to_zendnn()
        # some factory functions
        for creator in [torch.ones, torch.randn, torch.rand]:
            with self.assertRaises(RuntimeError) as context:
                creator(1, 2, 3, 4, dtype=torch.float,
                        device=torch.device('cpu'), layout=torch._zendnn)

    def test_zendnn_conv_shapecheck(self):
        input = torch.full((1, 1, 1, 24,), 1, dtype=torch.float32)
        w1 = torch.full((1, 1, 1, 24,), 1, dtype=torch.float32)
        b1 = torch.full((1,), 1, dtype=torch.float32)
        w2 = torch.full((1, 1, 2, 24,), 1, dtype=torch.float32)
        b2 = torch.full((2,), 1, dtype=torch.float32)
        options = zip([-1, 0, 0, 0, 0, 0, 0],  # padding
                      [1, 0, 1, 1, 1, 1, 1],  # stride
                      [1, 1, 0, 1, 1, 1, 1],  # dilation
                      [1, 1, 1, 0, 2, 1, 1],  # groups
                      [w1, w1, w1, w1, w1, w1, w2],  # weight
                      [b1, b1, b1, b1, b1, b2, b1])  # bias
        for pad, st, dil, gr, w, b in options:
            with self.assertRaises(RuntimeError) as _:
                torch.zendnn_convolution(input, w, b, [pad] * 2, [st] * 2, [dil] * 2, gr)

    def test_detach(self):
        root = torch.randn(
            4, 5, dtype=torch.float32).to_zendnn().requires_grad_()
        detach = root.detach()
        self.assertEqual((4, 5), detach.size())
        self.assertFalse(detach.requires_grad)
        self.assertTrue(root.requires_grad)

        detach_ = root.detach_()
        self.assertEqual((4, 5), detach_.size())
        self.assertFalse(detach_.requires_grad)
        self.assertFalse(root.requires_grad)

    def test_repr(self):
        self.assertTrue("layout=torch._zendnn" in str(torch.randn((1, 2, 3, 4),
                                                      dtype=torch.float, device=torch.device('cpu')).to_zendnn()))

    def test_is_zendnn(self):
        x = torch.randn(1, dtype=torch.float32)
        self.assertFalse(x.is_zendnn)
        self.assertTrue(x.to_zendnn().is_zendnn)

    def test_set_data_tensorimpl_type(self):
        # Dense tensor has impl of type `TensorImpl`, while ZENDNN tensor has impl
        # of type `OpaqueTensorImpl<IDeepTensorWrapperPtr>`.
        x = torch.randn((1, 2), dtype=torch.float, device=torch.device('cpu'))
        x_zendnn = x.to_zendnn()
        with self.assertRaisesRegex(RuntimeError, 'incompatible tensor type'):
            x.data = x_zendnn

    # legacy constructor/new doesn't support zendnn tensors
    def test_legacy_new_failure(self):
        x = torch.randn(1, dtype=torch.float32)
        x_zendnn = x.to_zendnn()
        self.assertRaises(RuntimeError, lambda: x_zendnn.new(device='cpu'))
        self.assertRaises(RuntimeError, lambda: x_zendnn.new(x.storage()))
        self.assertRaises(RuntimeError, lambda: x_zendnn.new(x))
        self.assertRaises(
            RuntimeError, lambda: x_zendnn.new(torch.Size([2, 3])))
        self.assertRaises(RuntimeError, lambda: x_zendnn.new([6]))

    def test_is_zendnn_jit(self):
        class EnsureZENDNN(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if not x.is_zendnn:
                    x = x.to_zendnn()
                return x

        m = EnsureZENDNN()
        x = torch.randn(1, dtype=torch.float32)
        self.assertTrue(m(x).is_zendnn)
        self.assertTrue(m(x.to_zendnn()).is_zendnn)

    def test_empty(self):
        x1 = torch.empty(4, 5, 2, 3, dtype=torch.float32)
        x2 = torch.empty(4, 5, 2, 3, dtype=torch.float32, layout=torch._zendnn)
        self.assertEqual(x1.size(), x2.to_dense().size())
        self.assertEqual(x1.dtype, x2.to_dense().dtype)

    @torch.no_grad()
    def test_conv1d(self):
        options = itertools.product([1, 4], [True, False], [1, 2])
        for groups, bias, dilation in options:
            N = torch.randint(3, 10, (1,)).item()
            C = torch.randint(1, 3, (1,)).item() * groups
            M = torch.randint(1, 3, (1,)).item() * groups
            x = torch.randn(N, C, 224, dtype=torch.float32)
            conv1d = torch.nn.Conv1d(in_channels=C,
                                     out_channels=M,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     dilation=dilation,
                                     bias=bias,
                                     groups=groups).float()

            y_zendnn = conv1d(x)
            with torch.backends.zendnn.flags(enabled=False):
                y_aten = conv1d(x)
            self.assertEqual(y_aten, y_zendnn)
            self._test_tracing(conv1d, (x,))
            self._test_scripting(conv1d, (x,))

    @torch.no_grad()
    def test_embeddingbag(self):
        R = torch.randint(11, 20, (1,)).item()
        W = torch.randint(1, 9, (1,)).item()
        input = torch.randint(1, 15, (W,))
        offsets = torch.tensor([0, W], dtype=torch.long)
        emb_d = torch.nn.EmbeddingBag(num_embeddings=R,
                                      embedding_dim=W,
                                      scale_grad_by_freq=False,
                                      mode="sum",
                                      sparse=True,
                                      include_last_offset=False,
                                      _weight=None,
                                      padding_idx=None)
        y_zendnn = emb_d(input, offsets)
        with torch.backends.zendnn.flags(enabled=False):
            y_aten = emb_d(input, offsets)
        self.assertEqual(y_aten, y_zendnn)

    @torch.no_grad()
    def test_conv2d(self):
        options = itertools.product([1, 4], [True, False], [1, 2])
        for groups, bias, dilation in options:
            N = torch.randint(3, 10, (1,)).item()
            C = torch.randint(1, 3, (1,)).item() * groups
            M = torch.randint(1, 3, (1,)).item() * groups
            x = torch.randn(N, C, 224, 224, dtype=torch.float32)
            conv2d = torch.nn.Conv2d(in_channels=C,
                                     out_channels=M,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     dilation=dilation,
                                     bias=bias,
                                     groups=groups).float()
            y_zendnn = conv2d(x)
            with torch.backends.zendnn.flags(enabled=False):
                y_aten = conv2d(x)
            self.assertEqual(y_aten, y_zendnn)
            self._test_tracing(conv2d, (x,))
            self._test_scripting(conv2d, (x,))

    @torch.no_grad()
    def test_conv3d(self):
        options = itertools.product([1, 4], [True, False], [1, 2])
        for groups, bias, dilation in options:
            N = torch.randint(3, 10, (1,)).item()
            C = torch.randint(1, 3, (1,)).item() * groups
            M = torch.randint(1, 3, (1,)).item() * groups
            x = torch.randn(N, C, 55, 55, 55, dtype=torch.float32)
            conv3d = torch.nn.Conv3d(in_channels=C,
                                     out_channels=M,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=bias,
                                     dilation=dilation,
                                     groups=groups).float()
            with torch.backends.zendnn.flags(enabled=False):
                y_aten = conv3d(x)
            y_zendnn = conv3d(x)
            self.assertEqual(y_aten, y_zendnn)
            self._test_tracing(conv3d, (x,))
            self._test_scripting(conv3d, (x,))

    def _test_conv_bf16_base(self, dim):
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        options = itertools.product([True, False], [1, 2], [1, 4])
        for bias, dilation, groups in options:
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32)

            conv = conv_module[dim](in_channels=C,
                                    out_channels=M,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=dilation,
                                    bias=bias,
                                    groups=groups)

            x_bf16 = x.bfloat16()
            if has_bf16_support():
                y_zen = conv(x)

                bf16_conv = conv.bfloat16()
                with torch.backends.zendnn.flags(enabled=False):
                    y_native_bf16 = bf16_conv(x_bf16)
                y_zen_bf16 = bf16_conv(x_bf16)

                self.assertEqual(y_native_bf16, y_zen_bf16, atol=1e-1, rtol=1e-3)
                self.assertEqual(y_zen, y_zen_bf16.to_dense(torch.float32), atol=1e-1, rtol=1e-3)
            else:
                bf16_conv = conv.bfloat16()
                msg = "zendnn_convolution: bf16 path needs the cpu support avx512bf16"
                with self.assertRaisesRegex(RuntimeError, msg):
                    y_bf16 = bf16_conv(x_bf16)

    @torch.no_grad()
    def test_conv1d_bf16(self):
        self._test_conv_bf16_base(dim=1)

    @torch.no_grad()
    def test_conv2d_bf16(self):
        self._test_conv_bf16_base(dim=2)

    @torch.no_grad()
    def test_conv3d_bf16(self):
        self._test_conv_bf16_base(dim=3)

    @torch.no_grad()
    def test_linear(self):
        options = [True, False]
        for bias in options:
            N = torch.randint(3, 64, (1,)).item()
            in_feats = torch.randint(1, 64, (1,)).item()
            out_feats = torch.randint(1, 64, (1,)).item()
            x = torch.randn((N, in_feats), dtype=torch.float32)

            linear = torch.nn.Linear(
                in_features=in_feats, out_features=out_feats, bias=bias)

            self.assertEqual(linear(x), linear(x.to_zendnn()).to_dense())
            self._test_tracing(linear, (x,))
            self._test_scripting(linear, (x,))

    def test_linear_bf16(self):
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        x_bf16 = x.bfloat16()

        for bias in [True, False]:
            linear = torch.nn.Linear(in_features, out_features, bias=bias)
            if has_bf16_support():
                y_zen = linear(x.to_zendnn()).to_dense()

                bf16_linear = linear.bfloat16()
                y_native_bf16 = bf16_linear(x_bf16)
                y_zen_bf16 = bf16_linear(x_bf16.to_zendnn()).to_dense(torch.bfloat16)

                self.assertEqual(y_native_bf16, y_zen_bf16, atol=1e-1, rtol=1e-3)
                self.assertEqual(y_zen, y_zen_bf16.to_dense(torch.float32), atol=1e-1, rtol=1e-3)
            else:
                msg = "zendnn_linear: bf16 path needs the cpu support avx512bf16"
                self.assertRaisesRegex(RuntimeError,
                                       msg,
                                       lambda: linear(x_bf16.to_zendnn()))

    # This test is to check whether 1D conv is supported for zendnn tensor,
    @torch.no_grad()
    def test_conv1d_functional(self):
        input = torch.randn(2, 3, 10).to_zendnn()
        weight = torch.randn(3, 3, 3).to_zendnn()
        bias = torch.randn(3).to_zendnn()
        output = torch.nn.functional.conv1d(input, weight, bias)
        self.assertEqual(output.size(), torch.Size([2, 3, 8]))

    def _test_tracing(self, module, inputs):
        traced = torch.jit.trace(module, inputs)
        self.assertEqual(
            module(*inputs),
            traced(*inputs))

    def _test_scripting(self, module, inputs):
        scripted = torch.jit.script(module)
        self.assertEqual(
            module(*inputs),
            scripted(*inputs))

    @torch.no_grad()
    def _test_imagenet_model(self, model):
        model = model.train(False).float()
        x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        with torch.backends.zendnn.flags(enabled=False):
            y_aten = model(x)
        y_zen = model(x)
        self.assertEqual(y_aten, y_zen)

    @skipIfNoTorchVision
    def test_resnet18(self):
        model = torchvision.models.resnet.resnet18(weights=None)
        self._test_imagenet_model(model)

    @skipIfNoTorchVision
    def test_resnext50_32x4d(self):
        model = torchvision.models.resnet.resnext50_32x4d(weights=None)
        self._test_imagenet_model(model)

    def test_max_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
                x = torch.randn(N, C, H, W, dtype=torch.float32) * 10

                for ceil_mode in [False, True]:
                    max_pool2d = torch.nn.MaxPool2d(
                        kernel_size=3 if not ceil_mode else 7,
                        stride=stride,
                        padding=1,
                        ceil_mode=ceil_mode)

                    self.assertEqual(
                        max_pool2d(x),
                        max_pool2d(x.to_zendnn()).to_dense())

    def test_max_pool2d_stride_none(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            for ceil_mode in [False, True]:
                y1 = F.max_pool2d(
                    x,
                    kernel_size=3 if not ceil_mode else 7,
                    stride=None,
                    padding=1,
                    ceil_mode=ceil_mode)

                y2 = F.max_pool2d(
                    x.to_zendnn(),
                    kernel_size=3 if not ceil_mode else 7,
                    stride=None,
                    padding=1,
                    ceil_mode=ceil_mode)

                self.assertEqual(y1, y2.to_dense())

    def test_max_pool3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
                x = torch.randn(N, C, D, H, W, dtype=torch.float32) * 10

                for ceil_mode in [False, True]:
                    max_pool3d = torch.nn.MaxPool3d(
                        kernel_size=3 if not ceil_mode else 7,
                        stride=stride,
                        padding=1,
                        ceil_mode=ceil_mode)

                    self.assertEqual(
                        max_pool3d(x),
                        max_pool3d(x.to_zendnn()).to_dense())

    def _test_max_pool_bf16_base(self, dim, input):
        pool_module = {2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        x_bf16 = input.bfloat16()
        for stride in [1, 2, 3]:
            for ceil_mode in [False, True]:
                max_pool = pool_module[dim](
                    kernel_size=3 if not ceil_mode else 7,
                    stride=stride,
                    padding=1,
                    ceil_mode=ceil_mode)

                if has_bf16_support():
                    y_zen = max_pool(input.to_zendnn()).to_dense()

                    bf16_max_pool = max_pool.bfloat16()
                    y_native_bf16 = bf16_max_pool(x_bf16)
                    y_zen_bf16 = bf16_max_pool(x_bf16.to_zendnn()).to_dense(torch.bfloat16)

                    self.assertEqual(y_native_bf16, y_zen_bf16, atol=1e-1, rtol=1e-3)
                    self.assertEqual(y_zen, y_zen_bf16.to_dense(torch.float32), atol=0.1, rtol=1e-3)
                else:
                    msg = f"zendnn_max_pool{dim}d: bf16 path needs the cpu support avx512bf16"
                    self.assertRaisesRegex(RuntimeError,
                                           msg,
                                           lambda: max_pool(x_bf16.to_zendnn()))

    def test_max_pool2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            self._test_max_pool_bf16_base(dim=2, input=x)

    def test_max_pool_unsupported(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        # 2d dilation case
        x = torch.randn(N, C, 7, 7, dtype=torch.float32).to_zendnn()
        max_pool2d = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=3,
            padding=1,
            dilation=2)
        self.assertRaisesRegex(RuntimeError,
                               'zendnn_max_pool2d does not support dilation case',
                               lambda: max_pool2d(x))

        # 3d dilation case
        x = torch.randn(N, C, 7, 7, 7, dtype=torch.float32).to_zendnn()
        max_pool3d = torch.nn.MaxPool3d(
            kernel_size=3,
            stride=3,
            padding=1,
            dilation=2)
        self.assertRaisesRegex(RuntimeError,
                               'zendnn_max_pool3d does not support dilation case',
                               lambda: max_pool3d(x))

    def test_avg_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10

        for count_include_pad in [True, False]:
            avg_pool2d = torch.nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            self.assertEqual(
                avg_pool2d(x),
                avg_pool2d(x.to_zendnn()).to_dense())

    def test_avg_pool2d_stride_none(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10

        for count_include_pad in [True, False]:
            y1 = F.avg_pool2d(
                x,
                kernel_size=3,
                stride=None,
                padding=1,
                count_include_pad=count_include_pad)
            y2 = F.avg_pool2d(
                x.to_zendnn(),
                kernel_size=3,
                stride=None,
                padding=1,
                count_include_pad=count_include_pad)

            self.assertEqual(y1, y2.to_dense())

    def test_avg_pool3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, 64, dtype=torch.float32) * 10

        for count_include_pad in [True, False]:
            avg_pool3d = torch.nn.AvgPool3d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            self.assertEqual(
                avg_pool3d(x),
                avg_pool3d(x.to_zendnn()).to_dense())

    def _test_avg_pool_bf16_base(self, dim, input):
        avg_module = {2: torch.nn.AvgPool2d, 3: torch.nn.AvgPool3d}
        x_bf16 = input.bfloat16()
        for count_include_pad in [True, False]:
            avg_pool = avg_module[dim](
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)
            if has_bf16_support():
                y_zen = avg_pool(input.to_zendnn()).to_dense()

                bf16_avg_pool = avg_pool.bfloat16()
                y_native_bf16 = bf16_avg_pool(x_bf16)
                y_zen_bf16 = bf16_avg_pool(x_bf16.to_zendnn()).to_dense(torch.bfloat16)

                self.assertEqual(y_native_bf16, y_zen_bf16, atol=1e-1, rtol=1e-3)
                self.assertEqual(y_zen, y_zen_bf16.to_dense(torch.float32), atol=1e-1, rtol=1e-3)
            else:
                msg = f"zendnn_avg_pool{dim}d: bf16 path needs the cpu support avx512bf16"
                self.assertRaisesRegex(RuntimeError,
                                       msg,
                                       lambda: avg_pool(x_bf16.to_zendnn()))

    def test_avg_pool2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_bf16_base(dim=2, input=x)

    def test_adaptive_avg_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

        self.assertEqual(
            adaptive_avg_pool2d(x),
            adaptive_avg_pool2d(x.to_zendnn()).to_dense())

    def test_adaptive_avg_pool2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        x_bf16 = x.bfloat16()
        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

        if has_bf16_support():
            y_zen = adaptive_avg_pool2d(x.to_zendnn()).to_dense()

            bf16_adaptive_avg_pool2d = adaptive_avg_pool2d.bfloat16()
            y_native_bf16 = bf16_adaptive_avg_pool2d(x_bf16)
            y_zen_bf16 = bf16_adaptive_avg_pool2d(x_bf16.to_zendnn()).to_dense(torch.bfloat16)

            self.assertEqual(y_native_bf16, y_zen_bf16, atol=1e-1, rtol=1e-3)
            self.assertEqual(y_zen, y_zen_bf16.to_dense(torch.float32), atol=1e-1, rtol=1e-3)
        else:
            msg = "zendnn_adaptive_avg_pool2d: bf16 path needs the cpu support avx512bf16"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: adaptive_avg_pool2d(x_bf16.to_zendnn()))

    def test_relu(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        self.assertEqual(torch.relu(x), torch.relu(x.to_zendnn()).to_dense())

    def test_relu_(self):
        x1 = torch.randn((4, 5), dtype=torch.float32) * 10
        x2 = x1.clone().to_zendnn()
        self.assertEqual(torch.relu_(x1), torch.relu_(x2).to_dense())

    def _test_relu_bf16_base(self, name):
        x = torch.randn((4, 6), dtype=torch.float32) * 10
        x_bf16 = x.bfloat16()
        fn = getattr(torch, name)
        if has_bf16_support():
            y_zen = fn(x.to_zendnn()).to_dense()

            y_native_bf16 = fn(x_bf16)
            y_zen_bf16 = fn(x_bf16.to_zendnn()).to_dense(torch.bfloat16)

            self.assertEqual(y_native_bf16, y_zen_bf16, atol=1e-1, rtol=1e-3)
            self.assertEqual(y_zen, y_zen_bf16.to_dense(torch.float32), atol=1e-1, rtol=1e-3)
        else:
            msg = "zendnn_" + name + ": bf16 path needs the cpu support avx512bf16"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: fn(x_bf16.to_zendnn()))

    def test_relu_bf16(self):
        self._test_relu_bf16_base("relu")

    def test_relu_inplace_bf16(self):
        self._test_relu_bf16_base("relu_")

    def test_concat(self):
        for dim in range(0, 4):
            N, C = torch.randint(3, 10, (1,)), torch.randint(3, 10, (1,))
            for H, W in zip(torch.randint(24, 256, (5,)), torch.randint(24, 256, (5,))):
                offset = torch.randint(1, 10, (1,))
                if dim == 0:
                    x, y = torch.randn(N, C, H, W, dtype=torch.float32) * \
                        10, torch.randn(N+offset, C, H, W,
                                        dtype=torch.float32) * 10
                elif dim == 1:
                    x, y = torch.randn(N, C+offset, H, W, dtype=torch.float32) * \
                        10, torch.randn(N, C, H, W, dtype=torch.float32) * 10
                elif dim == 2:
                    x, y = torch.randn(N, C, H, W, dtype=torch.float32) * \
                        10, torch.randn(N, C, H+offset, W,
                                        dtype=torch.float32) * 10
                else:
                    x, y = torch.randn(N, C, H, W+offset, dtype=torch.float32) * \
                        10, torch.randn(N, C, H, W, dtype=torch.float32) * 10
                self.assertEqual(torch.cat([x, y], dim), torch.cat(
                    [x.to_zendnn(), y.to_zendnn()], dim).to_dense())

        # Check Failing Case
        x, y = torch.randn(N, C, H, W, dtype=torch.float32) * \
            10, torch.randn(N+offset, C, H, W, dtype=torch.float32) * 10
        with self.assertRaises(RuntimeError) as context:
            torch.cat([x.to_zendnn(), y], 0).to_dense()
        self.assertTrue('zendnn_concat expects all the input tensors should be of type zendnn' == str(
            context.exception))

    def _test_batch_norm_base(self, dim, channels, input):
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        bn = bn_module[dim](channels).float().train(False)
        self.assertEqual(bn(input.to_zendnn()).to_dense(), bn(input))

    @torch.no_grad()
    def test_batch_norm_2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        self._test_batch_norm_base(dim=2, channels=C, input=x)

    @torch.no_grad()
    def test_batch_norm_3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 30, 30, 30, dtype=torch.float32) * 10
        self._test_batch_norm_base(dim=3, channels=C, input=x)

    def test_gelu(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        gelu = torch.nn.GELU()
        self.assertEqual(gelu(x), gelu(x.to_zendnn()).to_dense())

    def test_add(self):
        x = torch.randn((4, 4), dtype=torch.float32) * 10
        y = torch.randn((4, 4), dtype=torch.float32) * 10
        self.assertEqual(torch.add(x, y), torch.add(
            x.to_zendnn(), y.to_zendnn()).to_dense())

    def test_add_(self):
        x = torch.randn((4, 4), dtype=torch.float32) * 10
        y = torch.randn((4, 4), dtype=torch.float32) * 10
        mx = x.clone().to_zendnn()
        my = y.clone().to_zendnn()
        self.assertEqual(x.add_(y), mx.add_(my).to_dense())

    def test_add_out(self):
        x = torch.randn((4, 4), dtype=torch.float32) * 10
        y = torch.randn((4, 4), dtype=torch.float32) * 10
        res1 = torch.ones((4, 4), dtype=torch.float32)
        res2 = torch.ones((4, 4), dtype=torch.float32).to_zendnn()
        torch.add(x, y, alpha=10, out=res1)
        torch.add(x.to_zendnn(), y.to_zendnn(), alpha=10, out=res2)
        self.assertEqual(res1, res2.to_dense())

    def test_mul(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        value = torch.randn(1, dtype=torch.float32).item()

        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        y = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        mx = x.to_zendnn()
        my = y.to_zendnn()

        # mul
        self.assertEqual(
            x * y,
            (mx * my).to_dense())

        self.assertEqual(
            x * value,
            (mx * value).to_dense())

        self.assertEqual(
            torch.mul(x, y),
            torch.mul(mx, my).to_dense())

        self.assertEqual(
            torch.mul(x, value),
            torch.mul(mx, value).to_dense())

        # mul_
        x *= y
        mx *= my
        self.assertEqual(x, mx.to_dense())

        x *= value
        mx *= value
        self.assertEqual(x, mx.to_dense())

        # mul_out
        out = x.clone()
        zendnn_out = out.to_zendnn()
        torch.mul(x, y, out=out)
        torch.mul(mx, my, out=zendnn_out)
        self.assertEqual(out, zendnn_out.to_dense())

        out = x.clone()
        zendnn_out = out.to_zendnn()
        torch.mul(x, value, out=out)
        torch.mul(mx, value, out=zendnn_out)
        self.assertEqual(out, zendnn_out.to_dense())

    def test_mul_(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        y = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        mx = x.clone().to_zendnn()
        my = y.clone().to_zendnn()

        self.assertEqual(x.mul_(y), mx.mul_(my).to_dense())

    def test_mul_out(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()

        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        y = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        z = torch.ones(N, C, 35, 45, dtype=torch.float32) * 10

        mx = x.to_zendnn()
        my = y.to_zendnn()
        mz = z.to_zendnn()

        torch.mul(x, y, out=z)
        torch.mul(mx, my, out=mz)

        self.assertEqual(z, mz.to_dense())

    def test_reshape(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        size = (x.size(0), -1)

        self.assertEqual(
            x.reshape(size),
            x.to_zendnn().reshape(size).to_dense(),
        )
        # test whether share same memory for plain format tensor
        y = x.to_zendnn()
        z = y.reshape(size).add_(y.reshape(size))
        self.assertEqual(
            y.reshape(size).to_dense(),
            z.to_dense(),
        )

    def test_reshape_blocked_format(self):
        # construct an zendnn blocked tensor with zendnn conv2d
        C = 7
        m = zendnn_utils.to_zendnn(torch.nn.Conv2d(C, C, 3))
        x = torch.randn(1, C, 8, 8).to_zendnn()

        # zendnn tensor w/ blocked format
        y_block = m(x)
        # aten tensor w/ plain format
        y_plain = y_block.to_dense()

        y_block_reshape = y_block.reshape(C, -1)
        y_plain_reshape = y_plain.reshape(C, -1)

        self.assertEqual(y_plain_reshape, y_block_reshape.to_dense())

    def test_sigmoid(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        sigmoid = torch.nn.Sigmoid()
        self.assertEqual(sigmoid(x), sigmoid(x.to_zendnn()).to_dense())

    def test_sigmoid_(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        y = x.to_zendnn()
        self.assertEqual(torch.sigmoid_(x), torch.sigmoid_(y).to_dense())

    def test_softmax(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        for dim in range(x.ndim):
            softmax = torch.nn.Softmax(dim=dim)
            self.assertEqual(softmax(x), softmax(x.to_zendnn()).to_dense())

    def test__softmax(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        self.assertEqual(torch._softmax(x, 0, False), torch._softmax(
            x.to_zendnn(), 0, False).to_dense())

    def test_transpose(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        self.assertEqual(torch.transpose(x, 0, 1),
                         torch.transpose(x.to_zendnn(), 0, 1).to_dense())

    def test_transpose_(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        with self.assertRaises(RuntimeError) as context:
            torch._zendnn_transpose_(x.to_zendnn(), 0, 1).to_dense()

    def test_clone(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        self.assertEqual(torch.clone(x), torch.clone(x.to_zendnn()).to_dense())

    def test_zero_(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        self.assertEqual(x.zero_(), x.to_zendnn().zero_().to_dense())

    def test_view(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        self.assertEqual(x.view(-1, 2), x.to_zendnn().view(-1, 2).to_dense())

    def test_to_dense_list(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        self.assertEqual((x, x), torch.to_dense_list(
            (x.to_zendnn(), x.to_zendnn())))

    def test__to_dense(self):
        x = torch.randn((4, 5), dtype=torch.float32)
        self.assertEqual(x, x.to_zendnn()._to_dense())

    def test_layer_norm(self):
        batch, sentence_length, embedding_dim = 20, 5, 10
        embedding = torch.randn(batch, sentence_length, embedding_dim)
        layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.assertEqual(layer_norm(embedding), layer_norm(
            embedding.to_zendnn()).to_dense())

    def test_zendnn_convolution_relu(self):
        class get_model(torch.nn.Module):
            def __init__(self, mdl):
                super(get_model, self).__init__()
                self.features = torch.nn.Sequential(mdl)

            def forward(self, img):
                output = self.features(img)
                return output

        def optimize_model(model):
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)
            return model

        model = get_model(OrderedDict(
            [('0', torch.nn.Conv2d(3, 3, 1)), ('1', torch.nn.ReLU())]))
        model.eval()
        opt_model = optimize_model(model)
        input = torch.randn((1, 3, 3, 1), dtype=torch.float32)
        input2 = torch.randn((3, 3, 1), dtype=torch.float32)
        # print(opt_model.code)
        self.assertEqual(model(input), opt_model(input))
        # Enable it when the bug is fixed
        # self.assertEqual(model(input2),opt_model(input2))

    @torch.no_grad()
    def test_linear_gelu(self):
        class get_model(torch.nn.Module):
            def __init__(self, mdl):
                super(get_model, self).__init__()
                self.features = torch.nn.Sequential(mdl)

            def forward(self, img):
                output = self.features(img)
                return output

        def optimize_model(model):
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)
            return model

        options = [True, False]
        for bias in options:
            N = torch.randint(3, 64, (1,)).item()
            in_feats = torch.randint(1, 64, (1,)).item()
            out_feats = torch.randint(1, 64, (1,)).item()
            x = torch.randn((N, in_feats), dtype=torch.float32)
            model = get_model(OrderedDict([('0', torch.nn.Linear(
                in_features=in_feats, out_features=out_feats, bias=bias)), ('1', torch.nn.GELU())]))
            model.eval()
            opt_model = optimize_model(model)
            # print(opt_model.code)
            self.assertEqual(model(x), opt_model(x))

    def test_linear_non_contiguous_weight(self):
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        w = torch.randn(in_features, out_features, dtype=torch.float32)
        for bias in [True, False]:
            x1 = x.clone().requires_grad_()
            x2 = x.clone().to_zendnn().requires_grad_()
            linear = torch.nn.Linear(
                in_features, out_features, bias=bias).float()
            linear.weight = torch.nn.Parameter(w.t())
            y1 = linear(x1)
            y2 = linear(x2)
            self.assertEqual(y1, y2.to_dense())

    def _test_prelu_base(self, size, num_channels):
        x = torch.randn(size, dtype=torch.float32)
        x1 = x.clone()
        x2 = x.clone().to_zendnn()
        prelu = torch.nn.PReLU(num_channels)
        y1 = prelu(x1)
        y2 = prelu(x2).to_dense()
        self.assertEqual(y1, y2)

    @torch.no_grad()
    def test_prelu(self):
        self._test_prelu_base(torch.Size([16]), 1)
        self._test_prelu_base(torch.Size([16, 64]), 1)
        self._test_prelu_base(torch.Size([16, 64]), 64)
        self._test_prelu_base(torch.Size([16, 64, 112]), 1)
        self._test_prelu_base(torch.Size([16, 64, 112]), 64)
        self._test_prelu_base(torch.Size([16, 64, 112, 112]), 1)
        self._test_prelu_base(torch.Size([16, 64, 112, 112]), 64)
        self._test_prelu_base(torch.Size([16, 64, 112, 112, 1]), 1)
        self._test_prelu_base(torch.Size([16, 64, 112, 112, 1]), 64)

    @torch.no_grad()
    def test_tanh(self):
        x = torch.randn(4, 5, dtype=torch.float32) * 10
        zendnn_x = x.to_zendnn()
        self.assertEqual(
            torch.tanh(x),
            torch.tanh(zendnn_x).to_dense(),
        )
        # inplace
        torch.tanh_(x)
        torch.tanh_(zendnn_x)
        self.assertEqual(x, zendnn_x.to_dense())

    @torch.no_grad()
    def test_scalar_mul(self):
        with set_default_dtype(torch.float):
            class Mod(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mod = torch.nn.Linear(20, 20)

                def forward(self, x):
                    a1 = self.mod(x) * 4
                    return a1 * 4 + a1 * 5.

            mod = Mod().eval()
            scripted = torch.jit.freeze(torch.jit.script(mod))
            optimized = torch.jit.optimize_for_inference(scripted)
            inp = torch.rand([20, 20])
            # a1 can't be inplaced for first use, but can be for second use in "a1 * 4 + a1 * 5."
            FileCheck().check("ZENDNNScalarMul_").check("ZENDNNScalarMul(").check("ZENDNNScalarMul_").run(optimized.graph)
            self.assertEqual(optimized(inp), mod(inp))

if __name__ == '__main__':
    run_tests()
