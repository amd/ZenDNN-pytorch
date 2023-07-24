#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************

import itertools
import functools
import unittest

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

import torch
import torch.nn.functional as F
import torch.jit
import torch.backends.zendnn
from torch.testing._internal.common_utils import TestCase, \
    run_tests, gradcheck, gradgradcheck

# batched grad doesn't support zendnn
gradcheck = functools.partial(gradcheck, check_batched_grad=False)
gradgradcheck = functools.partial(gradgradcheck, check_batched_grad=False)

types = [torch.float]

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
                    self.assertEqual(cpu_tensor, cpu_tensor_2.float(), atol=atol, rtol=0)

                self.assertEqual(zendnn_tensor.device, torch.device('cpu'))
                self.assertEqual(zendnn_tensor.size(), torch.Size([1, 2, 3, 4]))
                self.assertEqual(zendnn_tensor.numel(), cpu_tensor.numel())
                if dtype1 == torch.float:
                    self.assertEqual(zendnn_tensor.element_size(), cpu_tensor.element_size())
                else:
                    self.assertEqual(zendnn_tensor.element_size(), cpu_tensor.element_size() / 2)
                self.assertRaisesRegex(RuntimeError,
                                       "Cannot access data pointer of Tensor that doesn't have storage",
                                       lambda: zendnn_tensor.data_ptr() != 0)

    def test_unsupported(self):
        # unsupported types
        for dtype in [torch.double, torch.half, torch.uint8, torch.int8,
                      torch.short, torch.int, torch.long]:
            with self.assertRaises(RuntimeError) as context:
                torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cpu')).to_zendnn()
        # some factory functions
        for creator in [torch.ones, torch.randn, torch.rand]:
            with self.assertRaises(RuntimeError) as context:
                creator(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu'), layout=torch._zendnn)

    def test_detach(self):
        root = torch.randn(4, 5, dtype=torch.float32).to_zendnn().requires_grad_()
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
        self.assertRaises(RuntimeError, lambda: x_zendnn.new(torch.Size([2, 3])))
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
        input=torch.randint(1, 15, (W,))
        offsets = torch.tensor([0,W], dtype=torch.long)
        emb_d = torch.nn.EmbeddingBag(  num_embeddings=R,
                                        embedding_dim=W,
                                        scale_grad_by_freq=False,
                                        mode="sum",
                                        sparse = True,
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

    @torch.no_grad()
    def test_linear(self):
        options = [True, False]
        for bias in options:
            N = torch.randint(3, 64, (1,)).item()
            in_feats = torch.randint(1, 64, (1,)).item()
            out_feats = torch.randint(1, 64, (1,)).item()
            x = torch.randn((N, in_feats), dtype=torch.float32)

            linear = torch.nn.Linear(in_features=in_feats, out_features=out_feats, bias=bias)

            self.assertEqual(linear(x), linear(x.to_zendnn()).to_dense())
            self._test_tracing(linear, (x,))
            self._test_scripting(linear, (x,))

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
        model = torchvision.models.resnet.resnet18(pretrained=False)
        self._test_imagenet_model(model)

    @skipIfNoTorchVision
    def test_resnext50_32x4d(self):
        model = torchvision.models.resnet.resnext50_32x4d(pretrained=False)
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

    def test_adaptive_avg_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

        self.assertEqual(
            adaptive_avg_pool2d(x),
            adaptive_avg_pool2d(x.to_zendnn()).to_dense())

    def test_relu(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        self.assertEqual(torch.relu(x), torch.relu(x.to_zendnn()).to_dense())

    def test_relu_(self):
        x1 = torch.randn((4, 5), dtype=torch.float32) * 10
        x2 = x1.clone().to_zendnn()
        self.assertEqual(torch.relu_(x1), torch.relu_(x2).to_dense())

    def test_concat(self):
        for dim in range(0,4):
            N,C = torch.randint(3, 10, (1,)),torch.randint(3, 10, (1,))
            for H,W in zip(torch.randint(24, 256, (5,)),torch.randint(24, 256, (5,))):
                offset=torch.randint(1, 10, (1,))
                if dim==0:
                    x,y = torch.randn(N, C, H, W, dtype=torch.float32) * 10,torch.randn(N+offset, C, H, W, dtype=torch.float32) * 10
                elif dim==1:
                    x,y = torch.randn(N, C+offset, H, W, dtype=torch.float32) * 10,torch.randn(N, C, H, W, dtype=torch.float32) * 10
                elif dim==2:
                    x,y = torch.randn(N, C, H, W, dtype=torch.float32) * 10,torch.randn(N, C, H+offset, W, dtype=torch.float32) * 10
                else:
                    x,y = torch.randn(N, C, H, W+offset, dtype=torch.float32) * 10,torch.randn(N, C, H, W, dtype=torch.float32) * 10
                self.assertEqual(torch.cat([x,y],dim),torch.cat([x.to_zendnn(),y.to_zendnn()],dim).to_dense())

        #Check Failing Case
        x,y = torch.randn(N, C, H, W, dtype=torch.float32) * 10,torch.randn(N+offset, C, H, W, dtype=torch.float32) * 10
        with self.assertRaises(RuntimeError) as context:
            torch.cat([x.to_zendnn(),y],0).to_dense()
        self.assertTrue('zendnn_concat expects all the input tensors should be of type zendnn' == str(context.exception))

    def _test_batch_norm_base(self, dim, channels, input):
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}
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

if __name__ == '__main__':
    run_tests()
