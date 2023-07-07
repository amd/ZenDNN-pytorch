/******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/zendnn/Utils.h>

namespace at { namespace native {

#if AT_ZENDNN_ENABLED()

Tensor zendnn_to_dense(const Tensor& zendnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(zendnn_tensor.scalar_type() == ScalarType::Float || zendnn_tensor.scalar_type() == ScalarType::BFloat16,
              "zendnn_to_dense expects float or bfloat16 tensor input");

  adeep::tensor& stensor = itensor_from_zendnn(zendnn_tensor);
  auto dims = stensor.get_dims();
  auto data_type = dtype.has_value() ? dtype.value() : zendnn_tensor.scalar_type();
  TORCH_CHECK(data_type == ScalarType::Float || data_type == ScalarType::BFloat16 || data_type == ScalarType::Char,
            "zendnn tensor only can be converted to be  float or bfloat16 or char cpu tensor")
  // NOTE: int32_t dims from adeep::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::empty(
    std::vector<int64_t>(dims.begin(), dims.end()),
    zendnn_tensor.options().layout(c10::kStrided).dtype(data_type));
  if (stensor.is_empty()) return cpu_tensor;

  adeep::tensor pub_tensor;
  if(data_type == ScalarType::Float || data_type == ScalarType::BFloat16)
  {
      pub_tensor =
      data_type == ScalarType::Float
      ? stensor.to_public(cpu_tensor.template data_ptr<float>(),
                          adeep::tensor::data_type::f32)
      : stensor.to_public(cpu_tensor.template data_ptr<BFloat16>(),
                        adeep::tensor::data_type::bf16);
  }
  else if(data_type == ScalarType::Char)
  {
      pub_tensor = stensor.to_public(cpu_tensor.template data_ptr<int8_t>(),
                        adeep::tensor::data_type::s8);
  }

  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  return cpu_tensor;
}

Tensor dense_to_zendnn(const Tensor& cpu_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(),
             "dense_to_zendnn expects CPU tensor input");
  TORCH_CHECK(cpu_tensor.layout() == Layout::Strided,
             "dense_to_zendnn expects strided tensor input");
  TORCH_CHECK(cpu_tensor.scalar_type() == ScalarType::Float || cpu_tensor.scalar_type() == ScalarType::BFloat16,
             "dense_to_zendnn expects float or bfloat16 tensor input");
  TORCH_CHECK(cpu_tensor.dim() <= 5,
             "Can't convert cpu tensor with the number of dimensions > 5");
  // TODO: consider to convert non-contiguous tensor to `adeep::tensor` directly.
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  auto data_type = dtype.has_value() ? dtype.value() : cpu_tensor.scalar_type();
  TORCH_CHECK(data_type == ScalarType::Float || data_type == ScalarType::BFloat16  || data_type == ScalarType::Char,
              "cpu tensor only can be converted to be a float or bfloat16 or char zendnn tensor")
  Tensor zendnn_tensor = empty_zendnn(cpu_tensor_cont.sizes(), data_type,
                                      cpu_tensor_cont.options().layout_opt(), cpu_tensor_cont.options().device_opt(),
                                      cpu_tensor_cont.options().pinned_memory_opt());
  adeep::tensor& dtensor = itensor_from_zendnn(zendnn_tensor);
  if (cpu_tensor.scalar_type() == ScalarType::Float) {
    dtensor.feed_from(dtensor.get_dims(),
                      adeep::tensor::data_type::f32,
                      (cpu_tensor_cont.template data_ptr<float>()));
  }
  else if(cpu_tensor.scalar_type() == ScalarType::Char)
  {
    dtensor.feed_from(dtensor.get_dims(),
                    adeep::tensor::data_type::s8,
                    (cpu_tensor_cont.template data_ptr<int8_t>()));
  }
  else {
    dtensor.feed_from(dtensor.get_dims(),
                      adeep::tensor::data_type::bf16,
                      cpu_tensor_cont.template data_ptr<BFloat16>());
  }
  return zendnn_tensor;
}

std::vector<at::Tensor> zendnn_to_dense_list( TensorList zendnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(zendnn_tensor[0].scalar_type() == ScalarType::Float,
              "zendnn_to_dense_list expects float tensor input");
  auto data_type = dtype.has_value() ? dtype.value() : zendnn_tensor[0].scalar_type();

  TORCH_CHECK(data_type == ScalarType::Float,
            "zendnn tensor only can be converted to be a float cpu tensor")
  std::vector<at::Tensor> list_tensor;
  for (size_t i = 0; i < zendnn_tensor.size(); i++) {
     Tensor tensor = zendnn_to_dense(zendnn_tensor[i]);
     list_tensor.push_back(tensor);
  }
  return list_tensor;
}
// zendnn tensor has special non-public format for conv2d weights
// (dense_to_zendnn only converts dense tensor to zendnn tensor with
// public format). Ideep conv kernel will do implicit reorder if the
// weight is not already in this optimized format. By the time I'm
// writing this note, we are seeing ~20% perf cost of doing the
// on-the-fly reorder.
Tensor zendnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_reorder_conv2d_weight: bf16 path needs the cpu support avx512bf16");
  }

  auto w = itensor_from_zendnn(self);

  // Legacy zendnn conv2d jitted module may contain a 5-d weight with an extra
  // dimension when groups > 1, having dimension [g, o/g, i, h, w] instead of
  // [o, i, h, w]. Ideally we should reorder the weight back in serialization.
  // For backward compatibility, we squash the first two dims (g * o/g) back to
  // its original form.
  if (w.ndims() == 5) {
    auto wdims = w.get_dims();
    w.reshape({wdims[0] * wdims[1], wdims[2], wdims[3], wdims[4]});
  }

  auto desc =
      adeep::convolution_forward::expected_weights_desc(
          w.get_dims(),
          w.get_data_type(),
          {stride.begin(), stride.end()},
          {padding.begin(), padding.end()},
          {padding.begin(), padding.end()},
          {dilation.begin(), dilation.end()},
          groups,
          adeep::algorithm::convolution_direct);
  adeep::tensor result;
  result.init(desc);
  result.feed_from(w);

  return new_with_itensor_zendnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor zendnn_reorder_conv3d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(false,
        "zendnn_reorder_conv3d_weight: bf16 path is not supported in zendnn yet");
  }

  auto w = itensor_from_zendnn(self);

  auto desc =
      adeep::convolution_forward::expected_weights_desc(
          w.get_dims(),
          w.get_data_type(),
          {stride.begin(), stride.end()},
          {padding.begin(), padding.end()},
          {padding.begin(), padding.end()},
          {dilation.begin(), dilation.end()},
          groups,
          adeep::algorithm::convolution_direct);
  adeep::tensor result;
  result.init(desc);
  result.feed_from(w);

  return new_with_itensor_zendnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()), self.options().device_opt());
}


#else

Tensor zendnn_to_dense(const Tensor& zendnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "ZENDNN build is disabled");
}

std::vector<at::Tensor> zendnn_to_dense_list( TensorList zendnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "zendnn_to_dense_list: ZENDNN build is disabled");
}

Tensor dense_to_zendnn(const Tensor& cpu_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "ZENDNN build is disabled");
}
Tensor zendnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(false, "zendnn_reorder_conv2d_weight: ZENDNN build is disabled");
}

Tensor zendnn_reorder_conv3d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(false, "zendnn_reorder_conv3d_weight: ZENDNN build is disabled");
}
#endif // AT_ZENDNN_ENABLED()
}}
