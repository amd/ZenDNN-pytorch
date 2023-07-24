/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/Activation.h>

#if !AT_ZENDNN_ENABLED()

namespace at { namespace native {

Tensor zendnn_gelu(const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(false, "zendnn_gelu: ATen not compiled with ZENDNN support");
}

Tensor zendnn_gelu_backward(const Tensor& grad_output, const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(false, "zendnn_gelu_backward: ATen not compiled with ZENDNN support");
}

}}

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>

namespace at { namespace native {

Tensor zendnn_gelu(const Tensor& input, c10::string_view approximate) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_gelu: bf16 path needs the cpu support avx512bf16");
  }
  TORCH_CHECK(get_gelutype_enum(approximate) == GeluType::None,
                  "zendnn_gelu: fast, approximate gelu is not supported");
  const adeep::tensor& x = itensor_from_tensor(input);
  adeep::tensor y;
  adeep::eltwise_forward::compute(
      x, y, adeep::algorithm::eltwise_gelu_erf, adeep::prop_kind::forward_training, /*alpha*/ 0.0);
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

Tensor zendnn_gelu_backward(const Tensor& grad_output, const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(get_gelutype_enum(approximate) == GeluType::None,
                  "zendnn_gelu_backward: fast, approximate gelu is not supported");
  const adeep::tensor& x = itensor_from_tensor(input);
  adeep::tensor grady = itensor_from_tensor(grad_output);
  adeep::tensor gradx;
  adeep::eltwise_backward::compute(x, grady, gradx,
      adeep::algorithm::eltwise_gelu_erf, /*alpha*/ 0.0);
  return new_with_itensor_zendnn(std::move(gradx),
                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

}}

#endif // AT_ZENDNN_ENABLED
