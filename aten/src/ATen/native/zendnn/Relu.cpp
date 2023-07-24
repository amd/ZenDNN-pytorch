/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>


#if !AT_ZENDNN_ENABLED()

namespace at { namespace native {

Tensor zendnn_relu(const Tensor& input) {
  TORCH_CHECK(false, "zendnn_relu: ATen not compiled with ZENDNN support");
}

Tensor& zendnn_relu_(Tensor& input) {
  TORCH_CHECK(false, "zendnn_relu_: ATen not compiled with ZENDNN support");
}
}}

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>

namespace at { namespace native {

Tensor zendnn_relu(const Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_relu: bf16 path needs the cpu support avx512bf16");
  }

  const adeep::tensor& x = itensor_from_zendnn(input);
  adeep::tensor y;
  adeep::eltwise_forward::compute(
      x, y, adeep::algorithm::eltwise_relu, adeep::prop_kind::forward_training, /*alpha*/ 0.0);
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

Tensor& zendnn_relu_(Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_relu_: bf16 path needs the cpu support avx512bf16");
  }

  adeep::tensor& x = itensor_from_zendnn(input);
  adeep::eltwise_forward::compute(
      x, x, adeep::algorithm::eltwise_relu, adeep::prop_kind::forward_training, /*alpha*/ 0.0);
  return input;
}
}}

#endif // AT_ZENDNN_ENABLED
