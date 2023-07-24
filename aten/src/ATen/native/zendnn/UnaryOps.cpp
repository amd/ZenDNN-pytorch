/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor zendnn_sigmoid(const Tensor& self) {
  TORCH_CHECK(false, "zendnn_sigmoid: ATen not compiled with ZENDNN support");
}

Tensor& zendnn_sigmoid_(Tensor& self) {
  TORCH_CHECK(false, "zendnn_sigmoid_: ATen not compiled with ZENDNN support");
}

Tensor zendnn_tanh(const Tensor& self) {
  TORCH_CHECK(false, "zendnn_tanh: ATen not compiled with ZENDNN support");
}

Tensor& zendnn_tanh_(Tensor& self) {
  TORCH_CHECK(false, "zendnn_tanh_: ATen not compiled with ZENDNN support");
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at {
namespace native {

Tensor zendnn_sigmoid(const Tensor& self) {
  adeep::tensor& x = itensor_from_zendnn(self);
  adeep::tensor y;
  adeep::eltwise_forward::compute(
      x, y, adeep::algorithm::eltwise_logistic, adeep::prop_kind::forward);
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& zendnn_sigmoid_(Tensor& self) {
  adeep::tensor& x = itensor_from_zendnn(self);
  adeep::eltwise_forward::compute(
      x, x, adeep::algorithm::eltwise_logistic, adeep::prop_kind::forward);
  return self;
}

Tensor zendnn_tanh(const Tensor& self) {
  adeep::tensor& x = itensor_from_zendnn(self);
  adeep::tensor y;
  adeep::eltwise_forward::compute(
      x, y, adeep::algorithm::eltwise_tanh, adeep::prop_kind::forward);
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& zendnn_tanh_(Tensor& self) {
  adeep::tensor& x = itensor_from_zendnn(self);
  adeep::eltwise_forward::compute(
      x, x, adeep::algorithm::eltwise_tanh, adeep::prop_kind::forward);
  return self;
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED
