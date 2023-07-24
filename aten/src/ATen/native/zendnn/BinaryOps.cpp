/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor& zendnn_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result
    ) {
  TORCH_CHECK(false, "zendnn_add_out: ATen not compiled with ZENDNN support");
}

Tensor zendnn_add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(false, "zendnn_add: ATen not compiled with ZENDNN support");
}

Tensor& zendnn_add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(false, "zendnn_add_: ATen not compiled with ZENDNN support");
}

Tensor& zendnn_mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(false, "zendnn_mul_out: ATen not compiled with ZENDNN support");
}

Tensor zendnn_mul(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "zendnn_mul: ATen not compiled with ZENDNN support");
}

Tensor& zendnn_mul_(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "zendnn_mul_: ATen not compiled with ZENDNN support");
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at {
namespace native {

Tensor emptyBinaryOp(const Tensor& self, const Tensor& other) {
  if (!self.requires_grad() && !other.requires_grad()) {
    auto out_size = infer_size(self.sizes(), other.sizes());
    auto out_dtype = promoteTypes(
        c10::typeMetaToScalarType(self.dtype()),
        c10::typeMetaToScalarType(other.dtype()));
    TORCH_CHECK(
        self.device() == other.device(),
        "Expected same device for binary zendnn op");
    return empty_zendnn(
        out_size,
        out_dtype,
        self.options().layout_opt(),
        self.options().device_opt(),
        self.options().pinned_memory_opt());
  } else {
    TORCH_CHECK(
        false,
        "ZENDNN does not support Binary Ops with a 0-dimension Tensor in training");
  }
}

Tensor& zendnn_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result
    ) {
  adeep::tensor& x = itensor_from_zendnn(self);
  adeep::tensor& y = itensor_from_zendnn(other);

  adeep::tensor& z = itensor_from_zendnn(result);
  if (result.is_same(other)) {
    const std::vector<float> scales{alpha.to<float>(), 1.0};
    adeep::sum::compute(scales, {y, x}, z);
  } else {
    const std::vector<float> scales{1.0, alpha.to<float>()};
    adeep::sum::compute(scales, {x, y}, z);
  }

  return result;
}

Tensor zendnn_add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  if (self.numel() == 0 || other.numel() == 0) {
    return emptyBinaryOp(self, other);
  }

  adeep::tensor& x = itensor_from_zendnn(self);
  adeep::tensor& y = itensor_from_zendnn(other);

  adeep::tensor z;
  const std::vector<float> scales{1.0, alpha.to<float>()};
  adeep::sum::compute(scales, {x, y}, z);

  return new_with_itensor_zendnn(std::move(z), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& zendnn_add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return native::zendnn_add_out(self, other, alpha, self);
}

Tensor& zendnn_mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(result.sizes() == self.sizes(),
             "zendnn_mul_out: the output size should be same as input size");
  adeep::tensor& z = itensor_from_zendnn(result);
  adeep::tensor& x = itensor_from_zendnn(self);

  // for zero_dim tensor
  if (other.ndimension() == 0) {
    adeep::eltwise_forward::compute(
      x, z, adeep::algorithm::eltwise_linear,
      adeep::prop_kind::forward_inference, /*alpha*/ other.item().to<float>());

    return result;
  } else {
    TORCH_CHECK(self.sizes() == other.sizes(),
               "zendnn_mul_out: currently zendnn not support broadcasting");
    adeep::tensor y = itensor_from_zendnn(other);
    adeep::binary::compute(x, y, z, zendnn::algorithm::binary_mul);

    return result;
  }
}

Tensor zendnn_mul(const Tensor& self, const Tensor& other) {
  if (self.numel() == 0 || other.numel() == 0) {
    return emptyBinaryOp(self, other);
  }
  Tensor result = empty_zendnn(self.sizes(), optTypeMetaToScalarType(self.options().dtype_opt()),
                               self.options().layout_opt(), self.options().device_opt(),
                               self.options().pinned_memory_opt());
  return native::zendnn_mul_out(self, other, result);
}

Tensor& zendnn_mul_(Tensor& self, const Tensor& other) {
  return native::zendnn_mul_out(self, other, self);
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_EBABLED
