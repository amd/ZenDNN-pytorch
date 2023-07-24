/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor zendnn_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  TORCH_CHECK(false, "zendnn_softmax: ATen not compiled with ZENDNN support");
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at {
namespace native {

Tensor zendnn_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on Mkldnn");
  const int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim());
  adeep::tensor& x = itensor_from_zendnn(self);
  adeep::tensor y;
  adeep::softmax_forward::compute(x, y, wrapped_dim);
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED
