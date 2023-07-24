/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor zendnn_view(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false, "zendnn_view: ATen not compiled with ZENDNN support");
}

Tensor zendnn_reshape(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false, "zendnn_reshape: ATen not compiled with ZENDNN support");
}

Tensor zendnn_clone(const Tensor& self, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(false, "zendnn_clone: ATen not compiled with ZENDNN support");
}

Tensor zendnn_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "zendnn_transpose: ATen not compiled with ZENDNN support");
}

Tensor& zendnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "zendnn_transpose_: ATen not compiled with ZENDNN support");
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at {
namespace native {

Tensor zendnn_view(const Tensor& self, IntArrayRef size) {
  // TORCH_CHECK(false, "Currently Zendnn tensor does not support view. Change to use reshape instead");
  if(self.is_zendnn()) {
          Tensor self_dense = zendnn_to_dense(self);
          Tensor res = view(self_dense, size);
          Tensor res_zendnn = dense_to_zendnn(res);
          return res_zendnn;}
  else
          return view(self, size);
}

Tensor zendnn_reshape(const Tensor& self, IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  if (self.sizes() == inferred_size) {
    return self;
  }
  const adeep::tensor& x = itensor_from_zendnn(self);
  adeep::tensor y{x};
  y.reshape(inferred_size);
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor zendnn_clone(const Tensor& self, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  adeep::tensor& src = itensor_from_zendnn(self);
  adeep::tensor dst;
  adeep::direct_copy::compute(src, dst);
  return new_with_itensor_zendnn(std::move(dst), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor zendnn_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  const adeep::tensor& x = itensor_from_zendnn(self);
  adeep::tensor y;
  std::vector<int> axes(x.ndims());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[dim0], axes[dim1]);
  y.transpose_from(x, axes);
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& zendnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "zendnn_transpose_: in-place zendnn operations are not supported yet");
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_EBABLED
