/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor& copy_zendnn_(Tensor& self, const Tensor& src, bool non_blocking) {
  TORCH_CHECK(false, "copy_zendnn_: ATen not compiled with ZENDNN support");
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_EBABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at {
namespace native {

Tensor& copy_zendnn_(Tensor& self, const Tensor& src, bool non_blocking) {
  TORCH_CHECK(
      self.sizes() == src.sizes(),
      "copy_zendnn_: only support same size tensor.");
  TORCH_CHECK(
      self.is_zendnn() && src.is_zendnn(),
      "copy_zendnn_: between zendnn layout and dense Tensors is not implemented! Found self type = ",
      self.toString(),
      " and src type = ",
      src.toString());
  adeep::tensor& x = itensor_from_zendnn(src);
  adeep::tensor& y = itensor_from_zendnn(self);
  adeep::direct_copy::compute(x, y);
  return self;
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_EBABLED
