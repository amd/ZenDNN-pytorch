/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/


#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor  zendnn_concat(TensorList tensors, int64_t dim) {
  TORCH_CHECK(false, "zendnn_concat: ATen not compiled with ZENDNN support");
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at {
namespace native {

 Tensor zendnn_concat(TensorList tensors, int64_t dim) {
  for (size_t i = 0; i < tensors.size(); i++) {
    if(!tensors[i].is_zendnn())
    {
      TORCH_CHECK(false, "zendnn_concat expects all the input tensors should be of type zendnn");
    }
  }
  std::vector<adeep::tensor> inputs;
  for (size_t i = 0; i < tensors.size(); i++) {
     adeep::tensor& x = itensor_from_zendnn(tensors[i]);
     inputs.push_back(x);
  }
  adeep::tensor y;
  adeep::concat::compute(inputs, dim, y);
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(tensors[0].options().dtype_opt()),
                                tensors[0].options().device_opt());
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED
