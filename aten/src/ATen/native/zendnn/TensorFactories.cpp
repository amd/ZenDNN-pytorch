/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at { namespace native {

#if AT_ZENDNN_ENABLED()

Tensor empty_zendnn(IntArrayRef sizes, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
     !optional_memory_format.has_value(),
     "'memory_format' argument is incompatible with zendnn tensor");
  // NOTE: int32_t dims from adeep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in adeep::tensor to avoid extra conversion
  adeep::tensor::dims dst_dims (sizes.begin(), sizes.end());
  auto data_type = dtype.has_value() ? get_zendnn_dtype(dtype.value()) : adeep::tensor::data_type::f32;
  adeep::tensor it {dst_dims, data_type};
  return new_with_itensor_zendnn(std::move(it), dtype, device);
}

#else

Tensor empty_zendnn(IntArrayRef sizes, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(false, "empty_zendnn: ZENDNN build is disabled");
}

#endif // AT_ZENDNN_ENABLED()

}}
