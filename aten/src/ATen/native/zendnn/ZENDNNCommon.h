/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_ZENDNN_ENABLED()
#include <adeep.hpp>

namespace at { namespace native {

// Mapping ScalarType to adeep tensor data_type
adeep::tensor::data_type get_zendnn_dtype(ScalarType type);

// Construct aten ZENDNN tensor given an adeep tensor
Tensor new_with_itensor_zendnn(adeep::tensor&& it, c10::optional<ScalarType> dtype, c10::optional<Device> device);

// Retrieve `adeep::tensor` from ZENDNN tensor
adeep::tensor& itensor_from_zendnn(const Tensor& zendnn_tensor);

// Construct an `adeep::tensor` "view" from dense tensor, note the
// adeep::tensor will share the underlying buffer
adeep::tensor itensor_view_from_dense(const Tensor& tensor);

// Helper function for getting an adeep tensor out of an aten Tensor or ZENDNN tensor.
adeep::tensor itensor_from_tensor(const Tensor& tensor);

// helper functions for pytorch to zendnn tensor conversion without wrapping
// into abstract tensor
adeep::tensor zendnn_tensor_view_from_dense(const Tensor& ttensor);
Tensor new_dense_from_zendnn(const adeep::tensor& zendnn_tensor,
                             const TensorOptions& options);

}}

#endif // AT_ZENDNN_ENABLED
