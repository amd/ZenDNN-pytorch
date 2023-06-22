/******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

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
