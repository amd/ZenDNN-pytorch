/******************************************************************************
* Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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


#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor  zendnn_concat(const ITensorListRef&  tensors, int64_t dim) {
  TORCH_CHECK(false, "zendnn_concat: ATen not compiled with ZENDNN support");
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at {
namespace native {

 Tensor zendnn_concat(const ITensorListRef&  tensors, int64_t dim) {
  auto materialized_tensors = tensors.materialize();
  for (size_t i = 0; i < tensors.size(); i++) {
    if(!materialized_tensors[i].get().is_zendnn())
    {
      TORCH_CHECK(false, "zendnn_concat expects all the input tensors should be of type zendnn");
    }
  }
  std::vector<adeep::tensor> inputs;
  for (size_t i = 0; i < materialized_tensors.size(); i++) {
     adeep::tensor& x = itensor_from_zendnn(materialized_tensors[i].get());
     inputs.push_back(x);
  }
  adeep::tensor y;
  adeep::concat::compute(inputs, dim, y);
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(materialized_tensors[0].get().options().dtype_opt()),
                                materialized_tensors[0].get().options().device_opt());
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED
