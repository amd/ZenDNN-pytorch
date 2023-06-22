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

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/Activation.h>

#if !AT_ZENDNN_ENABLED()

namespace at { namespace native {

Tensor zendnn_gelu(const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(false, "zendnn_gelu: ATen not compiled with ZENDNN support");
}

Tensor zendnn_gelu_backward(const Tensor& grad_output, const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(false, "zendnn_gelu_backward: ATen not compiled with ZENDNN support");
}

}}

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>

namespace at { namespace native {

Tensor zendnn_gelu(const Tensor& input, c10::string_view approximate) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_gelu: bf16 path needs the cpu support avx512bf16");
  }
  TORCH_CHECK(get_gelutype_enum(approximate) == GeluType::None,
                  "zendnn_gelu: fast, approximate gelu is not supported");
  const adeep::tensor& x = itensor_from_tensor(input);
  adeep::tensor y;
  adeep::eltwise_forward::compute(
      x, y, adeep::algorithm::eltwise_gelu_erf, adeep::prop_kind::forward_training, /*alpha*/ 0.0);
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

Tensor zendnn_gelu_backward(const Tensor& grad_output, const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(get_gelutype_enum(approximate) == GeluType::None,
                  "zendnn_gelu_backward: fast, approximate gelu is not supported");
  const adeep::tensor& x = itensor_from_tensor(input);
  adeep::tensor grady = itensor_from_tensor(grad_output);
  adeep::tensor gradx;
  adeep::eltwise_backward::compute(x, grady, gradx,
      adeep::algorithm::eltwise_gelu_erf, /*alpha*/ 0.0);
  return new_with_itensor_zendnn(std::move(gradx),
                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

}}

#endif // AT_ZENDNN_ENABLED
