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
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor zendnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    c10::optional<bool> fuse_gelu) {
  TORCH_CHECK(false, "zendnn_linear: ATen not compiled with ZENDNN support");
}

Tensor zendnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(false, "zendnn_linear_backward_input: ATen not compiled with ZENDNN support");
}

std::tuple<Tensor, Tensor> zendnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  TORCH_CHECK(false, "zendnn_linear_backward_weights: ATen not compiled with ZENDNN support");
}

std::tuple<Tensor, Tensor, Tensor> zendnn_linear_backward(
    const Tensor& input, const Tensor& grad_output_t,
    const Tensor& weight, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "zendnn_linear_backward: ATen not compiled with ZENDNN support");
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>

namespace at {
namespace native {

Tensor zendnn_linear(
    const Tensor& self,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    c10::optional<bool> fuse_gelu) {
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_linear: bf16 path needs the cpu support avx512bf16");
  }

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  const int64_t dim = self.dim();
  TORCH_CHECK(
      self.dim() != 0,
      "zendnn_linear: input needs to has dim at least 1, input dim ",
      self.dim());
  TORCH_CHECK(self.is_zendnn(),
      "zendnn_linear: input needs to be zendnn layout");

  // reshape first if input dim != 2 and the reshape will cost a memory copy.
  auto self_reshaped =
      dim == 2 ? self : self.reshape({-1, self.size(self.dim() - 1)});

  const adeep::tensor x = itensor_from_zendnn(self_reshaped);
  // weight can be a zendnn tensor or dense tensor.
  const Tensor weight = (weight_t.is_zendnn() || weight_t.is_contiguous()) ? weight_t : weight_t.contiguous();
  const adeep::tensor w = itensor_from_tensor(weight);

  adeep::tensor y;
  adeep::scale_t src_scales = adeep::scale_t();
  adeep::scale_t weights_scales = adeep::scale_t();
  adeep::scale_t dst_scales = adeep::scale_t();
  adeep::attr_t attr;

  if(fuse_gelu.has_value())
  {
     attr = *fuse_gelu ? adeep::attr_t::fuse_gelu(): attr;
  }

  if (bias.defined()) {
    const adeep::tensor b = itensor_from_tensor(bias);
    adeep::inner_product_forward::compute(x, w, b, y,
        src_scales,
        weights_scales,
        dst_scales,
        attr);
  } else {
    adeep::inner_product_forward::compute(x, w, y,
        src_scales,
        weights_scales,
        dst_scales,
        attr);
  }

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  if (self.dim() != 2) {
    return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                   self.options().device_opt()).reshape(output_size);
  }
  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor zendnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight_t){
  TORCH_CHECK(grad_output.is_zendnn(),
      "zendnn_linear_backward: grad_output needs to be zendnn layout");
  TORCH_CHECK(weight_t.device().is_cpu() && weight_t.scalar_type() == kFloat,
      "zendnn_linear_backward: weight_t needs to be a dense tensor");
  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;

  adeep::tensor& grady = itensor_from_zendnn(grad_output_reshaped);
  // weight_t always dense tensor for training.
  const Tensor weight = weight_t.is_contiguous() ? weight_t : weight_t.contiguous();
  const adeep::tensor w = itensor_view_from_dense(weight);

  std::vector<int64_t> input_reshaped_size;
  input_reshaped_size.push_back(grad_output_reshaped.size(0));
  input_reshaped_size.push_back(weight.size(1));

  adeep::tensor gradx;
  adeep::inner_product_backward_data::compute(
    grady, w, {input_reshaped_size.begin(), input_reshaped_size.end()}, gradx);
  if (input_size.size() > 2) {
    return new_with_itensor_zendnn(std::move(gradx), optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                   grad_output.options().device_opt()).reshape(input_size);
  }
  return new_with_itensor_zendnn(std::move(gradx), optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

std::tuple<Tensor, Tensor> zendnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  TORCH_CHECK(grad_output.is_zendnn() && input.is_zendnn(),
      "zendnn_linear_backward: grad_output and input needs to be zendnn layout");
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == kFloat,
      "zendnn_linear_backward: weight needs to be a dense tensor");

  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() > 2 ? input.reshape({-1, input.size(input.dim() - 1)}) : input;

  adeep::tensor& grady = itensor_from_zendnn(grad_output_reshaped);
  adeep::tensor& x = itensor_from_zendnn(input_reshaped);
  adeep::tensor gradw, gradb;
  if (bias_defined) {
    adeep::inner_product_backward_weights::compute(x, grady, gradw, gradb);
  } else {
    adeep::inner_product_backward_weights::compute(x, grady, gradw);
  }

  return std::tuple<Tensor, Tensor>{
    zendnn_to_dense(new_with_itensor_zendnn(std::move(gradw),
                    optTypeMetaToScalarType(weight.options().dtype_opt()),
                    weight.options().device_opt())),
    zendnn_to_dense(new_with_itensor_zendnn(std::move(gradb),
                    optTypeMetaToScalarType(weight.options().dtype_opt()),
                    weight.options().device_opt()))};
}

std::tuple<Tensor, Tensor, Tensor> zendnn_linear_backward(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, std::array<bool,3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::zendnn_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::zendnn_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED

#if !AT_ZENDNN_QUANT_ENABLED()

namespace at { namespace native {

Tensor zendnn_vitisai_linear(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    bool dequant,
    bool fuse_relu,
    int64_t input_scale,
    int64_t filter_scale,
    int64_t output_scale) {
    TORCH_CHECK(false, "zendnn_vitisai_linear: ATen not compiled with ZENDNN QUANTIZATION support");
}

Tensor zendnn_vitisai_matmul(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& add_input,
    bool dequant,
    bool fuse_div,
    int64_t gelu_scale,
    int64_t input_scale,
    int64_t filter_scale,
    int64_t output_scale,
    int64_t add_scale) {
    TORCH_CHECK(false, "zendnn_vitisai_matmul: ATen not compiled with ZENDNN QUANTIZATION support");

    }
}}

#else // AT_ZENDNN_QUANT_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/zendnn/ZENDNNTensors.h>
namespace at { namespace native {
using Convt = at::native::ZendnnTensorConvert;


//zendnn_vitisai_matmul Handles 3D and 4D matrix multiplications
//It returns aten tensor as an output as no blocked format
//conversion is involved and can avoid few output reorders
//in few cases.
Tensor zendnn_vitisai_matmul(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& add_input,
    bool dequant,
    bool fuse_div,
    int64_t gelu_scale,
    int64_t input_scale,
    int64_t filter_scale,
    int64_t output_scale,
    int64_t add_scale) {

  // See [Note: hacky wrapper removal for optional tensor]
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(false, "zendnn_vitisai_matmul: bf16 path is not supported for zendnn_vitisai_matmul");
  }

  if(input.dim() !=2 && input.dim() !=3 && input.dim() !=4)
  {
    TORCH_CHECK(false, "zendnn_vitisai_matmul: Currently support only for 2D 3D and 4D tensors");
  }
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  c10::MaybeOwned<Tensor> add_input_maybe_owned = at::borrow_from_optional_tensor(add_input);

  const Tensor& add_input_tensor = *add_input_maybe_owned;
  const bool bias_enable = bias.defined();

  bool fuse_add = add_input_tensor.defined();
  Tensor add_input_to_zendnn = add_input_tensor;

  float input_scale_pow_2 = std::pow(2, input_scale);
  float filter_scale_pow_2 = std::pow(2, filter_scale);
  float bias_scale_derived_pow_2 = std::pow(2, input_scale + filter_scale);
  float requantize_scale_pow_2 = std::pow(2, -input_scale - filter_scale + output_scale);
  float add_scale_pow_2 = 1.0f;
  float gelu_scale_pow_2 = -1.0f;

  Tensor input1 = input.contiguous();
  const adeep::tensor zendnn_input = itensor_from_tensor(input1);

  //FIXME: Since the divisor is 8 always in the model, merging it with
  //requant scale computation.

  //Modified the pattern in the JIT such that matmul ad fusion will happen
  //Original Pattern: Linear->reshape->permute->matmul->div->add
  //Modified Pattern: Linear->div->reshape->permute->matmul->add
  int div_scale = fuse_div ? 3:0;

  adeep::tensor zendnn_bias;
  if (bias.defined()) {
    zendnn_bias = itensor_from_tensor(bias);
  }

  ScalarType dtype = dequant == true? ScalarType::Float : ScalarType::Char;

  //Output tensor(aten) initialization with int8, and based on dequant flag
  //tensor dtype can be typecasted
  Tensor  cpu_tensor;
  if (input.dim() == 2)
  {
    cpu_tensor = at::empty(
          {input.sizes()[0], weight.sizes()[1]}, dtype
          );
  }
  else if(input.dim() == 3)
  {
    cpu_tensor = at::empty(
          {input.sizes()[0], input.sizes()[1], weight.sizes()[2]}, dtype
          );
  }
  else if(input.dim() == 4)
  {
    cpu_tensor = at::empty(
          {input.sizes()[0], input.sizes()[1] ,input.sizes()[2] , weight.sizes()[3]},
          dtype);
  }
  else
  {
    TORCH_CHECK(false, "zendnn_vitisai_matmul: Currently support only for 2D 3D and 4D tensors");
  }

  adeep::tensor zendnn_output;

  //Handling scales based on dequant flag
  if(dequant){
    requantize_scale_pow_2 = std::pow(2, -input_scale - filter_scale - div_scale);
  }
  else
  {
    requantize_scale_pow_2 = std::pow(2, -input_scale - filter_scale + output_scale - div_scale);
  }

  //Handling gelu scales, if gelu node is not present, by default it
  //is set to -1.
  if(gelu_scale != -1){
    requantize_scale_pow_2 = std::pow(2, -input_scale - filter_scale);
    gelu_scale_pow_2= 1.0f;
  }

  if(gelu_scale != -1 && !dequant){
      gelu_scale_pow_2 = std::pow(2, gelu_scale);
  }

  if(fuse_add){
      requantize_scale_pow_2 = std::pow(2, -input_scale - filter_scale);
      add_scale_pow_2 = std::pow(2, -add_scale);
      //Broadcasting the tensor to the relevant dimentions
      if(add_input_tensor.sizes() != cpu_tensor.sizes())
      {
       add_input_to_zendnn = add_input_tensor.expand(cpu_tensor.sizes()).contiguous();
      }
      zendnn_output =  Convt::zentensor_view_dense(add_input_to_zendnn);
  }
  else
  {
    zendnn_output =   Convt::zentensor_view_dense(cpu_tensor);
  }

  //Weight caching will only be on done on ZenDNN weight tesnors
  //aten weight is converted to zendnn weight tensor in JIT pass.
  if(weight.is_zendnn())
  {
    adeep::tensor& zendnn_weight = itensor_from_zendnn(weight);
    adeep::matmul_forward::zen_vitis_matmul(
        zendnn_input,
        zendnn_weight,
        zendnn_bias,
        bias_enable,
        zendnn_output,
        dequant,
        fuse_add,
        gelu_scale_pow_2,
        input_scale_pow_2,
        filter_scale_pow_2,
        bias_scale_derived_pow_2,
        requantize_scale_pow_2,
        add_scale_pow_2
      );
  }
  else
  {
    Tensor input2 = weight.contiguous();
    adeep::tensor zendnn_weight = itensor_from_tensor(input2);
    adeep::matmul_forward::zen_vitis_matmul(
        zendnn_input,
        zendnn_weight,
        zendnn_bias,
        bias_enable,
        zendnn_output,
        dequant,
        fuse_add,
        gelu_scale_pow_2,
        input_scale_pow_2,
        filter_scale_pow_2,
        bias_scale_derived_pow_2,
        requantize_scale_pow_2,
        add_scale_pow_2
      );
  }

  if(!fuse_add)
  {
    return cpu_tensor;
  }
  else
  {
    return add_input_to_zendnn;
  }

}


Tensor zendnn_vitisai_linear(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    bool dequant,
    bool fuse_relu,
    int64_t input_scale,
    int64_t filter_scale,
    int64_t output_scale) {

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(false, "zendnn_vitisai_linear: bf16 path is not supported for zendnn_vitisai_linear");
  }

  const adeep::tensor zendnn_input = itensor_from_tensor(input);
  const adeep::tensor zendnn_weight = itensor_from_tensor(weight);
  c10::optional<adeep::tensor> zendnn_bias{c10::nullopt};
  if (bias.defined()) {
    zendnn_bias = itensor_from_tensor(bias);
  }
  adeep::tensor zendnn_output;

  float input_scale_pow_2 = std::pow(2, input_scale);
  float filter_scale_pow_2 = std::pow(2, filter_scale);
  float bias_scale_derived_pow_2 = std::pow(2, input_scale + filter_scale);
  float requantize_scale_pow_2 = std::pow(2, -input_scale - filter_scale + output_scale);
  float dequantize_scale_pow_2 = std::pow(2, -output_scale);

  if(dequant){
    requantize_scale_pow_2 = std::pow(2, -input_scale - filter_scale);
  }

  adeep::inner_product_forward::zen_vitis_compute(
      zendnn_input,
      zendnn_weight,
      zendnn_bias.value(),
      zendnn_output,
      dequant,
      fuse_relu,
      input_scale_pow_2,
      filter_scale_pow_2,
      bias_scale_derived_pow_2,
      requantize_scale_pow_2,
      dequantize_scale_pow_2);

   if (!dequant) {
    return new_with_itensor_zendnn(std::move(zendnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                   input.options().device_opt());
  } else {
    return zendnn_to_dense(
        new_with_itensor_zendnn(std::move(zendnn_output), ScalarType::Float,
                                input.options().device_opt()));
  }
}

}}

#endif