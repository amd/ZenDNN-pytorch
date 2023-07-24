/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/zendnn/Utils.h>
#if !AT_ZENDNN_ENABLED()

namespace at { namespace native {

Tensor zendnn_convolution_relu(
    const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(false, "zendnn_convolution_relu: ATen not compiled with ZENDNN support");
}

}}

#else // AT_ZENDNN_EBABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>
#include <ATen/native/ConvUtils.h>

namespace at { namespace native {

adeep::tensor _zendnn_convolution_relu(
    const adeep::tensor& x,
    const adeep::tensor& w,
    const c10::optional<adeep::tensor>& b,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {

  auto kernel_size = w.get_dims();

  std::vector<int64_t> input_size = x.get_dims();
  std::vector<int64_t> output_sizes =
      conv_output_size(input_size, kernel_size, padding, stride, dilation);

  adeep::tensor y;
  adeep::scale_t src_scales = adeep::scale_t();
  adeep::scale_t weights_scales = adeep::scale_t();
  adeep::scale_t dst_scales = adeep::scale_t();
  adeep::attr_t attr = adeep::attr_t::fuse_relu();
  if (b.has_value()) {
    adeep::convolution_forward::compute(
        x,
        w,
        b.value(),
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        adeep::algorithm::convolution_direct,
        adeep::prop_kind::forward_inference);
  } else {
    adeep::convolution_forward::compute(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        adeep::algorithm::convolution_direct,
        adeep::prop_kind::forward_inference);
  }
  return y;
}

Tensor zendnn_convolution_relu(
    const Tensor& input,
    const Tensor& weight, const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_convolution_relu: bf16 path needs the cpu support avx512bf16");
  }

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  const adeep::tensor zendnn_input = itensor_from_tensor(input);
  const adeep::tensor zendnn_weight = itensor_from_tensor(weight);
  c10::optional<adeep::tensor> zendnn_bias{c10::nullopt};
  if (bias.defined()) {
    zendnn_bias = itensor_from_tensor(bias);
  }

  adeep::tensor zendnn_output = _zendnn_convolution_relu(
      zendnn_input,
      zendnn_weight,
      zendnn_bias,
      padding,
      stride,
      dilation,
      groups);

  if (input.is_zendnn()) {
    return new_with_itensor_zendnn(std::move(zendnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                   input.options().device_opt());
  } else {
    return zendnn_to_dense(
        new_with_itensor_zendnn(std::move(zendnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
  }
}

}}  // namespace at::native

#endif

#if !AT_ZENDNN_QUANT_ENABLED()

namespace at { namespace native {

Tensor zendnn_vitisai_convolution(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& add_input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool dequant,
    bool fuse_relu,
    int64_t input_scale,
    int64_t filter_scale,
    int64_t output_scale,
    int64_t add_scale,
    int64_t add_out_scale) {
    TORCH_CHECK(false, "zendnn_vitisai_convolution: ATen not compiled with ZENDNN QUANTIZATION support");
}

}}

#else // AT_ZENDNN_QUANT_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>
#include <ATen/native/ConvUtils.h>

namespace at { namespace native {

Tensor zendnn_vitisai_convolution(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& add_input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool dequant,
    bool fuse_relu,
    int64_t input_scale,
    int64_t filter_scale,
    int64_t output_scale,
    int64_t add_scale,
    int64_t add_out_scale) {

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  c10::MaybeOwned<Tensor> add_input_maybe_owned = at::borrow_from_optional_tensor(add_input);
  const Tensor& add_input_tensor = *add_input_maybe_owned;
  bool fuse_add = add_input_tensor.defined();

  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(false, "zendnn_vitisai_convolution: bf16 path is not supported in zendnn yet");
  }

  const adeep::tensor zendnn_input = itensor_from_tensor(input);
  const adeep::tensor zendnn_weight = itensor_from_tensor(weight);
  c10::optional<adeep::tensor> zendnn_bias{c10::nullopt};
  if (bias.defined()) { zendnn_bias = itensor_from_tensor(bias); }

  float input_scale_pow_2 = std::pow(2, input_scale);
  float filter_scale_pow_2 = std::pow(2, filter_scale);
  float bias_scale_derived_pow_2 = std::pow(2, input_scale + filter_scale);
  float requantize_scale_pow_2 = std::pow(2, -input_scale - filter_scale + output_scale);
  float dequantize_scale_pow_2 = std::pow(2, -output_scale);
  float add_scale_pow_2 = 1.0f;
  float add_out_scale_pow_2 = 1.0f;
  if(fuse_add){
    if(output_scale != add_out_scale || output_scale != add_scale){
      requantize_scale_pow_2 = std::pow(2, -input_scale - filter_scale);
      dequantize_scale_pow_2 = std::pow(2, -add_out_scale);
      add_scale_pow_2 = std::pow(2, -add_scale);
      add_out_scale_pow_2 = std::pow(2, add_out_scale);
    }
  }

  auto kernel_size = zendnn_weight.get_dims();
  std::vector<int64_t> input_size = zendnn_input.get_dims();
  std::vector<int64_t> output_sizes = conv_output_size(input_size, kernel_size, padding, stride, dilation);
  adeep::tensor zendnn_output;
  if(fuse_add){
    zendnn_output = itensor_from_tensor(add_input_tensor);
  }


  adeep::convolution_forward::zen_vitis_compute(
      zendnn_input,
      zendnn_weight,
      zendnn_bias.value(),
      {output_sizes.cbegin(), output_sizes.cend()},
      zendnn_output,
      {stride.begin(), stride.end()},
      {dilation.begin(), dilation.end()},
      {padding.begin(), padding.end()},
      {padding.begin(), padding.end()},
      groups,
      dequant,
      fuse_relu,
      fuse_add,
      input_scale_pow_2,
      filter_scale_pow_2,
      bias_scale_derived_pow_2,
      requantize_scale_pow_2,
      dequantize_scale_pow_2,
      add_scale_pow_2,
      add_out_scale_pow_2,
      adeep::algorithm::convolution_direct,
      adeep::prop_kind::forward_inference);

   if (input.is_zendnn()) {
    return new_with_itensor_zendnn(std::move(zendnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                   input.options().device_opt());
  } else {
    return zendnn_to_dense(
        new_with_itensor_zendnn(std::move(zendnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
  }
}

}}

#endif