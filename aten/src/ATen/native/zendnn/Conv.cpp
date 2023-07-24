/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/zendnn/Utils.h>

#if !AT_ZENDNN_ENABLED()

namespace at { namespace native {

Tensor zendnn_convolution(
    const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(false, "zendnn_convolution_forward: ATen not compiled with ZENDNN support");
}

Tensor zendnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  TORCH_CHECK(false, "zendnn_convolution_backward_input: ATen not compiled with ZENDNN support");
}

std::tuple<Tensor, Tensor> zendnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  TORCH_CHECK(false, "zendnn_convolution_backward_weights: ATen not compiled with ZENDNN support");
}

std::tuple<Tensor, Tensor, Tensor> zendnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "zendnn_convolution_backward: ATen not compiled with ZENDNN support");
}

REGISTER_NO_CPU_DISPATCH(zendnn_convolution_backward_stub);
}}

#else // AT_ZENDNN_EBABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
// #include <ATen/native/zendnn/Utils.h>
#include <ATen/native/ConvUtils.h>

namespace at { namespace native {

adeep::tensor _zendnn_convolution(
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
        groups);
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
        groups);
  }
  return y;
}

Tensor zendnn_convolution(
    const Tensor& input,
    const Tensor& weight, const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_convolution: bf16 path needs the cpu support avx512bf16");
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

  adeep::tensor zendnn_output = _zendnn_convolution(
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

Tensor zendnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  // for training case, grad_output can be cpu tensor or ZENDNN tensor,
  // but weight and bias always cpu tensor.
  auto zendnn_grad_output = itensor_from_tensor(grad_output);
  auto zendnn_weight = itensor_view_from_dense(weight);

  adeep::tensor zendnn_grad_input;
  adeep::convolution_backward_data::compute(
      zendnn_grad_output,
      zendnn_weight,
      input_size.vec(),
      zendnn_grad_input,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups);

  if (grad_output.is_zendnn()) {
    return new_with_itensor_zendnn(std::move(zendnn_grad_input),
                                   optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                   grad_output.options().device_opt());

  } else {
    return zendnn_to_dense(new_with_itensor_zendnn(std::move(zendnn_grad_input),
                                                   optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                                   grad_output.options().device_opt()));
  }
}

std::tuple<Tensor, Tensor> zendnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  // for training case, grad_output and input can be cpu tensor or ZENDNN tensor,
  // but weight and bias are always cpu tensor.
  const adeep::tensor zendnn_grad_output = itensor_from_tensor(grad_output);
  const adeep::tensor zendnn_input = itensor_from_tensor(input);

  adeep::tensor zendnn_grad_weight, zendnn_grad_bias;
  if (bias_defined) {
    adeep::convolution_backward_weights::compute(
        zendnn_input,
        zendnn_grad_output,
        weight_size.vec(),
        zendnn_grad_weight,
        zendnn_grad_bias,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  } else {
    adeep::convolution_backward_weights::compute(
        zendnn_input,
        zendnn_grad_output,
        weight_size.vec(),
        zendnn_grad_weight,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  }

  return std::make_tuple(
      zendnn_to_dense(new_with_itensor_zendnn(std::move(zendnn_grad_weight),
                                              optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                              grad_output.options().device_opt())),
      zendnn_to_dense(new_with_itensor_zendnn(std::move(zendnn_grad_bias),
                                              optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                              grad_output.options().device_opt())));
}

std::tuple<Tensor, Tensor, Tensor> zendnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  Tensor grad_output = grad_output_t.is_zendnn() ? grad_output_t : grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::zendnn_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::zendnn_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

REGISTER_ALL_CPU_DISPATCH(zendnn_convolution_backward_stub, &zendnn_convolution_backward);

}}  // namespace at::native

#endif
