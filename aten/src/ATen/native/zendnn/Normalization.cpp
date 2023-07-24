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

std::tuple<Tensor, Tensor, Tensor> zendnn_batch_norm(
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(false, "zendnn_batch_norm: ATen not compiled with ZENDNN support");
}

Tensor zendnn_layer_norm(
    at::Tensor const& input,
    at::Tensor const& weight,
    at::Tensor const& bias,
    double eps) {
  TORCH_CHECK(false, "zendnn_layer_norm: ATen not compiled with ZENDNN support");
}

std::tuple<Tensor, Tensor, Tensor> zendnn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,
    IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps, bool inplace) {
  TORCH_CHECK(false, "zendnn_layer_norm_last_index_weight_bias_f32: ATen not compiled with ZENDNN support");
}


} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>
#include <ATen/native/layer_norm.h>
#include <adeep/abstract_types.hpp>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> zendnn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,
    IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps, bool inplace) {

  TORCH_INTERNAL_ASSERT(normalized_shape.size() == 1, "only accept shapes with the last dimension");
  TORCH_INTERNAL_ASSERT(input.scalar_type() == at::kFloat);
  auto M_N = at::native::_check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;

  auto mean = empty_zendnn(
        {M},
        input.scalar_type(),
        input.options().layout_opt(),
        input.options().device_opt(),
        input.options().pinned_memory_opt());
  auto rstd = empty_zendnn(
        {M},
        input.scalar_type(),
        input.options().layout_opt(),
        input.options().device_opt(),
        input.options().pinned_memory_opt());

  auto mean_it = at::native::itensor_from_zendnn(mean);
  auto rstd_it = at::native::itensor_from_zendnn(rstd);

  auto input_it = at::native::itensor_from_zendnn(input);
  auto weight_it = at::native::itensor_from_zendnn(weight);
  auto bias_it = at::native::itensor_from_zendnn(bias);

  auto out_it = inplace ? input_it : adeep::tensor(input_it.get_desc());
  adeep::layer_normalization_forward::compute(input_it, weight_it, bias_it, out_it, mean_it, rstd_it, static_cast<float>(eps));

  auto dst = at::native::new_with_itensor_zendnn(
      std::move(out_it),
      optTypeMetaToScalarType(input.options().dtype_opt()),
      input.options().device_opt());

  return std::make_tuple(dst, mean, rstd);
}


std::tuple<Tensor, Tensor, Tensor> zendnn_batch_norm(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(false,
        "zendnn_batch_norm: bf16 path is not supported in zendnn yet");
  }
  TORCH_CHECK(weight.defined() && bias.defined(),
             "zendnn_batch_norm: currently zendnn only support affine model");

  adeep::tensor& x = itensor_from_zendnn(input);
  adeep::tensor w = itensor_from_tensor(weight);
  adeep::tensor b = itensor_from_tensor(bias);
  bool use_running_stat = (running_mean.defined() && running_var.defined());

  adeep::tensor y;

  if (train) {
    // TODO: enable 3d batchnorm.
    TORCH_CHECK(input.dim() == 4,
        "zendnn_batch_norm: currently zendnn training only support 2d batchnorm");
    adeep::tensor saved_mean;
    adeep::tensor saved_var;
    adeep::batch_normalization_forward_training::compute(
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        x, w, b, y, saved_mean, saved_var, momentum, eps);
    if (use_running_stat) {
      auto len = x.get_nelems() / w.get_nelems(); // n*h*w
      adeep::tensor m = itensor_from_tensor(running_mean);
      adeep::tensor v = itensor_from_tensor(running_var);
      const std::vector<float> scales_mean{static_cast<float>(1 - momentum),
                                           static_cast<float>(momentum)};
      const std::vector<float> scales_var{static_cast<float>(1 - momentum),
                                          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
                                          static_cast<float>(momentum * len / (len - 1))};
      adeep::sum::compute(scales_mean, {m, saved_mean}, m);
      adeep::sum::compute(scales_var, {v, saved_var}, v);
    }
    return std::make_tuple(
         new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt()),
         new_with_itensor_zendnn(std::move(saved_mean), optTypeMetaToScalarType(weight.options().dtype_opt()),
                                 weight.options().device_opt()),
         new_with_itensor_zendnn(std::move(saved_var), optTypeMetaToScalarType(weight.options().dtype_opt()),
                                 weight.options().device_opt()));
  } else {
    TORCH_CHECK(input.dim() == 4 || input.dim() == 5,
        "zendnn_batch_norm: currently zendnn inference only support 2d and 3d batchnorm");
    if (use_running_stat) {
      adeep::tensor m = itensor_from_tensor(running_mean);
      adeep::tensor v = itensor_from_tensor(running_var);
      adeep::batch_normalization_forward_inference::compute(
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          x, m, v, w, b, y, eps);
    } else {
      // TODO: keep running estimates.
      TORCH_CHECK(false, "zendnn_batch_norm: zendnn inference is not keep running estimates.");
    }
    return std::make_tuple(
        new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()),
        new_with_itensor_zendnn(adeep::tensor{}, optTypeMetaToScalarType(weight.options().dtype_opt()),
                                weight.options().device_opt()),
        new_with_itensor_zendnn(adeep::tensor{}, optTypeMetaToScalarType(weight.options().dtype_opt()),
                                weight.options().device_opt()));
  }
}

Tensor zendnn_layer_norm(
    at::Tensor const& input,
    at::Tensor const& weight,
    at::Tensor const& bias,
    double eps) {

  TORCH_CHECK(input.is_zendnn(),
      "zendnn_linear: input needs to be in zendnn layout");

  // c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);

  // const Tensor& w1 = *weight_maybe_owned;
  // c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
  // const Tensor& b1 = *bias_maybe_owned;

  TORCH_CHECK(weight.defined() && bias.defined(),
             "zendnn_layer_norm: currently ZenDNN only supports affine model");

  adeep::tensor& x = itensor_from_zendnn(input);
  adeep::tensor& w = itensor_from_zendnn(weight);
  adeep::tensor& b = itensor_from_zendnn(bias);

  adeep::tensor y;
  adeep::tensor mean;
  adeep::tensor variance;

  adeep::layer_normalization_forward::compute(x, w, b, y, mean, variance, eps);

  return  new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                              input.options().device_opt());
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED
