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

#include <ATen/native/layer_norm.h>

#include <ATen/AccumulateType.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <array>
#include <functional>
#include <numeric>
#include <tuple>
#include <vector>

namespace at {
namespace native {

void layer_norm_with_mean_rstd_out(
    at::Tensor& out,
    at::Tensor& mean,
    at::Tensor& rstd,
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& gamma,
    const Tensor& beta,
    double eps,
    int64_t M,
    int64_t N) {
  LayerNormKernel(kCPU, input, gamma, beta, M, N, eps, &out, &mean, &rstd);
  const auto input_shape = input.sizes();
  const size_t axis = input.dim() - normalized_shape.size();

  DimVector stat_shape;
  for (const auto idx : c10::irange(axis)) {
    stat_shape.emplace_back(input_shape[idx]);
  }
  for (const auto idx C10_UNUSED : c10::irange(axis, input.dim())) {
    stat_shape.emplace_back(1);
  }

  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);
}

void layer_norm_cpu_out(
    at::Tensor& out,
    const at::Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    double eps,
    int64_t M,
    int64_t N) {
  if (M <= 0) {
    return;
  }
  LayerNormKernel(kCPU, input, gamma, beta, M, N, eps, &out, /*mean=*/nullptr, /*rstd=*/nullptr);
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_cpu(
    const Tensor& input,
    IntArrayRef normalized_shape, const c10::optional<Tensor>& weight_opt /* optional */, const c10::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;


  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor Y = at::native::empty_like(
      *X,
      c10::nullopt /* dtype */,
      c10::nullopt /* layout */,
      c10::nullopt /* device */,
      c10::nullopt /* pin_memory */,
      at::MemoryFormat::Contiguous);
  Tensor mean = at::empty({M}, X->options());
  Tensor rstd = at::empty({M}, X->options());

  layer_norm_with_mean_rstd_out(Y, mean, rstd, *X, normalized_shape, *gamma, *beta, eps, M, N);
  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu(
    const Tensor& dY,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    std::array<bool, 3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(
        *X,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous)
                   : at::native::zeros_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous)
                  : at::native::zeros_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous);
  }
  if (M > 0) {
    LayerNormBackwardKernel(
        kCPU, dY, *X, mean, rstd, *gamma, M, N, &dX, &dgamma, &dbeta);
  }
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}

Tensor layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape, const c10::optional<Tensor>& weight_opt /* optional */, const c10::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  #if AT_ZENDNN_ENABLED()
  if (input.is_zendnn()) {
    return at::zendnn_layer_norm(input, weight, bias, eps);
  }
  #endif
  return std::get<0>(at::native_layer_norm(input, normalized_shape, weight, bias, eps));
}

DEFINE_DISPATCH(LayerNormKernel);
DEFINE_DISPATCH(LayerNormBackwardKernel);

// Ported from pytorch/xla repo
std::tuple<Tensor, Tensor, Tensor> math_native_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();

  auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  const int normalized_ndim = normalized_shape.size();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const int axis = input_ndim - normalized_ndim;

  // Properly handle zero-size inputs: the view(1, M, -1) call below breaks on this.
  if (input.numel() == 0) {
    auto result_type = c10::promoteTypes(input.scalar_type(), kFloat);
    return std::make_tuple(
      at::empty_like(input),
      at::empty_like(input, c10::TensorOptions().dtype(result_type)),
      at::empty_like(input, c10::TensorOptions().dtype(result_type))
    );
  }
  at::Tensor input_reshaped = input.reshape({1, M, -1});
  // Unlike Batch Normalization, which applies scalar scale and bias for each
  // entire channel/plane with the affine option, Layer Normalization applies
  // per-element scale and bias. E.g. For input {N, C, H, W}, weight for
  // batchnorm has shape {C} while weight for layernorm has shape {H, W} or {W}.
  auto outputs = at::native_batch_norm(
      input_reshaped, /*weight=*/{}, /*bias=*/{}, /*running_mean=*/{},
      /*running_var=*/{}, /*training=*/true, /*momentum=*/0, eps);
  at::Tensor out = std::get<0>(outputs);
  out = out.view(input_shape);
  if (weight.defined() && bias.defined()) {
    out = bias.addcmul(out, weight, 1);
  } else if (weight.defined()) {
    out = out.mul(weight);
  } else if (bias.defined()) {
    out = out.add(bias);
  }
  at::Tensor mean = std::get<1>(outputs);
  at::Tensor rstd = std::get<2>(outputs);
  std::vector<int64_t> stat_shape;
  for (const auto idx : c10::irange(axis)) {
    stat_shape.push_back(input_shape[idx]);
  }
  for (const auto idx C10_UNUSED : c10::irange(axis, input.dim())) {
    stat_shape.push_back(1);
  }
  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);
  return std::make_tuple(out, mean, rstd);
}
} // namespace native
} // namespace at
