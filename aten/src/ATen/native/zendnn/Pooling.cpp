/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/zendnn/Utils.h>
#include <tuple>


#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor zendnn_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(false, "zendnn_max_pool2d: ATen not compiled with ZENDNN support");
}

Tensor zendnn_max_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(false, "zendnn_max_pool3d: ATen not compiled with ZENDNN support");
}

Tensor zendnn_avg_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(false, "zendnn_avg_pool2d: ATen not compiled with ZENDNN support");
}

Tensor& zendnn_avg_pool2d_out(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& output) {
  TORCH_CHECK(false, "zendnn_avg_pool2d_out: ATen not compiled with ZENDNN support");
}

Tensor zendnn_avg_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(false, "zendnn_avg_pool3d: ATen not compiled with ZENDNN support");
}

Tensor& zendnn_avg_pool3d_out(const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& output) {
  TORCH_CHECK(false, "zendnn_avg_pool3d_out: ATen not compiled with ZENDNN support");
}

Tensor zendnn_adaptive_avg_pool2d(Tensor const& input, IntArrayRef output_size) {
  TORCH_CHECK(false, "zendnn_adaptive_avg_pool2d: ATen not compiled with ZENDNN support");
}

Tensor& zendnn_adaptive_avg_pool2d_out(const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  TORCH_CHECK(false, "zendnn_adaptive_avg_pool2d_out: ATen not compiled with ZENDNN support");
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>

namespace at {
namespace native {

static Tensor _zendnn_pooling(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    adeep::algorithm algo) {
  const int64_t dims = input.dim() - 2;
  auto kernel_size_vec = expand_param_if_needed(kernel_size, "kernel_size", dims);
  if (stride.empty()) stride = kernel_size;
  auto stride_vec = expand_param_if_needed(stride, "stride", dims);
  auto padding_vec = expand_param_if_needed(padding, "padding", dims);
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto padding_vec_l = padding_vec;
  auto padding_vec_r = padding_vec;
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

  const adeep::tensor& x = itensor_from_zendnn(input);
  std::vector<int64_t> output_sizes;

  if (ceil_mode) {
    // ZENDNN does not support ceil mode, so we adjust padding
    // on the right side to match behavior. Adjust output size
    // accordingly.
    const std::vector<int64_t> output_sizes_ceil = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        true /* ceil_mode */);

    // adjust padding until output sizes agree
    bool all_equal = false;
    while (!all_equal) {
      output_sizes = pool_output_sizes(
          input.sizes(),
          kernel_size_vec,
          stride_vec,
          padding_vec_l,
          padding_vec_r,
          dilation_vec,
          false /*ceil_mode */);

      all_equal = true;
      for (size_t i = 2; i < input.sizes().size(); ++i) {
        if (output_sizes[i] < output_sizes_ceil[i]) {
           padding_vec_r[i - 2]++;
           all_equal = false;
        }
      }
    }
  } else {
    output_sizes = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        false /*ceil_mode */);
  }

  auto aprop_kind = adeep::prop_kind::forward;
  // for max_pool, prop_kind::forward will save indices as workspace for backward use,
  // for inference, don't need the indices, set aprop_kind to prop_kind::forward_inference
  // can reduce the memory use.
  // for quantized pooling_avg_include_padding prop_kind::forward
  // throws a primitive creation error and requires prop_kind::forward_inference
  if ((adeep::algorithm::pooling_max == algo
      && !(input.requires_grad() && at::GradMode::is_enabled()))
      || (adeep::algorithm::pooling_avg_include_padding == algo 
      && (x.get_data_type() == adeep::data_type::s8) || (x.get_data_type() == adeep::data_type::u8))) {
    aprop_kind = adeep::prop_kind::forward_inference;
  }

  adeep::tensor y;
  adeep::pooling_forward::compute(
      x,
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride_vec.cbegin(), stride_vec.cend()},
      {kernel_size_vec.cbegin(), kernel_size_vec.cend()},
      {padding_vec_l.cbegin(), padding_vec_l.cend()},
      {padding_vec_r.cbegin(), padding_vec_r.cend()},
      algo,
      aprop_kind);

  return new_with_itensor_zendnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()), input.options().device_opt());
}

Tensor zendnn_max_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_max_pool2d: bf16 path needs the cpu support avx512bf16");
  }

  TORCH_CHECK(std::all_of(dilation.cbegin(), dilation.cend(), [](int64_t i) { return 1 == i; }),
      "zendnn_max_pool2d does not support dilation case");

  return _zendnn_pooling(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      adeep::algorithm::pooling_max);
}

Tensor zendnn_max_pool3d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(std::all_of(dilation.cbegin(), dilation.cend(), [](int64_t i) { return 1 == i; }),
      "zendnn_max_pool3d does not support dilation case");
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(false,
        "zendnn_max_pool3d: bf16 path is not supported in zendnn yet");
  }

  return _zendnn_pooling(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      adeep::algorithm::pooling_max);
}

Tensor zendnn_avg_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_avg_pool2d: bf16 path needs the cpu support avx512bf16");
  }

  TORCH_CHECK(!divisor_override.has_value(),
      "zendnn_avg_pool2d operator does not support divisor");

  return _zendnn_pooling(
      input,
      kernel_size,
      stride,
      padding,
      /*dilation*/ std::vector<int64_t>{1, 1},
      ceil_mode,
      count_include_pad ? adeep::algorithm::pooling_avg_include_padding
                        : adeep::algorithm::pooling_avg_exclude_padding);
}

Tensor& zendnn_avg_pool2d_out(const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& output) {
  TORCH_CHECK(false, "zendnn_avg_pool2d_out: in-place zendnn operations are not supported yet");
}

Tensor zendnn_avg_pool3d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(!divisor_override.has_value(), "zendnn_avg_pool3d operator does not support divisor");
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(false,
        "zendnn_avg_pool3d: bf16 path is not supported in zendnn yet");
  }

  return _zendnn_pooling(
      input,
      kernel_size,
      stride,
      padding,
      /*dilation*/ std::vector<int64_t>{1, 1, 1},
      ceil_mode,
      count_include_pad ? adeep::algorithm::pooling_avg_include_padding
                        : adeep::algorithm::pooling_avg_exclude_padding);
}

Tensor& zendnn_avg_pool3d_out(const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& output) {
  TORCH_CHECK(false, "zendnn_avg_pool3d_out: in-place zendnn operations are not supported yet");
}

Tensor zendnn_adaptive_avg_pool2d(
    Tensor const& input,
    IntArrayRef output_size) {
  TORCH_CHECK(input.dim() == 4, "zendnn_adaptive_avg_pool2d: Expect 2D input");

  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_adaptive_avg_pool2d: bf16 path needs the cpu support avx512bf16");
  }

  auto output_size_vec =
      expand_param_if_needed(output_size, "output_size", input.dim() - 2);
  std::vector<int64_t> kernel_size(input.dim() - 2);
  for (int64_t i = 2; i < input.dim(); ++i) {
    auto s1 = input.size(i);
    auto s2 = output_size_vec[i - 2];
    TORCH_CHECK(s2 != 0, "output size can not be zero");
    TORCH_CHECK(
        s1 % s2 == 0,
        "input size is not divisible by the output size is not supported yet");
    kernel_size[i - 2] = s1 / s2;
  }
  return _zendnn_pooling(
      input,
      kernel_size,
      /*stride*/ kernel_size,
      /*padding*/ {0, 0},
      /*dilation*/ {1, 1},
      /*ceil_mode*/ false,
      /*algo*/ adeep::algorithm::pooling_avg);
}

Tensor& zendnn_adaptive_avg_pool2d_out(const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  TORCH_CHECK(false, "zendnn_adaptive_avg_pool2d_out: in-place zendnn operations are not supported yet");
}


} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED
