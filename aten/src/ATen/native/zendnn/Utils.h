/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#pragma once
#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <vector>
#include <cpuinfo.h>
namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> zendnn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,
    IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps, bool inplace = false);

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef dilation,
    bool ceil_mode);
};

inline bool zendnn_bf16_device_check() {
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512bf16();
}

}
