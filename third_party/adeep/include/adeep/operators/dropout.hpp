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

#ifndef ADEEP_OPERATORS_DROPOUT_HPP
#define ADEEP_OPERATORS_DROPOUT_HPP

namespace adeep {

struct dropout_forward {
  static void compute(const tensor& src, float ratio, tensor& dst,
                      tensor& mask) {
    switch (src.get_data_type()) {
      case data_type::f32:
        compute_impl<float>(src, ratio, dst, mask);
        break;
      case data_type::s32:
        compute_impl<int32_t>(src, ratio, dst, mask);
        break;
      case data_type::s8:
        compute_impl<int8_t>(src, ratio, dst, mask);
        break;
      case data_type::u8:
        compute_impl<uint8_t>(src, ratio, dst, mask);
        break;
      default:
        throw error(zendnn_invalid_arguments, "Unsupported zendnn data type");
    }
  }

 private:
  template <typename T>
  static void compute_impl(const tensor& src, float ratio, tensor& dst,
                           tensor& mask) {
    mask.reinit_if_possible(src.get_desc());
    dst.reinit_if_possible(src.get_desc());
    if (src.has_scale()) {
      dst.set_scale(src.get_scale());
    }

    const auto scale = 1.0 / (1.0 - ratio);
    const auto size = src.get_size() / sizeof(T);
    std::unique_ptr<int[]> bernouli_nums(new int[size]);
    utils::bernoulli_generate(size, 1.0 - ratio, bernouli_nums.get());

    const auto src_data = static_cast<T*>(src.get_data_handle());
    const auto mask_data = static_cast<T*>(mask.get_data_handle());
    const auto dst_data = static_cast<T*>(dst.get_data_handle());
#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd
#else
# pragma omp parallel for schedule(static)
#endif
#endif
    for (auto i = 0; i < size; i++) {
      mask_data[i] = bernouli_nums[i] * scale;
      dst_data[i] = mask_data[i] * src_data[i];
    }
  }
};

struct dropout_backward {
  static void compute(const tensor& mask, const tensor& diff_dst,
                      tensor& diff_src) {
    switch (diff_dst.get_data_type()) {
      case data_type::f32:
        compute_impl<float>(mask, diff_dst, diff_src);
        break;
      case data_type::s32:
        compute_impl<int32_t>(mask, diff_dst, diff_src);
        break;
      case data_type::s8:
        compute_impl<int8_t>(mask, diff_dst, diff_src);
        break;
      case data_type::u8:
        compute_impl<uint8_t>(mask, diff_dst, diff_src);
        break;
      default:
        throw error(zendnn_invalid_arguments, "Unsupported zendnn data type!");
    }
  }

 private:
  template <typename T>
  static void compute_impl(const tensor& mask, const tensor& diff_dst,
                           tensor& diff_src) {
    diff_src.reinit_if_possible(diff_dst.get_desc());

    const auto size = mask.get_size() / sizeof(T);
    const auto mask_data = static_cast<T*>(mask.get_data_handle());
    const auto diff_dst_data = static_cast<T*>(diff_dst.get_data_handle());
    const auto diff_src_data = static_cast<T*>(diff_src.get_data_handle());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (auto i = 0; i < size; i++) {
      diff_src_data[i] = mask_data[i] * diff_dst_data[i];
    }
  }
};

}  // namespace adeep

#endif