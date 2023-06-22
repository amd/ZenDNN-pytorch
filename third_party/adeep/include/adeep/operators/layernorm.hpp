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

#ifndef ADEEP_OPERATORS_LAYERNORM_HPP
#define ADEEP_OPERATORS_LAYERNORM_HPP

namespace adeep {

struct layer_normalization_forward : public zendnn::layer_normalization_forward {

  using super = zendnn::layer_normalization_forward;

  static void compute(const tensor& src,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      tensor& mean,
                      tensor& variance,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
    auto flags = batch_normalization_flag::use_scale_shift;
    auto src_desc = src.get_desc();
    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);
    auto pd = primitive_desc(
        {prop_kind::forward_training, src_desc, epsilon, flags},
        op_attr,
        aengine);

    tensor scale_shift {pd.weights_desc()};
    auto* scale_shift_buf = static_cast<char *>(scale_shift.get_data_handle());
    std::memcpy(scale_shift_buf, scale.get_data_handle(), scale.get_size());
    std::memcpy(scale_shift_buf + scale.get_size(),
                shift.get_data_handle(), shift.get_size());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    mean.reinit_if_possible(pd.mean_desc());
    variance.reinit_if_possible(pd.variance_desc());
    dst.reinit_if_possible(pd.dst_desc());
    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(stream::default_stream(),
                      {{ZENDNN_ARG_SRC, expected_src},
                       {ZENDNN_ARG_SCALE_SHIFT, scale_shift},
                       {ZENDNN_ARG_MEAN, mean},
                       {ZENDNN_ARG_VARIANCE, variance},
                       {ZENDNN_ARG_DST, dst},
                       {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
  }
};

struct layer_normalization_backward :
    public zendnn::layer_normalization_backward {
  static void compute() {
  }
};

}  // namespace adeep

#endif
