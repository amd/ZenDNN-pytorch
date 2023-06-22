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

#ifndef ADEEP_OPERATORS_SOFTMAX_HPP
#define ADEEP_OPERATORS_SOFTMAX_HPP

namespace adeep {

struct softmax_forward : public zendnn::softmax_forward {

  using super = zendnn::softmax_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      int softmax_axis,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();
    dst.reinit_if_possible(src_desc);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aprop_kind, src_desc, softmax_axis}, op_attr, aengine);
    tensor scratchpad(pd.scratchpad_desc());
    super(pd).execute(
        stream::default_stream(),
        {{ZENDNN_ARG_SRC, src},
        {ZENDNN_ARG_DST, dst},
        {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
  }
};

struct softmax_backward : public zendnn::softmax_backward {

  using super = zendnn::softmax_backward;

  static void compute(const tensor& dst,
                      const tensor& diff_dst,
                      tensor& diff_src,
                      int softmax_axis,
                      const engine& aengine = engine::cpu_engine()) {

    auto forward_hints = softmax_forward::primitive_desc(
        {prop_kind::forward_inference, dst.get_desc(), softmax_axis}, aengine);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd =
        primitive_desc({diff_dst.get_desc(), dst.get_desc(), softmax_axis},
                       op_attr, aengine, forward_hints);
    auto expected_dst = dst.reorder_if_differ_in(pd.dst_desc());
    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(stream::default_stream(),
                      {{ZENDNN_ARG_DST, expected_dst},
                       {ZENDNN_ARG_DIFF_DST, expected_diff_dst},
                       {ZENDNN_ARG_DIFF_SRC, diff_src},
                       {ZENDNN_ARG_SCRATCHPAD, scratchpad}});

  }
};

}  // namespace adeep

#endif
