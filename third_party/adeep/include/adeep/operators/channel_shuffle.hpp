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

#ifndef ADEEP_OPERATORS_CHANNEL_SHUFFLE_HPP
#define ADEEP_OPERATORS_CHANNEL_SHUFFLE_HPP

namespace adeep {

struct channel_shuffle_forward: public zendnn::shuffle_forward {

  using super = zendnn::shuffle_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      const int group,
                      const int axis = 1,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    ADEEP_ENFORCE(src.get_dim(axis) % group == 0, "Invalid channel and group");
    ADEEP_ENFORCE(src.get_data_type() == data_type::f32, "invalid data type");

    auto group_size = static_cast<int>(src.get_dim(axis) / group);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd =
        primitive_desc({aprop_kind, src.get_desc(), axis, group_size}, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(stream::default_stream(),
                      {{ZENDNN_ARG_SRC, expected_src},
                       {ZENDNN_ARG_DST, dst},
                       {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
  }
};

struct channel_shuffle_backward : public zendnn::shuffle_backward {

  using super = zendnn::shuffle_backward;

  static void compute(const tensor& diff_dst,
                      tensor& diff_src,
                      const int group,
                      const int axis = 1,
                      const engine& aengine = engine::cpu_engine()) {
    auto group_size = static_cast<int>(diff_dst.get_dim(axis) / group);
    auto data_desc = diff_dst.get_desc();

    auto forward_hints = zendnn::shuffle_forward::primitive_desc(
        {prop_kind::forward, data_desc, group_size, axis}, aengine);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc(
        {data_desc, axis, group_size}, aengine, forward_hints, op_attr);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(stream::default_stream(),
                      {{ZENDNN_ARG_DIFF_DST, expected_diff_dst},
                       {ZENDNN_ARG_DIFF_SRC, diff_src},
                       {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
  }
};

}  // namespace adeep

#endif
