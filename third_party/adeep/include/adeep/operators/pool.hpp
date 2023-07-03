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

#ifndef ADEEP_OPERATORS_POOL_HPP
#define ADEEP_OPERATORS_POOL_HPP

namespace adeep {

struct pooling_forward : public zendnn::pooling_forward {

  using super = zendnn::pooling_forward;

  static void compute(const tensor& src,
                      const dims& output_sizes,
                      tensor& dst,
                      const dims& strides,
                      const dims& kernel,
                      const dims& padding_l,
                      const dims& padding_r,
                      algorithm aalgorithm,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    bool with_workspace = aprop_kind == prop_kind::forward_training &&
                          aalgorithm == zendnn::algorithm::pooling_max;

    // workaround: use src.get_desc() once issue intel/zen-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();

    tensor::desc dst_desc(output_sizes, src.get_data_type(), tag::any);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aprop_kind, aalgorithm, src_desc, dst_desc, strides, kernel, padding_l,
         padding_r}, op_attr, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());
    if (src.has_scale()) {
      dst.set_scale(src.get_scale());
    }

    tensor scratchpad(pd.scratchpad_desc());

    exec_args args {
        {ZENDNN_ARG_SRC, expected_src},
        {ZENDNN_ARG_DST, dst},
        {ZENDNN_ARG_SCRATCHPAD, scratchpad}};
    if (with_workspace) {
      dst.init_workspace(pd.workspace_desc());
      args.insert({ZENDNN_ARG_WORKSPACE, dst.get_workspace()});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct pooling_v2_forward : public zendnn::pooling_v2_forward {

  using super = zendnn::pooling_v2_forward;

  static void compute(const tensor& src,
                      const dims& output_sizes,
                      tensor& dst,
                      const dims& strides,
                      const dims& kernel,
                      const dims& dilation,
                      const dims& padding_l,
                      const dims& padding_r,
                      algorithm aalgorithm,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    bool with_workspace = aprop_kind == prop_kind::forward_training &&
                          aalgorithm == zendnn::algorithm::pooling_max;

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();

    tensor::desc dst_desc(output_sizes, src.get_data_type(), tag::any);

    auto dil_compatible = utils::get_compatible_dilates(dilation);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aprop_kind, aalgorithm, src_desc, dst_desc, strides, kernel,
         dil_compatible, padding_l, padding_r}, op_attr, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());
    if (src.has_scale()) {
      dst.set_scale(src.get_scale());
    }

    tensor scratchpad(pd.scratchpad_desc());

    exec_args args {
        {ZENDNN_ARG_SRC, expected_src},
        {ZENDNN_ARG_DST, dst},
        {ZENDNN_ARG_SCRATCHPAD, scratchpad}};

    if (with_workspace) {
      dst.init_workspace(pd.workspace_desc());
      args.insert({ZENDNN_ARG_WORKSPACE, dst.get_workspace()});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct pooling_backward : public zendnn::pooling_backward {

  using super = zendnn::pooling_backward;

  static void compute(const tensor& diff_dst,
                      const tensor& dst,
                      const tensor& src,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& kernel,
                      const dims& padding_l,
                      const dims& padding_r,
                      algorithm aalgorithm,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc().to_format_any();
    auto dst_desc = dst.get_desc();

    auto forward_hints =
        pooling_forward::primitive_desc(
            {prop_kind::forward, aalgorithm, src_desc, dst_desc, strides,
             kernel, padding_l, padding_r}, aengine);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aalgorithm, src_desc, dst_desc, strides, kernel, padding_l, padding_r},
        op_attr, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    tensor scratchpad(pd.scratchpad_desc());
    exec_args args {{ZENDNN_ARG_DIFF_DST, expected_diff_dst},
                    {ZENDNN_ARG_DIFF_SRC, diff_src},
                    {ZENDNN_ARG_SCRATCHPAD, scratchpad}};
    if (dst.has_workspace()) {
      auto expected_workspace =
          dst.get_workspace().reorder_if_differ_in(pd.workspace_desc());
      args.insert({ZENDNN_ARG_WORKSPACE, expected_workspace});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct pooling_v2_backward : public zendnn::pooling_v2_backward {

  using super = zendnn::pooling_v2_backward;

  static void compute(const tensor& diff_dst,
                      const tensor& dst,
                      const tensor& src,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& kernel,
                      const dims& dilation,
                      const dims& padding_l,
                      const dims& padding_r,
                      algorithm aalgorithm,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();
    auto dst_desc = dst.get_desc();
    auto dil_compatible = utils::get_compatible_dilates(dilation);

    auto forward_hints =
        pooling_v2_forward::primitive_desc(
            {prop_kind::forward, aalgorithm, src_desc, dst_desc, strides,
             kernel, dil_compatible, padding_l, padding_r}, aengine);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aalgorithm, src_desc, dst_desc, strides, kernel, dil_compatible,
         padding_l, padding_r}, op_attr, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args {{ZENDNN_ARG_DIFF_DST, expected_diff_dst},
                    {ZENDNN_ARG_DIFF_SRC, diff_src},
                    {ZENDNN_ARG_SCRATCHPAD, scratchpad}};
    if (dst.has_workspace()) {
      auto expected_workspace =
          dst.get_workspace().reorder_if_differ_in(pd.workspace_desc());
      args.insert({ZENDNN_ARG_WORKSPACE, expected_workspace});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace ideep

#endif
