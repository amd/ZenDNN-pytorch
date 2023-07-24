/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#ifndef ADEEP_OPERATORS_LRN_HPP
#define ADEEP_OPERATORS_LRN_HPP

namespace adeep {

struct lrn_forward : public zendnn::lrn_forward {

  using super = zendnn::lrn_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      dim local_size,
                      float alpha,
                      float beta,
                      float k = 1.0,
                      algorithm aalgorithm = algorithm::lrn_across_channels,
                      prop_kind aprop_kind = prop_kind::forward_training,
                      const engine& aengine = engine::cpu_engine()) {

    // workaround: use src.get_desc() once issue intel/zen-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    // auto src_desc = src.get_desc();
    auto pd = primitive_desc(
        {aprop_kind, aalgorithm, src_desc, local_size, alpha, beta, k},
        op_attr,
        aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args {
        {ZENDNN_ARG_SRC, expected_src},
        {ZENDNN_ARG_DST, dst},
        {ZENDNN_ARG_SCRATCHPAD, scratchpad}};

    bool with_workspace = aprop_kind == prop_kind::forward_training;
    if (with_workspace) {
      dst.init_workspace(pd.workspace_desc());
      args.insert({ZENDNN_ARG_WORKSPACE, dst.get_workspace()});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct lrn_backward : public zendnn::lrn_backward {

  using super = zendnn::lrn_backward;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const tensor& dst,
                      tensor& diff_src,
                      dim local_size,
                      float alpha,
                      float beta,
                      float k = 1.0,
                      algorithm aalgorithm = algorithm::lrn_across_channels,
                      const engine& aengine = engine::cpu_engine()) {

    // workaround: use src.get_desc() once issue intel/zen-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();
    auto forward_hints =
        lrn_forward::primitive_desc({prop_kind::forward_training, aalgorithm,
                                     src_desc, local_size, alpha, beta, k},
                                    aengine);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aalgorithm, src_desc, diff_dst.get_desc(), local_size, alpha, beta, k},
        op_attr, aengine, forward_hints);
    
    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args {{ZENDNN_ARG_SRC, src},
                    {ZENDNN_ARG_DIFF_DST, expected_diff_dst},
                    {ZENDNN_ARG_DIFF_SRC, diff_src},
                    {ZENDNN_ARG_SCRATCHPAD, scratchpad}};

    if (dst.has_workspace()) {
      args.insert({ZENDNN_ARG_WORKSPACE, dst.get_workspace()});
    }
    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace adeep

#endif
