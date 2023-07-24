/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#ifndef ADEEP_OPERATORS_BINARY_HPP
#define ADEEP_OPERATORS_BINARY_HPP

namespace adeep {

struct binary : public zendnn::binary {

  using super = zendnn::binary;

  static void compute(const tensor& src0,
                      const tensor& src1,
                      tensor& dst,
                      algorithm aalgorithm,
                      const engine& aengine = engine::cpu_engine()) {
    auto src0_desc = src0.get_desc();
    auto src1_desc = src1.get_desc();
    auto dst_desc = src0_desc.to_format_any();

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aalgorithm, src0_desc, src1_desc, dst_desc}, op_attr, aengine);

    tensor scratchpad(pd.scratchpad_desc());

    auto expected_src0 = src0.reorder_if_differ_in(pd.src0_desc());
    auto expected_src1 = src1.reorder_if_differ_in(pd.src1_desc());
    dst.reinit_if_possible(pd.dst_desc());

    super(pd).execute(stream::default_stream(),
                      {{ZENDNN_ARG_SRC_0, expected_src0},
                       {ZENDNN_ARG_SRC_1, expected_src1},
                       {ZENDNN_ARG_DST, dst},
                       {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
  }
};

}  // namespace adeep

#endif
