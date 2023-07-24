/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#ifndef ADEEP_OPERATORS_SUM_HPP
#define ADEEP_OPERATORS_SUM_HPP

namespace adeep {

struct sum : public zendnn::sum {

  using super = zendnn::sum;

  static void compute(const scale_t& scales,
                      const std::vector<tensor>& srcs,
                      tensor& dst,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_descs = utils::fmap(srcs, [](const tensor& t) {
      // "upcast" vector<tensor::desc> to vector<memory::desc>
      return static_cast<memory::desc>(t.get_desc());
    });

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc(scales, src_descs, aengine, op_attr);

    dst.reinit_if_possible(pd.dst_desc());
    tensor scratchpad(pd.scratchpad_desc());
    exec_args args {{ZENDNN_ARG_DST, dst}, {ZENDNN_ARG_SCRATCHPAD, scratchpad}};
    for (int i = 0; i < srcs.size(); ++i) {
      args.insert({ZENDNN_ARG_MULTIPLE_SRC + i, srcs[i]});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace adeep

#endif
