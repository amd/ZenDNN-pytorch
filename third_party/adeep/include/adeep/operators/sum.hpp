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
