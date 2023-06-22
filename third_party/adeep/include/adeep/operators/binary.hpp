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
