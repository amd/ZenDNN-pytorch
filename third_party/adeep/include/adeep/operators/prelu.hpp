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

#ifndef ADEEP_OPERATORS_PRELU_HPP
#define ADEEP_OPERATORS_PRELU_HPP

namespace adeep {

struct prelu_forward : public zendnn::prelu_forward {

  using super = zendnn::prelu_forward;

  static void compute(const tensor& src,
                      const tensor& weight,
                      tensor& dst,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_in = src;
    auto weight_in = weight;

    // Reshape weight to src dimension
    auto new_dims = src.get_dims();
    if (src.ndims() != weight.ndims()) {
      std::vector<dim> dim_w(src.ndims(), 1);
      dim_w[1] = weight.get_dim(0);
      weight_in.reshape(dim_w);
    }

    auto src_desc = src_in.get_desc();
    auto weight_desc = weight_in.get_desc().to_format_any();

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc({aprop_kind, src_desc, weight_desc}, op_attr, aengine);
    auto expected_weights = weight_in.reorder_if_differ_in(pd.weights_desc());
    dst.reinit_if_possible(pd.dst_desc());

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(stream::default_stream(),
                      {{ZENDNN_ARG_SRC, src_in},
                       {ZENDNN_ARG_WEIGHTS, expected_weights},
                       {ZENDNN_ARG_DST, dst},
                       {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
  }
};

}  // namespace adeep

#endif
