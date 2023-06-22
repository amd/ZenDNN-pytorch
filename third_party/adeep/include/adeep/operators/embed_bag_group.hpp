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

#ifndef ADEEP_OPERATORS_EMBED_BAG_GROUP_HPP
#define ADEEP_OPERATORS_EMBED_BAG_GROUP_HPP

#include <vector>

#define ADEEP_EMBED_BAG_THRDS        (1)

namespace adeep {
struct embed_bag_group : public zendnn::embedding_bag {

    using super          = zendnn::embedding_bag;
    using tensor_vector  = std::vector<tensor>;

    static void compute(tensor        &self,
                        tensor_vector &weights,
                        tensor_vector &indices,
                        tensor_vector &offsets,
                        tensor        &per_sample_weights,
                        int           pad_index,
                        algorithm     aalgorithm,
                        prop_kind     aprop_kind = prop_kind::forward_inference,
                        const engine  &aengine = engine::cpu_engine()) {

        auto emb_cnt = weights.size();
        auto self_md = self.get_desc();
        auto per_sample_weights_md = per_sample_weights.get_desc();

        uint32_t scatter_stride = 1 + emb_cnt;
        for (auto i = 0; i < emb_cnt; ++i) {
            auto weights_md   = weights[i].get_desc();
            auto indices_md   = indices[i].get_desc();
            auto offsets_md   = offsets[i].get_desc();

            uint32_t scatter_offset = i+1;

            auto pdesc = desc(aprop_kind, aalgorithm, ADEEP_EMBED_BAG_THRDS,
                              weights_md, indices_md, offsets_md,
                              per_sample_weights_md, self_md,
                              pad_index, scatter_stride, scatter_offset);

            auto pd         = primitive_desc(pdesc, aengine);

            super(pd).execute(stream::default_stream(),
                              {   {ZENDNN_ARG_SRC_0, weights[i]},
                                  {ZENDNN_ARG_SRC_1, indices[i]},
                                  {ZENDNN_ARG_SRC_2, offsets[i]},
                                  {ZENDNN_ARG_SRC_3, per_sample_weights},
                                  {ZENDNN_ARG_DST, self}
                              });
        }
    }

    static void compute(tensor         &self,
                        tensor_vector  &weights,
                        tensor_vector  &indices,
                        tensor_vector  &offsets,
                        int            pad_index,
                        algorithm      aalgorithm,
                        prop_kind      aprop_kind = prop_kind::forward_inference,
                        const engine   &aengine = engine::cpu_engine()) {


        auto emb_cnt = weights.size();
        auto self_md = self.get_desc();

        uint32_t scatter_stride = 1 + emb_cnt;
        for (auto i = 0; i < emb_cnt; ++i) {
            auto weights_md   = weights[i].get_desc();
            auto indices_md   = indices[i].get_desc();
            auto offsets_md   = offsets[i].get_desc();

            uint32_t scatter_offset = i+1;

            auto pdesc = desc(aprop_kind, aalgorithm, ADEEP_EMBED_BAG_THRDS,
                              weights_md, indices_md, offsets_md, self_md,
                              pad_index, scatter_stride, scatter_offset);

            auto pd         = primitive_desc(pdesc, aengine);

            super(pd).execute(stream::default_stream(),
                              {   {ZENDNN_ARG_SRC_0, weights[i]},
                                  {ZENDNN_ARG_SRC_1, indices[i]},
                                  {ZENDNN_ARG_SRC_2, offsets[i]},
                                  {ZENDNN_ARG_DST, self}
                              });
        }
    }

};

}  // namespace adeep

#endif

