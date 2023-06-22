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

#ifndef ADEEP_OPERATORS_EMBED_BAG_HPP
#define ADEEP_OPERATORS_EMBED_BAG_HPP

#define ADEEP_EMBED_BAG_THRDS        (16)

namespace adeep {

struct embed_bag : public zendnn::embedding_bag {

    using super = zendnn::embedding_bag;

    static void compute(const tensor &input,
                        tensor   &indices,
                        tensor   &offsets,
                        tensor   &weights,
                        tensor   &dst,
                        int       pad_index,
                        algorithm aalgorithm,
                        prop_kind aprop_kind = prop_kind::forward_inference,
                        const engine &aengine = engine::cpu_engine()) {

        // get input tesnor memory descriptors
        auto input_md   = input.get_desc();
        auto indices_md = indices.get_desc();
        auto offsets_md = offsets.get_desc();
        auto weights_md = weights.get_desc();
        auto dst_md     = dst.get_desc();

        // declare embedding bag primitive
        auto pdesc      = desc(aprop_kind, aalgorithm, ADEEP_EMBED_BAG_THRDS,
                               input_md, indices_md, offsets_md,
                               weights_md, dst_md, pad_index);
        auto pd         = primitive_desc(pdesc, aengine);

        super(pd).execute(stream::default_stream(),
        {   {ZENDNN_ARG_SRC_0, input},
            {ZENDNN_ARG_SRC_1, indices},
            {ZENDNN_ARG_SRC_2, offsets},
            {ZENDNN_ARG_SRC_3, weights},
            {ZENDNN_ARG_DST, dst}
        });
    }

    static void compute(const tensor &input,
                        tensor   &indices,
                        tensor   &offsets,
                        tensor   &dst,
                        int       pad_index,
                        algorithm aalgorithm,
                        prop_kind aprop_kind = prop_kind::forward_inference,
                        const engine &aengine = engine::cpu_engine()) {

      // get input tesnor memory descriptors
        auto input_md   = input.get_desc();
        auto indices_md = indices.get_desc();
        auto offsets_md = offsets.get_desc();
        auto dst_md     = dst.get_desc();

        // declare embedding bag primitive
        auto pdesc      = desc(aprop_kind, aalgorithm, ADEEP_EMBED_BAG_THRDS,
                               input_md, indices_md, offsets_md,
                               dst_md, pad_index);
        auto pd         = primitive_desc(pdesc, aengine);

        super(pd).execute(stream::default_stream(),
        {   {ZENDNN_ARG_SRC_0, input},
            {ZENDNN_ARG_SRC_1, indices},
            {ZENDNN_ARG_SRC_2, offsets},
            {ZENDNN_ARG_DST, dst}
        });
    }

};

}  // namespace adeep

#endif

