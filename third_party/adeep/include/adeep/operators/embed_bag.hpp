/*******************************************************************************
* Modifications Copyright (c) 2022-2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

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

