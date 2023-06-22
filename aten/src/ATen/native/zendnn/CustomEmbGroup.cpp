/******************************************************************************
* Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <ATen/native/zendnn/ZenTensorIface.hpp>

#if AT_ZENDNN_ENABLED()

namespace zendnn{
using namespace at;
Tensor& embag_grp_cpu(Tensor&                       at_self,
                      ArrayRef<Tensor>              at_weights,
                      Tensor&                       at_indices,
                      Tensor&                       at_offsets,
                      int64_t                       mode,
                      int64_t                       padding_idx,
                      bool                          include_last_offset,
                      const c10::optional<Tensor>&  at_per_sample_weights){


    // sanity check on input dimensions
    auto emb_cnt_wt = at_weights.size();
    auto emb_cnt_in = at_indices.size(0);
    auto emb_cnt_of = at_offsets.size(0);
    TORCH_CHECK(((emb_cnt_wt == emb_cnt_in) && (emb_cnt_in == emb_cnt_of)),
                "embedding bag counts mismatch");

    auto bs_in    = at_indices.size(1);
    auto words_sl = at_self.size(0);
    auto words_bs = (1 + emb_cnt_wt)*bs_in;
    TORCH_CHECK((words_sl == words_bs),"output length ", words_sl,
                "is expected to be ", words_bs);

    auto m_spa_wt = at_weights[0].size(1);
    auto m_spa_sl = at_self.size(1);

    TORCH_CHECK((m_spa_wt == m_spa_sl), "output width {", m_spa_sl ,"} and weights width {",
                m_spa_wt, "} mismatch");

    // sanity check on types
    auto indices_arg = TensorArg(at_indices, "at_indices", 1);
    checkScalarTypes("embag_grp_cpu", indices_arg, {kLong, kInt});
    auto offsets_arg = TensorArg(at_offsets, "at_offsets", 1);
    checkScalarTypes("embag_grp_cpu", offsets_arg, {kLong, kInt});
    checkSameType("embag_grp_cpu", indices_arg, offsets_arg);

    for (auto i = 0; i < emb_cnt_in; ++i) {
        auto weight_arg = TensorArg(at_weights[i], "weight", 1);
        checkScalarTypes("embag_grp_cpu", weight_arg, {kFloat, kDouble});
    }

    // somehow dtype conversion is not working out in python
    Tensor  at_indices_lp    = at_indices.toType(kInt);
    Tensor  at_offsets_lp    = at_offsets.toType(kInt);

    //convert to adeep tensors
    ZenTensorType self       = ZenTensorIface::zentensor_view_dense(at_self);
    ZenTensorVecType weights = ZenTensorIface::zentensor_view_dense(at_weights);

    ZenTensorVecType indices;
    ZenTensorVecType offsets;
    for (auto i = 0; i < emb_cnt_in; ++i) {
        Tensor at_indices_slice     = at_indices_lp[i];
        ZenTensorType indices_slice = ZenTensorIface::zentensor_view_dense(at_indices_slice);
        indices.push_back(indices_slice);

        Tensor at_offsets_slice     = at_offsets_lp[i];
        ZenTensorType offsets_slice = ZenTensorIface::zentensor_view_dense(at_offsets_slice);
        offsets.push_back(offsets_slice);
    }

    // figure out the mode
    adeep::algorithm aalgorithm;
    switch(mode) {
    case 0 :
        aalgorithm = adeep::algorithm::embedding_bag_sum;
        break;
    case 1 :
        aalgorithm = adeep::algorithm::embedding_bag_mean;
        break;
    case 2 :
        aalgorithm = adeep::algorithm::embedding_bag_max;
        break;
    default:
        aalgorithm = adeep::algorithm::embedding_bag_sum;
        break;
    }

    // call appropriate adeep routine
    if(at_per_sample_weights.has_value() && at_per_sample_weights.value().defined()) {
        ZenTensorType per_sample_weights
            = ZenTensorIface::zentensor_view_dense(at_per_sample_weights.value());
        adeep::embed_bag_group::compute(self, weights, indices, offsets,
                                        per_sample_weights, padding_idx, aalgorithm);
    }
    else {
        adeep::embed_bag_group::compute(self, weights, indices, offsets,
                                        padding_idx, aalgorithm);
    }

    return at_self;
}
} //namespace zendnn

TORCH_LIBRARY_FRAGMENT(zendnn, m){
  m.def("embag_grp(Tensor(a!) self, Tensor[] weights, Tensor indices, Tensor offsets, int mode, int padding_idx, bool include_last_offset, Tensor? at_per_sample_weights=None)->Tensor(a!)");
}

TORCH_LIBRARY_IMPL(zendnn, ZendnnCPU, m) {
  m.impl("embag_grp", zendnn::embag_grp_cpu);
}

TORCH_LIBRARY_IMPL(zendnn, CPU, m) {
  m.impl("embag_grp", zendnn::embag_grp_cpu);
}

#endif
