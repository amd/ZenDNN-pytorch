/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>
#include <ATen/native/zendnn/ZENDNNTensors.h>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_zendnn_impl(const Tensor &weight, const Tensor &indices,
                           const Tensor &offsets, int64_t mode,
                           const c10::optional<Tensor> &per_sample_weights,
                           bool include_last_offset, int64_t padding_idx) {
    AT_ERROR("embedding_bag_zendnn_: ATen not compiled with ZENDNN support");
}

}
} //namespace at::native

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>
namespace {
// Helper function for getting an adeep tensor out of an aten Tensor.
// Note in case the aten Tensor is a dense tensor, the returned adeep
// tensor is just a view of the storage of the aten dense tensor, so
// caller needs to make sure the aten dense tensor's lifetime is
// longer than the adeep tensor.
inline adeep::tensor get_zendnn_tensor(const at::Tensor &tensor) {
    if(tensor.is_zendnn()) {
        return at::native::itensor_from_zendnn(tensor);
    } else {
        return at::native::zendnn_tensor_view_from_dense(tensor);
    }
}
}

namespace at {
namespace native {
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_zendnn_impl(const Tensor &weight, const Tensor &indices,
                           const Tensor &offsets, int64_t mode,
                           const c10::optional<Tensor> &per_sample_weights,
                           bool include_last_offset, int64_t padding_idx) {

    Tensor offset2bag  = at::empty({});
    Tensor bag_size    = at::empty({});
    Tensor max_indices = at::empty({});

    // convert pytorch tensors to zendnn tensors
    using Convt = at::native::ZendnnTensorConvert;

    adeep::tensor z_input   = Convt::zentensor_view_dense(weight);
    adeep::tensor z_indices = Convt::zentensor_view_dense(indices);
    adeep::tensor z_offsets = Convt::zentensor_view_dense(offsets);

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

    int dim_embedding      = z_input.get_dim(1);
    int num_bags           = z_offsets.get_dim(0);
    adeep::data_type dtype = z_input.get_data_type();

    // at::empty instead of at::zero is more efficient
    Tensor output = at::empty({num_bags, dim_embedding}, weight.options());
    //Tensor output = at::zeros({num_bags, dim_embedding}, weight.options());

    adeep::tensor z_dst    = Convt::zentensor_view_dense(output);

    if(per_sample_weights.has_value() && per_sample_weights.value().defined()) {
        adeep::tensor z_weights
            = Convt::zentensor_view_dense(per_sample_weights.value());
        adeep::embed_bag::compute(z_input, z_indices, z_offsets,
                                  z_weights, z_dst, padding_idx,
                                  aalgorithm);
    }
    else {
        adeep::embed_bag::compute(z_input, z_indices, z_offsets,
                                  z_dst, padding_idx,
                                  aalgorithm);
    }

    std::tuple<Tensor, Tensor, Tensor, Tensor >  out;
    out = std::make_tuple(std::move(output), std::move(offset2bag),
                          std::move(bag_size), std::move(max_indices));

    return out;
}

}
}  // namespace at::native

#endif

