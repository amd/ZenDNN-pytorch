/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

// tensor conversion routines to convert from pytorch tensors to zendnn
// tensors and vice versa.
// zendnn tensor is not wrapped in oblique pytorch tensor, and shares the
// underlaying storage.

#if AT_ZENDNN_ENABLED()
#include <adeep.hpp>

namespace at {
namespace native {
class ZendnnTensorConvert {
public:
    static adeep::tensor zentensor_view_dense(const Tensor &atensor);
    static Tensor  pytensor_view_dense(const adeep::tensor &ztensor,
                                       const TensorOptions &aoptions);

    static adeep::tensor::data_type zen_dtype(ScalarType atype);
    static ScalarType  py_dtype(adeep::tensor::data_type ztype);
};

}
} // namespace native, at
#endif // AT_ZENDNN_ENABLED()


