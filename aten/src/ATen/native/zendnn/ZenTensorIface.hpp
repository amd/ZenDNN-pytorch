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

#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_ZENDNN_ENABLED()
#include <adeep.hpp>

// ZenTensorIFace is a class that provides an interface for conversion of torch
// tensors to zendnn tensors and vice versa.
// torch tensors use an intrusive_ptr to keep track of reference count. however
// this reference count is not maintained once the tensor is passed to zendnn.
// it is assumed that no native torch routine will access this tensor till
// zendnn returns it back.

namespace zendnn {
using  ZenTensorType    = adeep::tensor;
using  ATTensorType     = at::Tensor;
using  ZenTensorVecType = std::vector<ZenTensorType>;
using  ATTensorVecType  = std::vector<ATTensorType>;
using  ZenDType         = adeep::tensor::data_type;
using  ATDType          = c10::ScalarType;

class ZenTensorIface {
public:
    static ZenDType         to_zen_dtype(ATDType atype);
    static ATDType          to_at_dtype(ZenDType ztype);

    static ZenTensorType    zentensor_view_dense(const ATTensorType &atensor);
    static ATTensorType     to_at_tensor(const ZenTensorType &ztensor,
                                        const at::TensorOptions &aoptions);

    static ZenTensorVecType zentensor_view_dense(c10::ArrayRef<ATTensorType> attensor_vec);
    static ATTensorVecType  to_at_tensor(ZenTensorVecType &zentensor_vec,
                                         const at::TensorOptions &aoptions);
};

} //namespace zendnn

#endif
