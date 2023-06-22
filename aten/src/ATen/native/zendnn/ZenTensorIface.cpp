/******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <ATen/native/zendnn/ZenTensorIface.hpp>

#if AT_ZENDNN_ENABLED()
namespace zendnn {

using namespace at;
using namespace c10;

// macro to convert a pytorch tensor to zendnn tensor
#define  GET_ZENDNN_TENSOR(atensor_, atype_, ztype_)                       \
    {                                                                      \
      using cpptype = decltype(c10::impl::ScalarTypeToCPPType<atype_>::t); \
      return ZenTensorType{{atensor_.sizes().cbegin(),                     \
                              atensor_.sizes().cend()},                    \
                              ztype_,                                      \
                              atensor_.template data_ptr<cpptype>()};      \
    }

// implementation
ZenDType
ZenTensorIface::to_zen_dtype(ATDType atype) {

    switch(atype) {
    case ATDType::Float:
        return ZenDType::f32;

    case ATDType::QInt32:
    case ATDType::Int:
        return ZenDType::s32;

    case ATDType::BFloat16:
        return ZenDType::bf16;

    case ATDType::QUInt8:
    case ATDType::Byte:
        return ZenDType::u8;

    case ATDType::QInt8:
    case ATDType::Char:
        return ZenDType::s8;

    default:
        TORCH_CHECK(false, "ZenTensorIface:to_zen_dtype: unsupported data type");
    }

    // should never reach here.
    return ZenDType::undef;
}

ATDType
ZenTensorIface::to_at_dtype(ZenDType ztype) {
    switch(ztype) {
    case ZenDType::f32:
        return ATDType::Float;
    case ZenDType::s32:
        return ATDType::Int;
    case ZenDType::u8:
        return ATDType::Byte;
    case ZenDType::bf16:
        return ATDType::BFloat16;
    case ZenDType::s8:
        return ATDType::Char;
    default:
        TORCH_CHECK(false, "ZendnnTensorConvert:py_dtype: unsupported data type");
    }

    // should never reach here.
    return ATDType::Undefined;
}

ZenTensorType
ZenTensorIface::zentensor_view_dense(const ATTensorType &atensor) {
    // sanity check on input
    TORCH_CHECK(atensor.device().type() == DeviceType::CPU,
                "zentensor_view_dense expects CPU tensor input");
    TORCH_CHECK(atensor.layout() == Layout::Strided,
                "zentensor_view_dense expects dense tensor input");

    // check if tensor type is supported in zendnn
    auto atensor_arg = at::TensorArg(atensor, "atensor", 1);
    checkScalarTypes("zentensor_view_dense", atensor_arg,
                     {kByte, kChar, kInt, kFloat, kBFloat16});

    // get corrresponding zendnn tensor
    auto atype     = atensor.scalar_type();

    switch(atype) {
    case kByte:
        GET_ZENDNN_TENSOR(atensor, kByte, ZenDType::u8)
    case kChar:
        GET_ZENDNN_TENSOR(atensor, kChar, ZenDType::s8)
    case kInt:
        GET_ZENDNN_TENSOR(atensor, kInt, ZenDType::s32)
    case kFloat:
        GET_ZENDNN_TENSOR(atensor, kFloat, ZenDType::f32)
    case kBFloat16:
        GET_ZENDNN_TENSOR(atensor, kBFloat16, ZenDType::bf16)
    default:
        TORCH_CHECK(false,"ZenTensorIFace:zentensor_view: unsupported data type");
    }

    // should never reach here
    GET_ZENDNN_TENSOR(atensor, kFloat, ZenDType::f32);
}

ATTensorType
ZenTensorIface::to_at_tensor(const ZenTensorType  &ztensor,
                             const at::TensorOptions &aoptions) {
    // get data_type of zendnn_tensor and figure out appropriate
    // pytorch ScalarType
    auto ztype   = ztensor.get_data_type();
    auto atype   = to_at_dtype(ztype);

    // allocate empty tensor
    auto dims = ztensor.get_dims();

    // NOTE: int32_t dims from adeep::tensor but sizes needs int64_t
    Tensor cpu_tensor = at::empty(
                            std::vector<int64_t>(dims.begin(), dims.end()),
                            aoptions.layout(c10::kStrided).dtype(atype));

    // if input zendnn tensor is empty, return empty tensor
    if(ztensor.is_empty()) return cpu_tensor;

    auto pub_tensor = ztensor.to_public(cpu_tensor.data_ptr(), ztype);
    cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
    return cpu_tensor;
}

ZenTensorVecType
ZenTensorIface::zentensor_view_dense(c10::ArrayRef<ATTensorType> attensor_vec) {
    ZenTensorVecType out;

    for(auto i = 0; i < attensor_vec.size(); ++i)
        out.push_back(zentensor_view_dense(attensor_vec[i]));

    return out;
}

ATTensorVecType
ZenTensorIface::to_at_tensor(ZenTensorVecType &zentensor_vec,
                             const at::TensorOptions &aoptions) {
    ATTensorVecType out;

    for (auto i = 0; i < zentensor_vec.size(); ++i)
        out.push_back(to_at_tensor(zentensor_vec[i], aoptions));

    return out;
}

} //namespace zendnn

#endif
