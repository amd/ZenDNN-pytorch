/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/native/zendnn/ZENDNNTensors.h>

#if AT_ZENDNN_ENABLED()

namespace at {
namespace native {

// macro to convert a pytorch tensor to zendnn tensor
#define  GET_ZENDNN_TENSOR(atensor_, atype_, ztype_)                       \
    {                                                                      \
      using cpptype = decltype(c10::impl::ScalarTypeToCPPType<atype_>::t); \
      return adeep::tensor{{atensor_.sizes().cbegin(),                     \
                            atensor_.sizes().cend()},                      \
                           ztype_,                                         \
                           atensor_.template data_ptr<cpptype>()};         \
    }

// implementation
adeep::tensor
ZendnnTensorConvert::zentensor_view_dense(const Tensor &atensor) {
    // sanity check on input
    TORCH_CHECK(atensor.device().type() == DeviceType::CPU,
                "zendnn_tensor_view_from_dense expects CPU tensor input");
    TORCH_CHECK(atensor.layout() == Layout::Strided,
                "zendnn_tensor_view_from_dense expects dense tensor input");

    // check if tensor type is supported in zendnn
    auto atensor_arg = TensorArg(atensor, "atensor", 1);
    checkScalarTypes("zendnn_tensor_view_from_dense", atensor_arg,
    {kByte, kChar, kInt, kFloat});

    // get corrresponding zendnn tensor
    using ZenDType = adeep::tensor::data_type;
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
    default:
        TORCH_CHECK(false,"ZendnnTensorConvert:zentensor_view_dense: unsupported data type");
    }

    // should never reach here
    GET_ZENDNN_TENSOR(atensor, kFloat, ZenDType::f32);
}

Tensor
ZendnnTensorConvert::pytensor_view_dense(const adeep::tensor &ztensor,
        const TensorOptions &aoptions) {
    // get data_type of zendnn_tensor and figure out appropriate
    // pytorch ScalarType
    using ZenDType = adeep::tensor::data_type;

    auto ztype   = ztensor.get_data_type();
    auto atype   = py_dtype(ztype);

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

adeep::tensor::data_type
ZendnnTensorConvert::zen_dtype(ScalarType atype) {

    using ZenDType = adeep::tensor::data_type;

    switch(atype) {
    case ScalarType::Float:
        return ZenDType::f32;

    case ScalarType::QInt32:
    case ScalarType::Int:
        return ZenDType::s32;

    case ScalarType::BFloat16:
        return ZenDType::bf16;

    case ScalarType::QUInt8:
    case ScalarType::Byte:
        return ZenDType::u8;

    case ScalarType::QInt8:
    case ScalarType::Char:
        return ZenDType::s8;

    default:
        TORCH_CHECK(false, "ZendnnTensorConvert:zen_type: unsupported data type");
    }

    // should never reach here.
    return adeep::tensor::data_type::undef;
}

ScalarType
ZendnnTensorConvert::py_dtype(adeep::tensor::data_type ztype) {
    using ZenDType = adeep::tensor::data_type;
    switch(ztype) {
    case ZenDType::f32:
        return ScalarType::Float;
    case ZenDType::s32:
        return ScalarType::Int;
    case ZenDType::u8:
        return ScalarType::Byte;
    case ZenDType::bf16:
        return ScalarType::BFloat16;
    case ZenDType::s8:
        return ScalarType::Char;
    default:
        TORCH_CHECK(false, "ZendnnTensorConvert:py_dtype: unsupported data type");
    }

    // should never reach here.
    return ScalarType::Undefined;
}

}
} // native, at

#endif // AT_ZENDNN_ENABLED()

