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

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Allocator.h>

#if AT_ZENDNN_ENABLED()

#include <adeep.hpp>

namespace at { namespace native {

/**
 * `IntrusivePtrTargetWrapper` wraps a custom storage handle  of a tensor
*  (as template param) and inherits `c10::intrusive_ptr_target` so that it
*  can be used with `c10::intrusive_ptr`.
 *
 * It currently only supports wrapping the custom handle by:
 * - Constructing with an existing custom handle by copy/move constructor.
 *
 * See `OpaqueTensorImpl::opaque_handle_`.
 *
 * NOTE: if this is generally useful we may want to move this to its own header.
 */
template <typename T>
struct TORCH_API IntrusivePtrTargetWrapper : c10::intrusive_ptr_target {
private:
  T target_;

public:
  IntrusivePtrTargetWrapper() = delete;
  IntrusivePtrTargetWrapper(const T& target): target_(target) {}
  IntrusivePtrTargetWrapper(T&& target): target_(std::move(target)) {}

  T& get_target() {
    return target_;
  }
};

using IDeepTensorWrapper = IntrusivePtrTargetWrapper<adeep::tensor>;
using IDeepTensorWrapperPtr = c10::intrusive_ptr<IDeepTensorWrapper>;
using ZENDNNTensorImpl = OpaqueTensorImpl<IDeepTensorWrapperPtr>;
using ZENDNNTensor = Tensor;

// conversion from pytorch type to zendnn type
adeep::tensor::data_type get_zendnn_dtype(ScalarType type) {
  switch (type) {
    case ScalarType::Float:
      return adeep::tensor::data_type::f32;
    case ScalarType::QInt32:
      return adeep::tensor::data_type::s32;
    case ScalarType::QInt8:
      return adeep::tensor::data_type::s8;
    case ScalarType::QUInt8:
    case ScalarType::Byte:
      return adeep::tensor::data_type::u8;
    case ScalarType::BFloat16:
      return adeep::tensor::data_type::bf16;
    case ScalarType::Char:
      return adeep::tensor::data_type::s8;
    case ScalarType::Int:
      return adeep::tensor::data_type::s32;
    default:
      TORCH_CHECK(false, "get_zendnn_dtype: unsupported data type");
  }
}

// conversion from zendnn type to pytorch type
ScalarType get_tensor_dtype(adeep::tensor::data_type type) {
  switch (type) {
    case adeep::tensor::data_type::f32:
      return ScalarType::Float;
    case adeep::tensor::data_type::s32:
      return ScalarType::Int;
    case adeep::tensor::data_type::u8:
      return ScalarType::Byte;
    case adeep::tensor::data_type::bf16:
      return ScalarType::BFloat16;
    case adeep::tensor::data_type::s8:
      return ScalarType::Char;
    default:
      TORCH_CHECK(false, "get_tensor_dtype: unsupported data type");
  }
}

Tensor new_with_itensor_zendnn(adeep::tensor&& it, c10::optional<ScalarType> dtype, c10::optional<Device> device) {
  // NOTE: int32_t dims from adeep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in adeep::tensor to avoid extra conversion
  auto dims = it.get_dims();
  IDeepTensorWrapperPtr handle = c10::make_intrusive<IDeepTensorWrapper>(std::move(it));
  caffe2::TypeMeta dtype_ = scalarTypeToTypeMeta(dtype_or_default(dtype));
  Device device_ = device_or_default(device);
  return detail::make_tensor<ZENDNNTensorImpl>(
    DispatchKeySet(DispatchKey::ZendnnCPU),
    dtype_, device_, handle,
    std::vector<int64_t>(dims.begin(), dims.end()));
}

adeep::tensor& itensor_from_zendnn(const ZENDNNTensor& zendnn_tensor) {
  TORCH_CHECK(zendnn_tensor.is_zendnn(),
             "itensor_from_zendnn expects ZENDNN tensor input");
  TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());
  ZENDNNTensorImpl *zendnnimpl = static_cast<ZENDNNTensorImpl *>(zendnn_tensor.unsafeGetTensorImpl());
  return zendnnimpl->unsafe_opaque_handle()->get_target();
}

adeep::tensor itensor_view_from_dense(const Tensor& tensor) {
  TORCH_CHECK(
      tensor.device().is_cpu(),
      "itensor_view_from_dense expects CPU tensor input");
  TORCH_CHECK(
      tensor.layout() == Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  TORCH_CHECK(tensor.scalar_type() == ScalarType::Float || tensor.scalar_type() == ScalarType::BFloat16,
             "itensor_view_from_dense expects float or bfloat16 tensor input");
  TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());

  adeep::tensor atensor;
  //Providing stride information while initializing the tensor.
  //Otherwise, tensor data will be read in coloumn major format.
  if(tensor.scalar_type() == ScalarType::BFloat16)
  {
    atensor =  {{tensor.sizes().vec(),
            adeep::tensor::data_type::bf16,
            tensor.strides().vec()},
            tensor.template data_ptr<BFloat16>()};
  }
  else
  {
    atensor = {{tensor.sizes().vec(),
            adeep::tensor::data_type::f32,
            tensor.strides().vec()},
            tensor.template data_ptr<float>()};
  }
  return atensor;
}

// Helper function for getting an adeep tensor out of an aten Tensor.
// Note in case the aten Tensor is a dense tensor, the returned adeep
// tensor is just a view of the storage of the aten dense tensor, so
// caller needs to make sure the aten dense tensor's lifetime is
// longer than the adeep tensor.
adeep::tensor itensor_from_tensor(const Tensor& tensor) {
  if (tensor.is_zendnn()) {
    return itensor_from_zendnn(tensor);
  } else {
    return itensor_view_from_dense(tensor);
  }
}

// functions to get a zendnn tensor from a Tensor and vice versa.
adeep::tensor zendnn_tensor_view_from_dense(const Tensor& ttensor){
  // sanity check on input
  TORCH_CHECK( ttensor.device().type() == DeviceType::CPU,
               "zendnn_tensor_view_from_dense expects CPU tensor input");
  TORCH_CHECK( ttensor.layout() == Layout::Strided,
               "zendnn_tensor_view_from_dense expects dense tensor input");

  // check if tensor type is supported in zendnn
  auto ttensor_arg = TensorArg(ttensor, "ttensor", 1);
  checkScalarTypes("zendnn_tensor_view_from_dense", ttensor_arg,
                   {kByte, kChar, kInt, kLong, kFloat});

  // get c++ type corresponding to ScalarType
  // TODO : remove switch statment from here and make a function/macro
  // for type conversion between torch and zendnn tensor.
  auto dtype     = ttensor.scalar_type();

  switch (dtype)
    {
    case kByte:
      {
        auto  atype   = adeep::tensor::data_type::u8;
        using cpptype = decltype(c10::impl::ScalarTypeToCPPType<kByte>::t);
        return adeep::tensor{{ttensor.sizes().cbegin(),
                              ttensor.sizes().cend()},
                             atype,
                             ttensor.template data_ptr<cpptype>()};
      }
    case kChar:
      {
        auto  atype   =  adeep::tensor::data_type::s8;
        using cpptype = decltype(c10::impl::ScalarTypeToCPPType<kChar>::t);
        return adeep::tensor{{ttensor.sizes().cbegin(),
                              ttensor.sizes().cend()},
                             atype,
                             ttensor.template data_ptr<cpptype>()};
      }
    case kLong:
    case kInt:
      {
        auto atype    = adeep::tensor::data_type::s32;
        using cpptype = decltype(c10::impl::ScalarTypeToCPPType<kInt>::t);
        return adeep::tensor{{ttensor.sizes().cbegin(),
                              ttensor.sizes().cend()},
                             atype,
                             ttensor.template data_ptr<cpptype>()};
      }
    case kFloat:
      {
        auto  atype   =  adeep::tensor::data_type::f32;
        using cpptype = decltype(c10::impl::ScalarTypeToCPPType<kFloat>::t);
        return adeep::tensor{{ttensor.sizes().cbegin(),
                              ttensor.sizes().cend()},
                             atype,
                             ttensor.template data_ptr<cpptype>()};
      }
    default:
      TORCH_CHECK(false,"zendnn_tensor_view_from_dense: unsupported data type");
    }

  // default is always float
  auto   atype     = adeep::tensor::data_type::f32;
  using  cpptype   = float;

  return adeep::tensor{{ttensor.sizes().cbegin(),
                        ttensor.sizes().cend()},
                        atype,
                        ttensor.template data_ptr<cpptype>()};
}

Tensor new_dense_from_zendnn(const adeep::tensor& zendnn_tensor,
                             const TensorOptions& options) {

  // get data_type of zendnn_tensor and figure out appropriate
  // pytorch ScalarType
  using adeep_type          = adeep::tensor::data_type;

  auto zendnn_tensor_type   = zendnn_tensor.get_data_type();
  ScalarType  tensor_type   = get_tensor_dtype(zendnn_tensor_type);

  // allocate empty tensor
  auto dims = zendnn_tensor.get_dims();

  // NOTE: int32_t dims from adeep::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::empty(
                        std::vector<int64_t>(dims.begin(), dims.end()),
                        options.layout(c10::kStrided).dtype(tensor_type));

  // if input zendnn tensor is empty, return empty tensor
  if (zendnn_tensor.is_empty()) return cpu_tensor;

  auto pub_tensor
    = zendnn_tensor.to_public(cpu_tensor.data_ptr(),
                              zendnn_tensor_type);
  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  return cpu_tensor;
}

}}

#endif // AT_ZENDNN_ENABLED()
