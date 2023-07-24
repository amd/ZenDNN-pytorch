/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/zendnn/Utils.h>

namespace at { namespace native {

#if AT_ZENDNN_ENABLED()

Tensor zendnn_to_dense(const Tensor& zendnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(zendnn_tensor.scalar_type() == ScalarType::Float || zendnn_tensor.scalar_type() == ScalarType::BFloat16,
              "zendnn_to_dense expects float or bfloat16 tensor input");

  adeep::tensor& stensor = itensor_from_zendnn(zendnn_tensor);
  auto dims = stensor.get_dims();
  auto data_type = dtype.has_value() ? dtype.value() : zendnn_tensor.scalar_type();
  TORCH_CHECK(data_type == ScalarType::Float || data_type == ScalarType::BFloat16,
            "zendnn tensor only can be converted to be  float or bfloat16 cpu tensor")
  // NOTE: int32_t dims from adeep::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::empty(
    std::vector<int64_t>(dims.begin(), dims.end()),
    zendnn_tensor.options().layout(c10::kStrided).dtype(data_type));
  if (stensor.is_empty()) return cpu_tensor;
  auto pub_tensor =
      data_type == ScalarType::Float
      ? stensor.to_public(cpu_tensor.template data_ptr<float>(),
                          adeep::tensor::data_type::f32)
      : stensor.to_public(cpu_tensor.template data_ptr<BFloat16>(),
                         adeep::tensor::data_type::bf16);
  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  return cpu_tensor;
}

Tensor dense_to_zendnn(const Tensor& cpu_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(),
             "dense_to_zendnn expects CPU tensor input");
  TORCH_CHECK(cpu_tensor.layout() == Layout::Strided,
             "dense_to_zendnn expects strided tensor input");
  TORCH_CHECK(cpu_tensor.scalar_type() == ScalarType::Float || cpu_tensor.scalar_type() == ScalarType::BFloat16,
             "dense_to_zendnn expects float or bfloat16 tensor input");
  TORCH_CHECK(cpu_tensor.dim() <= 5,
             "Can't convert cpu tensor with the number of dimensions > 5");
  // TODO: consider to convert non-contiguous tensor to `adeep::tensor` directly.
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  auto data_type = dtype.has_value() ? dtype.value() : cpu_tensor.scalar_type();
  TORCH_CHECK(data_type == ScalarType::Float || data_type == ScalarType::BFloat16,
              "cpu tensor only can be converted to be a float or bfloat16 zendnn tensor")
  Tensor zendnn_tensor = empty_zendnn(cpu_tensor_cont.sizes(), data_type,
                                      cpu_tensor_cont.options().layout_opt(), cpu_tensor_cont.options().device_opt(),
                                      cpu_tensor_cont.options().pinned_memory_opt());
  adeep::tensor& dtensor = itensor_from_zendnn(zendnn_tensor);
  if (cpu_tensor.scalar_type() == ScalarType::Float) {
    dtensor.feed_from(dtensor.get_dims(),
                      adeep::tensor::data_type::f32,
                      (cpu_tensor_cont.template data_ptr<float>()));
  } else {
    dtensor.feed_from(dtensor.get_dims(),
                      adeep::tensor::data_type::bf16,
                      cpu_tensor_cont.template data_ptr<BFloat16>());
  }
  return zendnn_tensor;
}

std::vector<at::Tensor> zendnn_to_dense_list( TensorList zendnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(zendnn_tensor[0].scalar_type() == ScalarType::Float,
              "zendnn_to_dense_list expects float tensor input");
  auto data_type = dtype.has_value() ? dtype.value() : zendnn_tensor[0].scalar_type();

  TORCH_CHECK(data_type == ScalarType::Float,
            "zendnn tensor only can be converted to be a float cpu tensor")
  std::vector<at::Tensor> list_tensor;
  for (size_t i = 0; i < zendnn_tensor.size(); i++) {
     Tensor tensor = zendnn_to_dense(zendnn_tensor[i]);
     list_tensor.push_back(tensor);
  }
  return list_tensor;
}
// zendnn tensor has special non-public format for conv2d weights
// (dense_to_zendnn only converts dense tensor to zendnn tensor with
// public format). Ideep conv kernel will do implicit reorder if the
// weight is not already in this optimized format. By the time I'm
// writing this note, we are seeing ~20% perf cost of doing the
// on-the-fly reorder.
Tensor zendnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(zendnn_bf16_device_check(),
        "zendnn_reorder_conv2d_weight: bf16 path needs the cpu support avx512bf16");
  }

  auto w = itensor_from_zendnn(self);

  // Legacy zendnn conv2d jitted module may contain a 5-d weight with an extra
  // dimension when groups > 1, having dimension [g, o/g, i, h, w] instead of
  // [o, i, h, w]. Ideally we should reorder the weight back in serialization.
  // For backward compatibility, we squash the first two dims (g * o/g) back to
  // its original form.
  if (w.ndims() == 5) {
    auto wdims = w.get_dims();
    w.reshape({wdims[0] * wdims[1], wdims[2], wdims[3], wdims[4]});
  }

  auto desc =
      adeep::convolution_forward::expected_weights_desc(
          w.get_dims(),
          w.get_data_type(),
          {stride.begin(), stride.end()},
          {padding.begin(), padding.end()},
          {padding.begin(), padding.end()},
          {dilation.begin(), dilation.end()},
          groups,
          adeep::algorithm::convolution_direct);
  adeep::tensor result;
  result.init(desc);
  result.feed_from(w);

  return new_with_itensor_zendnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor zendnn_reorder_conv3d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(false,
        "zendnn_reorder_conv3d_weight: bf16 path is not supported in zendnn yet");
  }

  auto w = itensor_from_zendnn(self);

  auto desc =
      adeep::convolution_forward::expected_weights_desc(
          w.get_dims(),
          w.get_data_type(),
          {stride.begin(), stride.end()},
          {padding.begin(), padding.end()},
          {padding.begin(), padding.end()},
          {dilation.begin(), dilation.end()},
          groups,
          adeep::algorithm::convolution_direct);
  adeep::tensor result;
  result.init(desc);
  result.feed_from(w);

  return new_with_itensor_zendnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()), self.options().device_opt());
}


#else

Tensor zendnn_to_dense(const Tensor& zendnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "ZENDNN build is disabled");
}

std::vector<at::Tensor> zendnn_to_dense_list( TensorList zendnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "zendnn_to_dense_list: ZENDNN build is disabled");
}

Tensor dense_to_zendnn(const Tensor& cpu_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "ZENDNN build is disabled");
}
Tensor zendnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(false, "zendnn_reorder_conv2d_weight: ZENDNN build is disabled");
}

Tensor zendnn_reorder_conv3d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(false, "zendnn_reorder_conv3d_weight: ZENDNN build is disabled");
}
#endif // AT_ZENDNN_ENABLED()


#if AT_ZENDNN_QUANT_ENABLED()

Tensor zendnn_reorder_conv2d_vitisai_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    int64_t filter_scale,
    bool d_flag) {

  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(false,
        "zendnn_reorder_conv2d_weight: bf16 path is not supported in zendnn yet");
  }
  adeep::dims padding_l, padding_r ,strides, dilates;
  adeep::dims x_dims, y_dims, kernel_size1;
  adeep::scale_t f_scale;
  adeep::attr_t weights_attr;
  float filter_scale_pow_2;

  adeep::data_type dtype_s8 = adeep::data_type::s8;
  adeep::data_type dtype_u8 = adeep::data_type::u8;

  adeep::algorithm aalgorithm = adeep::algorithm::convolution_direct;
  adeep::prop_kind aprop_kind = adeep::prop_kind::forward;
  const adeep::engine& aengine = adeep::engine::cpu_engine();

  auto w = itensor_from_zendnn(self);

  filter_scale_pow_2 = std::pow(2, filter_scale);
  f_scale = {filter_scale_pow_2};

  weights_attr = {adeep::utils::tensor_scale_mask(1, groups > 1), f_scale};
  auto weights_dims  = w.get_dims();

  auto src_size = weights_dims.size(); // weights_dims is 4 for conv2d and 5 for conv3d
  auto grouped = groups > 1;
  auto weights_dims_g =
      grouped ? adeep::utils::group_dims(weights_dims, groups) : weights_dims;
  auto weights_desc = adeep::tensor::desc(weights_dims_g, dtype_s8);
  auto dims_in = weights_desc.get_dims();
  auto ndims = dims_in.size();

  padding_l =  {padding.begin(), padding.end()};
  padding_r =  {padding.begin(), padding.end()};
  strides   =  {stride.begin(), stride.end()};
  dilates   =  {dilation.begin(), dilation.end()};

  auto dilates_ = adeep::utils::get_compatible_dilates(dilates);
  auto ic = groups * dims_in[1 + grouped];
  auto oc = groups * dims_in[0 + grouped];
  if (5 == src_size) {
    kernel_size1.push_back(dims_in[ndims - 3]);
  }

  kernel_size1.push_back(dims_in[ndims - 2]);
  kernel_size1.push_back(dims_in[ndims - 1]);

  // Construct a dummy case
  x_dims.push_back(1);
  x_dims.push_back(ic);
  y_dims.push_back(1);
  y_dims.push_back(oc);
  if (4 == src_size) {
    x_dims.push_back(2 * kernel_size1[0]);
    x_dims.push_back(4 * kernel_size1[1]);
  } else {
    x_dims.push_back(2 * kernel_size1[0]);
    x_dims.push_back(4 * kernel_size1[1]);
    x_dims.push_back(8 * kernel_size1[2]);
  }

  for (auto d = 2; d < src_size; ++d) {
    auto out_size = (x_dims[d] - ((kernel_size1[d-2] - 1) * (dilates_[d-2] + 1) + 1)
        + (padding_l[d-2] + padding_r[d-2])) / strides[d-2] + 1;
    y_dims.push_back(out_size);
  }

  auto x_dtype = dtype_s8;
  if(d_flag)
  {
    x_dtype= dtype_u8;
  }
  adeep::tensor::desc src_desc(x_dims, x_dtype);
  adeep::tensor::desc dst_desc(y_dims, dtype_u8);

  auto src_desc_query = src_desc;
  auto weights_desc_query = weights_desc;
  auto dst_desc_query = dst_desc;
  auto bias_desc_query =  adeep::tensor::desc();

  src_desc_query = src_desc.to_format_any();
  weights_desc_query = weights_desc.to_format_any();
  bias_desc_query =  adeep::tensor::desc();
  dst_desc_query = dst_desc.to_format_any();


  //Disabling primitive caching while reorder the weights at opt_fot_inference call
  //it causes primitive conflicts when paramenters are same
  adeep::convolution_forward::primitive_desc pd;
  pd =  adeep::convolution_forward::primitive_desc({aprop_kind, aalgorithm, src_desc_query,
                          weights_desc_query, dst_desc_query,
                          strides, dilates_, padding_l, padding_r},
                          weights_attr, aengine);

  auto desc = adeep::tensor::desc(pd.weights_desc(), groups);
  adeep::tensor zendnn_weight1 = w.make_grouped_weights(groups).reorder_if_differ_in(desc, weights_attr);
  return new_with_itensor_zendnn(std::move(zendnn_weight1), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

#else // AT_ZENDNN_QUANT_ENABLED()

Tensor zendnn_reorder_conv2d_vitisai_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    int64_t filter_scale,
    bool d_flag)
    {
  TORCH_CHECK(false, "zendnn_reorder_conv2d_vitisai_weight: ZENDNN build is disabled");
}

#endif // AT_ZENDNN_QUANT_ENABLED()

}}
