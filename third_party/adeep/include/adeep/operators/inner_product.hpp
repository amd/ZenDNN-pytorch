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

#ifndef ADEEP_OPERATORS_INNER_PRODUCT_HPP
#define ADEEP_OPERATORS_INNER_PRODUCT_HPP

namespace adeep {

struct inner_product_forward : public zendnn::inner_product_forward {

  using super = zendnn::inner_product_forward;

  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(src, weights, bias, dst, src_scales,
                                     weights_scales, dst_scales, attr,
                                     aprop_kind, alowp_kind, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& weights,
                      tensor& dst,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(src, weights, dummy_bias, dst, src_scales,
                                      weights_scales, dst_scales, attr,
                                      aprop_kind, alowp_kind, aengine);
  }

  static void zen_vitis_compute(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      bool dequant,
      bool fuse_relu,
      float input_scale,
      float filter_scale,
      float bias_scale,
      float requantize_scale,
      float dequantize_scale,
      const prop_kind aprop_kind = prop_kind::forward_inference,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {

    const scale_t& src_scales = {input_scale};
    const scale_t& weights_scales = {filter_scale};
    const scale_t& b_scale = {bias_scale};
    const scale_t& r_scale = {requantize_scale};
    const scale_t& d_scale = {dequantize_scale};

    tensor::desc src_desc, weights_desc, bias_desc;    // Descriptors for source, weights, and bias tensors
    attr_t src_attr, weights_attr, bias_attr, op_attr;           // Attrs for source, weights, bias, and output

    // Set number of scales
    int scale_size = 1;

    if (src.get_data_type() == data_type::f32) {
      src_desc = {src.get_dims(), data_type::s8, tag::any};
      src_attr = {0, src_scales};
    } else {
      src_desc = {src.get_dims(), src.get_data_type(), tag::any};
    }

    weights_desc = {weights.get_dims(), data_type::s8, tag::any};
    if (weights.get_data_type() == data_type::f32) {
      weights_attr = {utils::tensor_scale_mask(scale_size, false),
                      weights_scales};
    }

    bias_desc = {bias.get_dims(), data_type::s32, format_tag::any};
    if (bias.get_data_type() == data_type::f32) {
      bias_attr = {utils::tensor_scale_mask(scale_size, false),
                    b_scale};
    }

    post_ops post_ops;
    if (fuse_relu){
      post_ops.append_eltwise(1.0f, algorithm::eltwise_relu, 0.0f, 0.0f);
    }
    op_attr.set_post_ops(post_ops);
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);
    op_attr.set_output_scales(utils::op_scale_mask(scale_size), r_scale);

    auto dst_dims = {src.get_dim(0), weights.get_dim(0)};
    auto dst_data_type = dequant ? data_type::f32 : data_type::u8;
    tensor::desc dst_desc(dst_dims, dst_data_type, format_tag::any);

    auto pd = primitive_desc({aprop_kind, src_desc, weights_desc, bias_desc,
                              dst_desc}, op_attr, aengine);

    tensor scratchpad(pd.scratchpad_desc());

    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc(), weights_attr);
    auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);
    dst.reinit_if_possible(pd.dst_desc());

    super(pd).execute(stream::default_stream(),
                      {{ZENDNN_ARG_SRC, expected_src},
                      {ZENDNN_ARG_WEIGHTS, expected_weights},
                      {ZENDNN_ARG_BIAS, expected_bias},
                      {ZENDNN_ARG_DST, dst},
                      {ZENDNN_ARG_SCRATCHPAD, scratchpad}});

  }


  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      const dims& src_dims = dims(),
      data_type dtype = data_type::f32,
      data_type x_dtype = data_type::f32,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto x_dims = weights_dims;
    x_dims[0] = src_dims.empty() ? 1 : src_dims[0];
    auto y_dims = {x_dims[0], weights_dims[0]};
    auto ndims = weights_dims.size();
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;

    ADEEP_ENFORCE(x_dims.size() == weights_dims.size(),
                  "Invalid dims for data and weights");
    tensor::desc src_desc(x_dims, x_dtype, tag::any);
    tensor::desc dst_desc(y_dims, y_dtype, tag::any);
    tensor::desc weights_desc(weights_dims, dtype, tag::any);
    auto pd =
        primitive_desc({aprop_kind, src_desc, weights_desc, dst_desc}, aengine);
    return pd.weights_desc();
  }

private:
  template <bool with_bias>
  static void compute_impl(const tensor& src,
                           const tensor& weights,
                           const tensor& bias,
                           tensor& dst,
                           const scale_t& src_scales,
                           const scale_t& weights_scales,
                           const scale_t& dst_scales,
                           const attr_t& attr,
                           const prop_kind aprop_kind,
                           const lowp_kind alowp_kind,
                           const engine& aengine) {
    // workaround: src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    auto src_ = src;
    if (src.ndims() != weights.ndims()) {
      auto new_dims = weights.get_dims();
      new_dims[0] = src.get_dim(0);
      src_.reshape(new_dims);
    }
    compute_impl_<with_bias>(src_, weights, bias, dst, src_scales,
                             weights_scales, dst_scales, attr, aprop_kind,
                             alowp_kind, aengine);
  }

  template <bool with_bias>
  static void compute_impl_(const tensor& src,
                            const tensor& weights,
                            const tensor& bias,
                            tensor& dst,
                            const scale_t& src_scales,
                            const scale_t& weights_scales,
                            const scale_t& dst_scales,
                            const attr_t& attr,
                            const prop_kind aprop_kind,
                            const lowp_kind alowp_kind,
                            const engine& aengine) {
    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    scale_t dst_scales_in;
    data_type dst_data_type;
    auto dst_dims = {src.get_dim(0), weights.get_dim(0)};

    auto weights_scales_in =
        weights.has_scale() ? weights.get_scale() : weights_scales;

    // TODO(xpz): Remove int8 inner product implementation. We are switching to
    // matmul for quantized *mm ops
    if (!weights_scales_in.empty()) {
      ADEEP_ENFORCE(alowp_kind == u8s8 || alowp_kind == s8s8,
                    "Unsupported lowp kind");

      auto src_scales_in =
          src.has_scale() ? src.get_scale()
                          : src_scales.empty() ? ADEEP_DEF_SCALE : src_scales;

      src_desc = {src.get_dims(),
                  alowp_kind == u8s8 ? data_type::u8 : data_type::s8,
                  tag::any};
      if (src.get_data_type() == data_type::f32) {
        src_attr = {0, src_scales_in};
      }

      int scale_size = weights_scales_in.size() > 1 ? weights.get_dim(0) : 1;

      weights_desc = {weights.get_dims(), data_type::s8, tag::any};
      if (weights.get_data_type() == data_type::f32) {
        weights_attr = {utils::tensor_scale_mask(scale_size, false),
                        weights_scales_in};
      }

      // determine dst data type
      if (dst_scales.empty() || dst_scales == ADEEP_DEF_SCALE) {
        dst_data_type = data_type::f32;
      } else if (attr.non_negitive_output()) {
        dst_data_type = data_type::u8;
      } else {
        dst_data_type = data_type::s8;
      }

      // fill primitive attr
      scale_t op_scales(scale_size), bias_scales(scale_size);
      dst_scales_in = dst_scales.empty() || dst_data_type == data_type::f32
                          ? ADEEP_DEF_SCALE
                          : dst_scales;
      for (int i = 0; i < scale_size; i++) {
        bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
        op_scales[i] = dst_scales_in[0] / bias_scales[i];
      }
      op_attr.set_output_scales(utils::op_scale_mask(scale_size), op_scales);

      if (with_bias) {
        bias_desc = {bias.get_dims(), data_type::s32, format_tag::x};
        if (bias.get_data_type() == data_type::f32) {
          bias_attr = {utils::tensor_scale_mask(scale_size, false),
                       bias_scales};
        }
      }
    } else {
      op_attr = attr;
      src_desc = {src.get_dims(), data_type::f32, format_tag::any};
      if (src.has_scale()) {
        auto src_scale = src.get_scale();
        src_scale[0] = 1.f / src_scale[0];
        src_attr = {0, src_scale};
      }

      ADEEP_ENFORCE(utils::one_of(weights.get_data_type(),
                                  data_type::f32, data_type::bf16),
              "Incorrect data type in weights");

      // align weights data type with src
      dst_data_type = src.get_data_type() == data_type::bf16 ? data_type::bf16
                                                             : data_type::f32;
      src_desc = src.get_desc().to_type(dst_data_type);
      weights_desc = weights.get_desc().to_type(dst_data_type);
      if (with_bias) {
        ADEEP_ENFORCE(utils::one_of(bias.get_data_type(),
                                    data_type::f32, data_type::bf16),
                      "Incorrect data type in bias");
        //bias_desc = bias.get_desc().to_format_any();
	bias_desc = {bias.get_dims(), bias.get_data_type(), format_tag::x};
      }
    }

    tensor::desc dst_desc(dst_dims, dst_data_type, format_tag::nc);
    auto pd = with_bias
       ? primitive_desc({aprop_kind, src_desc, weights_desc, bias_desc,
                         dst_desc}, op_attr, aengine)
       : primitive_desc({aprop_kind, src_desc, weights_desc, dst_desc},
                        op_attr, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc(), weights_attr);

    /*std::cout<<pd.src_desc().get_size()<<std::endl;
    std::cout<<pd.weights_desc().get_size()<<std::endl;
    std::cout<<pd.bias_desc().get_size()<<std::endl;
    std::cout<<pd.dst_desc().get_size()<<std::endl;*/

    dst.reinit_if_possible(pd.dst_desc());
    if (!dst_scales.empty() && dst.get_data_type() != data_type::f32) {
      dst.set_scale(dst_scales_in);
    }

    if (with_bias){
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);
      super(pd).execute(stream::default_stream(),
                        {{ZENDNN_ARG_SRC, expected_src},
                         {ZENDNN_ARG_WEIGHTS, expected_weights},
                         {ZENDNN_ARG_BIAS, expected_bias},
                         {ZENDNN_ARG_DST, dst}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{ZENDNN_ARG_SRC, expected_src},
                         {ZENDNN_ARG_WEIGHTS, expected_weights},
                         {ZENDNN_ARG_DST, dst}});
    }

    if (attr.non_negitive_output() && dst.get_data_type() == data_type::s8) {
      dst.to_type(data_type::u8);
    }
  }
};


struct inner_product_backward_data : public zendnn::inner_product_backward_data {

  using super = zendnn::inner_product_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const engine& aengine = engine::cpu_engine()) {
    auto weights_ = weights;
    if (diff_dst.get_data_type() == data_type::bf16) {
      weights_.init(weights.get_desc().to_type(data_type::bf16));
      weights_.reorder_from(weights);
    }

    // workaround: diff_src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    if (diff_src_dims.size() != weights.ndims()) {
      auto new_dims = diff_src_dims;
      new_dims[0] = weights.get_dim(0);
      weights_.reshape(new_dims);
    }

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto weights_desc = weights_.get_desc();
    auto diff_src_desc =
        tensor::desc(diff_src_dims, diff_dst.get_data_type(), tag::any);

    auto forward_hints =
        inner_product_forward::primitive_desc(
            {prop_kind::forward, diff_src_desc, weights_desc, diff_dst_desc},
            aengine);

    auto pd = primitive_desc(
        {diff_src_desc, weights_desc, diff_dst_desc}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    super(pd).execute(stream::default_stream(),
                      {{ZENDNN_ARG_DIFF_DST, expected_diff_dst},
                       {ZENDNN_ARG_WEIGHTS, expected_weights},
                       {ZENDNN_ARG_DIFF_SRC, diff_src}});
  }
};

struct inner_product_backward_weights
    : public zendnn::inner_product_backward_weights {

  using super = zendnn::inner_product_backward_weights;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const data_type diff_weight_type = data_type::undef,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights, diff_bias, diff_weight_type);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      const data_type diff_weight_type = data_type::undef,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights, dummy_diff_bias, diff_weight_type);
  }

private:
  template<bool with_diff_bias = true>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const data_type diff_weight_type,
                           const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc().to_format_any();
    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto diff_weights_dims = src.get_dims();
    diff_weights_dims[0] = diff_dst.get_dim(1);
    data_type diff_dst_type = diff_dst.get_data_type();
    data_type diff_weight_type_in = data_type::undef== diff_weight_type ?
                                    diff_dst_type : diff_weight_type;
    auto diff_weights_desc =
        tensor::desc(diff_weights_dims, diff_weight_type_in, tag::any);

    auto diff_bias_desc =
        tensor::desc({diff_dst.get_dim(1)}, diff_weight_type_in, tag::any);

    // for forward hint, weights_desc should have same data_type
    // with other input desc, expect for bias_desc
    auto weights_desc = diff_weights_desc;
    if (diff_weight_type_in != diff_dst_type) {
      weights_desc = weights_desc.to_type(diff_dst_type);
    }
    auto forward_hints = with_diff_bias
        ? inner_product_forward::primitive_desc({prop_kind::forward, src_desc,
            weights_desc, diff_bias_desc, diff_dst_desc}, aengine)
        : inner_product_forward::primitive_desc({prop_kind::forward, src_desc,
            weights_desc, diff_dst_desc}, aengine);
    auto pd = with_diff_bias
        ? primitive_desc({src_desc, diff_weights_desc, diff_bias_desc,
                          diff_dst_desc}, aengine, forward_hints)
        : primitive_desc({src_desc, diff_weights_desc, diff_dst_desc},
                          aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    diff_weights.reinit_if_possible(pd.diff_weights_desc());

    exec_args args {{ZENDNN_ARG_DIFF_DST, expected_diff_dst},
                    {ZENDNN_ARG_SRC, expected_src},
                    {ZENDNN_ARG_DIFF_WEIGHTS ,diff_weights}};

    if (with_diff_bias) {
      diff_bias.reinit_if_possible(pd.diff_bias_desc());
      args.insert({ZENDNN_ARG_DIFF_BIAS, diff_bias});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace adeep

#endif
