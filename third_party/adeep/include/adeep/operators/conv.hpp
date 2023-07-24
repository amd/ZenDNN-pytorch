/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#ifndef ADEEP_OPERATORS_CONV_HPP
#define ADEEP_OPERATORS_CONV_HPP

namespace adeep {

struct convolution_forward_params {
  zendnn::convolution_forward::primitive_desc pd;
  // bias_attr contains requantization scales for bias
  attr_t bias_attr;
  scale_t dst_scales;
  int groups;
  tensor scratchpad;
};

struct convolution_forward
    : public zendnn::convolution_forward,
      utils::computation_cache<zendnn::convolution_forward::primitive_desc> {

  using super = zendnn::convolution_forward;

  // prepare with bias
  static void prepare(
      convolution_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    do_prepare</*with_bias=*/true, /*keep_format=*/false>(
        param, src, weights, bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // prepare without bias
  static void prepare(
      convolution_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    do_prepare</*with_bias=*/false, /*keep_format=*/false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // compute with bias
  static void compute(const convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst) {
    do_compute</*with_bias=*/true>(param, src, weights, bias, dst);
  }

  // compute without bias
  static void compute(const convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      tensor& dst) {
    static tensor dummy_bias;
    do_compute</*with_bias=*/false>(param, src, weights, dummy_bias, dst);
  }

  // 2-in-1 compute (prepare & compute) with bias
  template <bool plain_format = false>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    compute_dispatch</*with_bias=*/true, plain_format>(
        src, weights, bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // 2-in-1 compute (prepare & compute) without bias
  template <bool plain_format = false>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_dispatch</*with_bias=*/false, plain_format>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // Quantized convolution kernel for vitis_ai convolution
  static void zen_vitis_compute(const tensor& src,
                                const tensor& weights,
                                const tensor& bias,
                                const dims& dst_dims,
                                tensor& dst,
                                const dims& strides,
                                const dims& dilates,
                                const dims& padding_l,
                                const dims& padding_r,
                                int groups,
                                bool dequant,
                                bool fuse_relu,
                                bool fuse_add,
                                float input_scale,
                                float filter_scale,
                                float bias_scale,
                                float requantize_scale,
                                float dequantize_scale,
                                float add_scale,
                                float add_out_scale,
                                algorithm aalgorithm = algorithm::convolution_direct,
                                prop_kind aprop_kind = prop_kind::forward,
                                const engine& aengine = engine::cpu_engine()) {

    // Convert the entered scales from float to scale_t
    const scale_t& i_scale = {input_scale};
    const scale_t& f_scale = {filter_scale};
    const scale_t& b_scale = {bias_scale};
    const scale_t& r_scale = {requantize_scale};
    const scale_t& d_scale = {dequantize_scale};

    data_type dtype_s8 = data_type::s8;
    data_type dtype_u8 = data_type::u8;
    data_type dtype_s32 = data_type::s32;
    data_type dtype_f32 = data_type::f32;
    tensor::desc src_desc, weights_desc, bias_desc, dst_desc;    // Descriptors for source, weights, and bias tensors
    attr_t src_attr, weights_attr, bias_attr, op_attr;           // Attrs for source, weights, bias, and output

    // Make weights and dilates compatible with ZENDNN
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    // Set number of scales
    int scale_size = 1;

    // Set the operation type to fused relu
    post_ops post_ops;
    if (fuse_add){
      post_ops.append_sum(add_scale);
    }
    if (fuse_relu){
      if(dst.get_data_type() == data_type::s8){
        post_ops.append_eltwise(1.0f, algorithm::eltwise_clip, 0, 127);
      }
      post_ops.append_eltwise(add_out_scale, algorithm::eltwise_relu, 1.0f, 0.0f);
    }
    else if(add_out_scale != 1.0f){
      post_ops.append_eltwise(add_out_scale, algorithm::eltwise_linear, 1.0f, 0.0f);
    }
    op_attr.set_post_ops(post_ops);

    // Set the input scale on the input tensor attr
    if (src.get_data_type() == dtype_f32) {
      src_desc = {src.get_dims(), dtype_s8, tag::any};
      src_attr = {utils::tensor_scale_mask(scale_size, groups > 1), i_scale};
    } else {
      src_desc = {src.get_dims(), src.get_data_type(), tag::any};
    }

    // Set the weight scale on the weight tensor attr
    weights_desc = weights_.get_desc().to_type(dtype_s8);
    if (weights_.get_data_type() == dtype_f32){
      weights_attr = {utils::tensor_scale_mask(scale_size, groups > 1), f_scale};
    }

    // Set the bias scale on the bias tensor attr
    bias_desc = {bias.get_dims(), dtype_s32, tag::any};
    if (bias.get_data_type() == dtype_f32){
      bias_attr = {utils::tensor_scale_mask(scale_size, false), b_scale};
    }

    // Set the output scale on the output tensor
    op_attr.set_output_scales(utils::op_scale_mask(scale_size), r_scale);
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    if(fuse_add){
      dst_desc = dst.get_desc();
    }
    else if (fuse_relu){
        dst_desc = tensor::desc(dst_dims, dtype_u8);
    }
    else {
        dst_desc = tensor::desc(dst_dims, dtype_s8);
    }

    // Get the primitive descriptor for the convolution
    auto pd = get_primitive_desc</*with_bias*/ true, /*keep_format=*/false>(
        src_desc, weights_desc, bias_desc, dst_desc, strides, dilates_,
        padding_l, padding_r, op_attr, aalgorithm, aprop_kind, aengine
    );

    // Allocate scratchpad
    tensor scratchpad(pd.scratchpad_desc());

    // Reorder the source, weights, and bias to make them quantized types
    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights = weights.make_grouped_weights(groups).reorder_if_differ_in(pd.weights_desc(), weights_attr);
    auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);

    // Prepare the output tensor
    dst.reinit_if_possible(pd.dst_desc());

    // Perform the operation
    super(pd).execute(stream::default_stream(),
                      {{ZENDNN_ARG_SRC, expected_src},
                      {ZENDNN_ARG_WEIGHTS, expected_weights},
                      {ZENDNN_ARG_BIAS, expected_bias},
                      {ZENDNN_ARG_DST, dst},
                      {ZENDNN_ARG_SCRATCHPAD, scratchpad}});

    if(dst.get_data_type() == data_type::s8 && fuse_relu){
      dst = dst.reorder_if_differ_in(tensor::desc(dst_dims, data_type::u8, tag::acdb));
    }

    // Dequantize if required
    if (dequant){
      dst = dst.reorder_if_differ_in(
        tensor::desc(dst_dims, dtype_f32, tag::acdb),
        {utils::tensor_scale_mask(scale_size, groups > 1), d_scale}
      );
    }
  }

  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      data_type dtype = data_type::f32,
      const dims& strides = {1, 1},
      const dims& padding_l = {0, 0},
      const dims& padding_r = {0, 0},
      const dims& dilates = {1, 1},
      int groups = 1,
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      data_type x_dtype = data_type::f32,
      const dims& src_dims = dims(),
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {

    auto src_size = weights_dims.size(); // weights_dims is 4 for conv2d and 5 for conv3d
    auto grouped = groups > 1;
    auto weights_dims_g =
        grouped ? utils::group_dims(weights_dims, groups) : weights_dims;
    auto weights_desc = tensor::desc(weights_dims_g, dtype);

    auto dims_in = weights_desc.get_dims();
    auto ndims = dims_in.size();
    auto dilates_ = utils::get_compatible_dilates(dilates, src_size);

    ADEEP_ENFORCE(
        !(aalgorithm == algorithm::convolution_winograd && src_dims.empty()),
        "Incorrect src_dims");
    dims x_dims, y_dims, kernel_size;
    auto ic = groups * dims_in[1 + grouped];
    auto oc = groups * dims_in[0 + grouped];
    if (5 == src_size) {
      kernel_size.push_back(dims_in[ndims - 3]);
    }
    kernel_size.push_back(dims_in[ndims - 2]);
    kernel_size.push_back(dims_in[ndims - 1]);
    if (src_dims.empty()) {
      // Construct a dummy case
      x_dims.push_back(1);
      x_dims.push_back(ic);
      y_dims.push_back(1);
      y_dims.push_back(oc);
      if (4 == src_size) {
        x_dims.push_back(4 * kernel_size[0]);
        x_dims.push_back(4 * kernel_size[1]);
      } else {
        x_dims.push_back(8 * kernel_size[0]);
        x_dims.push_back(8 * kernel_size[1]);
        x_dims.push_back(8 * kernel_size[2]);
      }
    } else {
      // Use the real data
      for (auto i=0; i < src_size; ++i) {
        x_dims.push_back(src_dims[i]);
      }
      y_dims.push_back(src_dims[0]);
      y_dims.push_back(oc);
    }
    for (auto d = 2; d < src_size; ++d) {
      auto out_size = (x_dims[d] - ((kernel_size[d-2] - 1) * (dilates_[d-2] + 1) + 1)
          + (padding_l[d-2] + padding_r[d-2])) / strides[d-2] + 1;
      y_dims.push_back(out_size);
    }
    x_dtype = dtype == data_type::bf16 ? dtype : x_dtype;
    auto y_dtype = dtype != data_type::s8 ? dtype : data_type::s32;
    tensor::desc src_desc(x_dims, x_dtype);
    tensor::desc dst_desc(y_dims, y_dtype);

    // FIXME: workaroud winograd format issue in inference
    // If prop_kind == forward_inference, the zendnn_wino_fmt for weights is
    // required by winograd primitive. Then, in the cases of variable input
    // shape, the detials of zendnn_wino_fmt will be changed. And, extra weihgts
    // reorder is inevitable each time, leading to bad performance. Here, we set
    // the prop_kind to forward, in order to reorder and cache weights as
    // blocked format, instead of zendnn_wino_fmt.
    auto apkind = aprop_kind;
    if (aalgorithm == algorithm::convolution_winograd &&
        aprop_kind == prop_kind::forward_inference) {
      apkind = prop_kind::forward;
    }

    auto pd = get_primitive_desc</*with_bias=*/false>(
        src_desc, weights_desc, tensor::desc(), dst_desc, strides, dilates_,
        padding_l, padding_r, attr_t(), aalgorithm, apkind);

    // embed group info into weights_desc
    return tensor::desc(pd.weights_desc(), groups);
  }

  // [keep_format]
  // - Set to true would let onednn to choose the optimal
  //   blocked format for dst tensor
  // - Set to false would keep dst tensor format as it is.
  //   We used this mode in pytorch plain-in-plain-out path to force
  //   the dst to be plain as src, so that it would also instruct onednn
  //   to use gemm-based conv implementation. Apply to both NCHW and NHWC.
  template <bool with_bias, bool keep_format = false>
  static primitive_desc get_primitive_desc(
      const tensor::desc& src_desc,
      const tensor::desc& weights_desc,
      const tensor::desc& bias_desc,
      const tensor::desc& dst_desc,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc_query = src_desc;
    auto weights_desc_query = weights_desc;
    auto bias_desc_query = with_bias ? bias_desc : tensor::desc();
    auto dst_desc_query = dst_desc;
    if (!keep_format) {
      src_desc_query = src_desc.to_format_any();
      weights_desc_query = weights_desc.to_format_any();
      bias_desc_query = with_bias ? bias_desc.to_format_any() : tensor::desc();
      dst_desc_query = dst_desc.to_format_any();
    }

    // For nhwc path, weight uses format_tag::any,
    // while activation uses format_tag::nhwc.
    bool is_nhwc = src_desc.is_nhwc() || weights_desc.is_nhwc();
    if (is_nhwc) {
      src_desc_query = src_desc.to_format(tag::nhwc);
      weights_desc_query = weights_desc.to_format_any();
      bias_desc_query = with_bias ? bias_desc.to_format_any() : tensor::desc();
      dst_desc_query = dst_desc.to_format(tag::nhwc);
    }

    auto key = utils::create_key(aprop_kind, aalgorithm, src_desc_query,
                                 weights_desc_query, with_bias, strides,
                                 dilates, padding_l, padding_r, attr);
    return fetch_or_create(key, [&]() {
    if (with_bias) {
      return primitive_desc({aprop_kind, aalgorithm, src_desc_query,
                             weights_desc_query, bias_desc_query, dst_desc_query,
                             strides, dilates, padding_l, padding_r},
                            attr, aengine);
    } else {
      return primitive_desc({aprop_kind, aalgorithm, src_desc_query,
                             weights_desc_query, dst_desc_query,
                             strides, dilates, padding_l, padding_r},
                            attr, aengine);
    }
    });
  }

private:
  static bool use_gemm(const dims& src, const dims& weight, const dims& dst,
                       int groups) {
    if (groups != 1)
      return false;

    auto product = [](const dims& v, size_t start_offset = 0) {
      return std::accumulate(
          v.begin() + start_offset, v.end(), 1, std::multiplies<size_t>());
    };

    auto ker_spatial = product(weight, 2);
    bool pointwise = ker_spatial == 1;
    if (pointwise)
      return true;

    auto im2col_cost = ker_spatial * product(src);
    auto reorder_cost = product(src) + 2 * product(weight) + 2 * product(dst);
    return im2col_cost < reorder_cost;
  }

  template <bool with_bias, bool plain_format>
  static void compute_dispatch(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    convolution_forward_params params;

    if (plain_format) {
      // Used for pytorch default CPU path, i.e. plain-in-plain-out
      // see [keep_format] for more details
      bool is_nhwc = src.get_desc().is_nhwc() || weights.get_desc().is_nhwc();
      bool use_plain_dst = use_gemm(src.get_dims(), weights.get_dims(), dst_dims, groups) || is_nhwc;
      if (use_plain_dst) {
        do_prepare<with_bias, /*keep_format=*/true>(
            params, src, weights, bias, dst_dims, dst, strides, dilates,
            padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
            attr, aalgorithm, aprop_kind, alowp_kind, aengine);
        do_compute<with_bias>(params, src, weights, bias, dst);
      } else {
        tensor dst_blocked;
        do_prepare<with_bias, /*keep_format=*/false>(
            params, src, weights, bias, dst_dims, dst_blocked, strides, dilates,
            padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
            attr, aalgorithm, aprop_kind, alowp_kind, aengine);
        do_compute<with_bias>(params, src, weights, bias, dst_blocked);
        dst.feed_from(dst_blocked);
      }
    } else {
      // Used for to_zendnn() path
      do_prepare<with_bias, /*keep_format=*/false>(
          params, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          attr, aalgorithm, aprop_kind, alowp_kind, aengine);
      do_compute<with_bias>(params, src, weights, bias, dst);
    }
  }

  template <bool with_bias, bool keep_format>
  static void do_prepare(
      convolution_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const attr_t& attr,
      algorithm aalgorithm,
      prop_kind aprop_kind,
      const lowp_kind alowp_kind,
      const engine& aengine) {

    scale_t dst_scales_in;
    data_type dst_data_type;
    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;

    // make weights and dilates compatible with ZENDNN
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto& weights_scales_in =
        weights_.has_scale() ? weights_.get_scale() : weights_scales;
    if (!weights_scales_in.empty()) {
      ADEEP_ENFORCE(alowp_kind == u8s8 || alowp_kind == s8s8,
                    "Unsupported lowp kind");
      int scale_size = (weights_scales_in.size() > 1) ? dst_dims[1] : 1;
      auto src_scales_in =
          src.has_scale() ? src.get_scale()
                          : (src_scales.empty() ? ADEEP_DEF_SCALE : src_scales);

      // determine dst data type
      if (attr.has_op_kind(kind::sum)) {
        dst_data_type = dst.get_data_type();
      } else if (dst_scales.empty() || dst_scales == ADEEP_DEF_SCALE) {
        dst_data_type = data_type::f32;
      } else if (attr.non_negitive_output()) {
        dst_data_type = data_type::u8;
      } else {
        dst_data_type = data_type::s8;
      }

      // fill primitive attr
      dst_scales_in = dst_scales.empty() || dst_data_type == data_type::f32
                          ? ADEEP_DEF_SCALE
                          : dst_scales;

      scale_t bias_scales, op_scales;
      std::tie(bias_scales, op_scales) = utils::compute_scales(
          src_scales_in[0], dst_scales_in[0], weights_scales_in);

      if (attr.has_op_kind(kind::sum)) {
        float sum_scale =
            dst_scales_in[0] / (dst.has_scale() ? dst.get_scale()[0] : 1.0f);
        if (attr.has_op_kind(kind::eltwise)) {
          op_attr = attr_t::residual(sum_scale);
        } else {
          op_attr = attr_t::fuse_sum(sum_scale);
        }
      } else if (attr.has_op_kind(kind::eltwise)) {
        op_attr = attr_t::fuse_relu();
      }
      op_attr.set_output_scales(utils::op_scale_mask(scale_size), op_scales);

      src_desc = {src.get_dims(),
                  alowp_kind == u8s8 ? data_type::u8 : data_type::s8, tag::any};
      if (src.get_data_type() == data_type::f32) {
        src_attr = {0, src_scales_in};
      }

      weights_desc = weights_.get_desc().to_type(data_type::s8);
      if (weights_.get_data_type() == data_type::f32) {
        weights_attr = {utils::tensor_scale_mask(scale_size, groups > 1),
                        weights_scales_in};
      }

      if (with_bias) {
        bias_desc = {bias.get_dims(), data_type::s32, tag::any};
        if (bias.get_data_type() == data_type::f32) {
          bias_attr = {utils::tensor_scale_mask(scale_size, false),
                       bias_scales};
        }
      }
    } else {
      op_attr = attr;

      if (src.has_scale()) {
        auto src_scale = src.get_scale();
        src_scale[0] = 1.0f / src_scale[0];
        src_attr = {0, src_scale};
      }

      ADEEP_ENFORCE(utils::one_of(weights_.get_data_type(),
                                  data_type::f32, data_type::bf16),
                    "Incorrect data type in weights");

      // align weights data type with src
      dst_data_type = src.get_data_type() == data_type::bf16 ? data_type::bf16
                                                             : data_type::f32;
      src_desc = src.get_desc().to_type(dst_data_type);
      weights_desc = weights_.get_desc().to_type(dst_data_type);

      if (with_bias) {
        ADEEP_ENFORCE(utils::one_of(bias.get_data_type(),
                                    data_type::f32, data_type::bf16),
                      "Incorrect data type in bias");
        bias_desc = bias.get_desc();
      }
    }

    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto dst_desc = attr.has_op_kind(kind::sum)
                        ? dst.get_desc()
                        : tensor::desc(dst_dims, dst_data_type);

    auto pd = get_primitive_desc<with_bias, keep_format>(
        src_desc, weights_desc, bias_desc, dst_desc, strides, dilates_,
        padding_l, padding_r, op_attr, aalgorithm, aprop_kind, aengine);

    // allocate scratchpad
    tensor scratchpad(pd.scratchpad_desc());

    param = {pd, bias_attr, dst_scales, groups, scratchpad};
  }

  template <bool with_bias>
  static void do_compute(const convolution_forward_params& param,
                         const tensor& src, const tensor& weights,
                         const tensor& bias, tensor& dst) {
    auto& pd = param.pd;
    auto scratchpad = param.scratchpad;
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_weights = weights.make_grouped_weights(param.groups)
                                .reorder_if_differ_in(pd.weights_desc());
    dst.reinit_if_possible(pd.dst_desc());

    if (!param.dst_scales.empty() && dst.get_data_type() != data_type::f32) {
      dst.set_scale(param.dst_scales);
    }

    if (with_bias) {
      auto expected_bias =
          bias.reorder_if_differ_in(pd.bias_desc(), param.bias_attr);
      super(pd).execute(stream::default_stream(),
                        {{ZENDNN_ARG_SRC, expected_src},
                         {ZENDNN_ARG_WEIGHTS, expected_weights},
                         {ZENDNN_ARG_BIAS, expected_bias},
                         {ZENDNN_ARG_DST, dst},
                         {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{ZENDNN_ARG_SRC, expected_src},
                         {ZENDNN_ARG_WEIGHTS, expected_weights},
                         {ZENDNN_ARG_DST, dst},
                         {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
    }
  }
};


struct convolution_backward_data : public zendnn::convolution_backward_data {

  using super = zendnn::convolution_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    // make weights and dilates compatible with ZENDNN
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    bool is_nhwc = diff_dst.get_desc().is_nhwc();
    bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
    auto format_tag = is_nhwc ? tag::nhwc : (is_ndhwc ? tag::ndhwc : tag::any);
    auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
    // align weight data type with diff_dst for bf16
    auto weights_desc =
        weights_.get_desc().to_format_any().to_type(diff_dst.get_data_type());

    auto diff_src_desc =
        tensor::desc(diff_src_dims, diff_dst_desc.get_data_type(), format_tag);

    auto forward_hints =
        convolution_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates_, padding_l, padding_r);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
         dilates_, padding_l, padding_r}, op_attr, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    tensor scratchpad(pd.scratchpad_desc());
    super(pd).execute(stream::default_stream(),
                      {{ZENDNN_ARG_DIFF_DST, expected_diff_dst},
                       {ZENDNN_ARG_WEIGHTS, expected_weights},
                       {ZENDNN_ARG_DIFF_SRC, diff_src},
                       {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
  }
};


struct convolution_backward_weights
    : public zendnn::convolution_backward_weights {

  using super = zendnn::convolution_backward_weights;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      const data_type diff_weight_type = data_type::undef,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
        strides, dilates, padding_l, padding_r, groups, diff_weight_type, aalgorithm, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      const data_type diff_weight_type = data_type::undef,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights_dims, diff_weights, dummy_diff_bias,
        strides, dilates, padding_l, padding_r, groups, diff_weight_type, aalgorithm, aengine);
  }

 private:
  template <bool with_diff_bias>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           const dims& diff_weights_dims,
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const dims& strides,
                           const dims& dilates,
                           const dims& padding_l,
                           const dims& padding_r,
                           const int groups,
                           const data_type diff_weight_type,
                           algorithm aalgorithm,
                           const engine& aengine) {

    // make diff_weights and dilates compatible with ZENDNN
    auto dilates_ = utils::get_compatible_dilates(dilates);
    data_type diff_dst_type = diff_dst.get_data_type();
    data_type diff_weight_type_in = data_type::undef == diff_weight_type ?
                                    diff_dst_type : diff_weight_type;
    auto diff_weights_desc =
        tensor::desc(diff_weights_dims, diff_weight_type_in, tag::any);
    if (groups > 1) {
        diff_weights_desc = diff_weights_desc.to_grouped(groups).to_format_any();
    }

    bool is_nhwc = diff_dst.get_desc().is_nhwc();
    bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
    auto format_tag = is_nhwc ? tag::nhwc : (is_ndhwc ? tag::ndhwc : tag::any);
    auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
    auto src_desc = src.get_desc().to_format(format_tag);

    auto diff_bias_desc =
        tensor::desc({diff_dst.get_dim(1)}, diff_weight_type_in, tag::any);

    // for forward hint, weights_desc should have same data_type
    // with other input desc, expect for bias_desc
    auto weights_desc = diff_weights_desc;
    if (diff_weight_type_in != diff_dst_type) {
      weights_desc = weights_desc.to_type(diff_dst_type);
    }
    auto forward_hints =
        convolution_forward::get_primitive_desc<with_diff_bias>(
            src_desc, weights_desc, diff_bias_desc, diff_dst_desc, strides,
            dilates_, padding_l, padding_r, attr_t(), aalgorithm,
            prop_kind::forward, aengine);

    auto op_attr = zendnn::primitive_attr();
    op_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    auto pd = with_diff_bias
        ? primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_bias_desc, diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, op_attr, aengine, forward_hints)
        : primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, op_attr, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    // embed group info into diff_weights_desc
    auto expected_diff_weights_desc =
        tensor::desc(pd.diff_weights_desc(), groups);
    diff_weights.reinit_if_possible(expected_diff_weights_desc);

    tensor scratchpad(pd.scratchpad_desc());

    if (with_diff_bias) {
      diff_bias.reinit_if_possible(pd.diff_bias_desc());
      super(pd).execute(stream::default_stream(),
                        {{ZENDNN_ARG_DIFF_DST, expected_diff_dst},
                         {ZENDNN_ARG_SRC, expected_src},
                         {ZENDNN_ARG_DIFF_WEIGHTS, diff_weights},
                         {ZENDNN_ARG_DIFF_BIAS, diff_bias},
                         {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{ZENDNN_ARG_DIFF_DST, expected_diff_dst},
                         {ZENDNN_ARG_SRC, expected_src},
                         {ZENDNN_ARG_DIFF_WEIGHTS, diff_weights},
                         {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
    }
  }
};
}  // namespace adeep

#endif
