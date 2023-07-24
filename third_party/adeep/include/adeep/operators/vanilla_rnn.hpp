/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#ifndef ADEEP_OPERATORS_VANILLA_RNN_HPP
#define ADEEP_OPERATORS_VANILLA_RNN_HPP

namespace adeep {

struct rnn_forward : public zendnn::vanilla_rnn_forward {
  static void compute(const tensor& src_layer, const tensor& src_iter,
      const tensor& weights_layer, const tensor& weights_iter, const tensor& bias,
      const dims& dst_layer_dims, tensor& dst_layer,
      const dims& dst_iter_dims, tensor& dst_iter,
      tensor& workspace, rnn_kind akind, zendnn_rnn_direction_t direction,
      prop_kind aprop_kind = prop_kind::forward_training) {
  }
};

struct rnn_backward : public zendnn::vanilla_rnn_backward {
  template <class alloc = utils::allocator>
  static void compute(const tensor& src_layer, const tensor& src_iter, const tensor& weights_layer,
      const tensor& weights_iter, const tensor& bias, const tensor& dst_layer, const tensor& dst_iter,
      const tensor& diff_dst_layer, const tensor& diff_dst_iter, const tensor& workspace,
      const bool with_bias, tensor& diff_src_layer, tensor& diff_src_iter, tensor& diff_weights_layer,
      tensor& diff_weights_iter, tensor& diff_bias, rnn_kind akind, zendnn_rnn_direction_t direction,
      prop_kind aprop_kind = prop_kind::backward) {
  }
};

}  // namespace adeep

#endif