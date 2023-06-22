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