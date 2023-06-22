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

#ifndef ADEEP_COMPUTATIONS_HPP
#define ADEEP_COMPUTATIONS_HPP

#include "lru_cache.hpp"
#include "operators/batchnorm.hpp"
#include "operators/binary.hpp"
#include "operators/channel_shuffle.hpp"
#include "operators/concat.hpp"
#include "operators/conv.hpp"
#include "operators/deconv.hpp"
#include "operators/direct_copy.hpp"
#include "operators/dropout.hpp"
#include "operators/eltwise.hpp"
#include "operators/gru.hpp"
#include "operators/inner_product.hpp"
#include "operators/layernorm.hpp"
#include "operators/lbr_gru.hpp"
#include "operators/lrn.hpp"
#include "operators/lstm.hpp"
#include "operators/matmul.hpp"
#include "operators/pool.hpp"
#include "operators/prelu.hpp"
#include "operators/softmax.hpp"
#include "operators/spliter.hpp"
#include "operators/sum.hpp"
#include "operators/vanilla_rnn.hpp"
#include "operators/embed_bag.hpp"
#include "operators/embed_bag_group.hpp"
#endif
