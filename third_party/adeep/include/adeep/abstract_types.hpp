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

#ifndef ADEEP_ABSTRACT_TYPES_HPP
#define ADEEP_ABSTRACT_TYPES_HPP

#include <string>
#include <cstring>
#include <map>
#include <vector>
#include <cstdlib>
#include <functional>
#include <zendnn.h>
#include <zendnn.hpp>
#include "allocators.hpp"

namespace adeep {

#ifdef _WIN32
#define ADEEP_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define ADEEP_EXPORT __attribute__((__visibility__("default")))
#else
#define ADEEP_EXPORT
#endif

using error = zendnn::error;
using memory = zendnn::memory;
using format_tag = memory::format_tag;
using tag = memory::format_tag;
using data_type = typename memory::data_type;
using dims = typename memory::dims;
using dim = memory::dim;
using query = zendnn::query;
using kind = zendnn::primitive::kind;
using prop_kind = zendnn::prop_kind;
using algorithm = zendnn::algorithm;
using batch_normalization_flag = zendnn::normalization_flags;
using query = zendnn::query;
using scale_t = std::vector<float>;
using zero_point_t = std::vector<int32_t>;
using exec_args = std::unordered_map<int, memory>;

// for computation cache
using key_t = std::string;

#ifndef NDEBUG
#define ADEEP_ENFORCE(condition, message) \
  do {  \
    error::wrap_c_api((condition) \
        ? zendnn_success : zendnn_invalid_arguments, (message));  \
  } while(false)
#else
#define ADEEP_ENFORCE(condition, message)
#endif

const scale_t ADEEP_DEF_SCALE {1.0f};

enum lowp_kind {
  u8s8 = 0,
  s8s8 = 1,
  LOWP_U8S8 = u8s8,
  LOWP_S8S8 = s8s8,
};

enum rnn_kind {
  RNN_RELU = 0,
  RNN_TANH = 1,
  LSTM = 2,
  GRU = 3
};

/// cpu execution engine only.
struct engine : public zendnn::engine {
  friend class tensor;

  /// Singleton CPU engine for all primitives
  static ADEEP_EXPORT engine& cpu_engine();

  /// Singleton GPU engine for all primitives
  static ADEEP_EXPORT engine& gpu_engine();

  engine(kind akind = kind::cpu, size_t index = 0)
      : zendnn::engine(akind, index),
        malloc(utils::allocator::malloc),
        free(utils::allocator::free) {}

  void set_allocator(const std::function<void*(size_t)>& malloc,
                     const std::function<void(void*)>& free) {
    this->malloc = malloc;
    this->free = free;
  }

 private:
  std::function<void*(size_t)> malloc;
  std::function<void(void*)> free;
};

/// A default stream
struct stream : public zendnn::stream {
  static zendnn::stream& default_stream() {
    static zendnn::stream s(engine::cpu_engine());
    return s;
  }
};
}

#endif
