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

#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <ostream>

namespace c10 {
enum class Layout : int8_t {
  Strided,
  Sparse,
  SparseCsr,
  Mkldnn,
  Zendnn,
  SparseCsc,
  SparseBsr,
  SparseBsc,
  NumOptions
};

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kSparseCsr = Layout::SparseCsr;
constexpr auto kMkldnn = Layout::Mkldnn;
constexpr auto kZendnn = Layout::Zendnn;
constexpr auto kSparseCsc = Layout::SparseCsc;
constexpr auto kSparseBsr = Layout::SparseBsr;
constexpr auto kSparseBsc = Layout::SparseBsc;

inline Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
    case Backend::SparseVE:
    case Backend::SparseXPU:
      return Layout::Sparse;
    case Backend::MkldnnCPU:
      return Layout::Mkldnn;
    case Backend::ZendnnCPU:
      return Layout::Zendnn;
    case Backend::SparseCsrCPU:
    case Backend::SparseCsrCUDA:
      TORCH_CHECK(
          false,
          "Cannot map Backend SparseCsrCPU|SparseCsrCUDA to a unique layout.");
    default:
      return Layout::Strided;
  }
}

inline std::ostream& operator<<(std::ostream& stream, at::Layout layout) {
  switch (layout) {
    case at::kStrided:
      return stream << "Strided";
    case at::kSparse:
      return stream << "Sparse";
    case at::kSparseCsr:
      return stream << "SparseCsr";
    case at::kSparseCsc:
      return stream << "SparseCsc";
    case at::kSparseBsr:
      return stream << "SparseBsr";
    case at::kSparseBsc:
      return stream << "SparseBsc";
    case at::kMkldnn:
      return stream << "Mkldnn";
    case at::kZendnn:
      return stream << "Zendnn";
    default:
      TORCH_CHECK(false, "Unknown layout");
  }
}

} // namespace c10
