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

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_ZENDNN_ENABLED()

// needs to be included only once in library.
#include <adeep_pin_singletons.hpp>

using namespace adeep;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
RegisterEngineAllocator cpu_alloc(
  engine::cpu_engine(),
  [](size_t size) {
    return c10::GetAllocator(c10::DeviceType::CPU)->raw_allocate(size);
  },
  [](void* p) {
    c10::GetAllocator(c10::DeviceType::CPU)->raw_deallocate(p);
  }
);

namespace at { namespace native { namespace zendnn {

void clear_computation_cache() {
  // Reset computation_cache for forward convolutions
  // As it also caches max number of OpenMP workers
  adeep::convolution_forward::t_store().clear();
}

}}} // namespace  at::native::zendnn

#endif // AT_ZENDNN_ENALBED()
