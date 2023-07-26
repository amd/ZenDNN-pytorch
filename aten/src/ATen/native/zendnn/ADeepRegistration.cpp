/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

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
