/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#ifndef _ADEEP_PIN_SINGLETONS_HPP_
#define _ADEEP_PIN_SINGLETONS_HPP_

#include "adeep.hpp"

namespace adeep {

engine& engine::cpu_engine() {
  static engine cpu_engine(kind::cpu, 0);
  return cpu_engine;
}

engine& engine::gpu_engine() {
  static engine gpu_engine(kind::gpu, 0);
  return gpu_engine;
}

struct RegisterEngineAllocator {
  RegisterEngineAllocator(engine& eng,
                          const std::function<void*(size_t)>& malloc,
                          const std::function<void(void*)>& free) {
    eng.set_allocator(malloc, free);
  }
};

}

#endif
