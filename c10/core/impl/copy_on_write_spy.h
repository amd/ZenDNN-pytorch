#pragma once

#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {
struct Storage;
struct TensorImpl;
} // namespace c10

namespace c10::impl {

class CopyOnWriteSimulator;

// Allows for introspection into TensorImpl and StorageImpl's
// copy-on-write state.
class CopyOnWriteSpy {
 public:
  // Gets the generation number from the storage.
  static C10_API auto get_generation(Storage const& tensor) -> std::uint32_t;
  // Gets the copy-on-write simulator instance from the tensor.
  static C10_API auto get_simulator(TensorImpl const& tensor)
      -> CopyOnWriteSimulator const*;
};

} // namespace c10::impl
