#pragma once

#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {
class Storage;
class TensorImpl;
} // namespace c10

namespace c10::impl {

class CopyOnWriteSimulator;

// Allows for internal access to TensorImpl and StorageImpl's
// copy-on-write state.
class CopyOnWritePeer {
 public:
  // Gets the generation number from the storage.
  static C10_API auto get_generation(Storage const& tensor) -> std::uint32_t;
  // Bumps generation numbers because of a read.
  static C10_API auto bump_copy_on_write_generation(TensorImpl& tensor) -> void;
  // Gets the copy-on-write simulator instance from the tensor.
  static C10_API auto get_simulator(TensorImpl const& tensor)
      -> CopyOnWriteSimulator const&;
};

} // namespace c10::impl
