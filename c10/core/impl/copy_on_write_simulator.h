#pragma once

#include <c10/util/intrusive_ptr.h>

#include <cassert>
#include <cstdint>
#include <limits>
#include <mutex>

namespace c10::impl {

// Simulates a copy-on-write storage for a tensor.
//
// This simulator is used to identify situations where a copy-on-write
// tensor would have been created and whether or not a logical copy
// uses the storage after a different copy has modified it. So in
// theory, it can identify reads/writes to copies after a write to a
// different copy that is presently an alias.
//
// However, the fidelity of this check is limited by the extent to
// which we have instrumented operations as reading or writing to
// storage.
//
// We use a monotonically increasing generation number to track
// modifications to storage.
//
// Note: this privately inherits from intrusive_ptr_target and friends
// intrusive_ptr so that its intrusiveness may remain an
// implementation detail.
class CopyOnWriteSimulator final : private intrusive_ptr_target {
 public:
  // Creates an instance from an existing storage generation.
  explicit CopyOnWriteSimulator(std::uint32_t storage_generation) noexcept
      : storage_generation_(storage_generation) {}

  // Gets the current generation.
  auto storage_generation() noexcept -> std::uint32_t {
    std::lock_guard lock(mtx_);
    return storage_generation_;
  }

  // Increments the current generation.
  auto bump_storage_generation() noexcept -> std::uint32_t {
    std::lock_guard lock(mtx_);
    assert(storage_generation_ != std::numeric_limits<std::uint32_t>::max());
    return ++storage_generation_;
  }

 private:
  std::mutex mtx_;
  // From the user's perspective, copies are fully distinct and
  // require no external synchronization, so we need to ensure this
  // field's concurrency is properly managed.
  std::uint32_t storage_generation_;

  friend class intrusive_ptr<CopyOnWriteSimulator>;
};

} // namespace c10::impl
