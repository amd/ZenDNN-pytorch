#pragma once

#include <c10/core/impl/copy_on_write_simulator.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <mutex>

namespace c10::impl {

class CopyOnWriteSimulator;

// Responsible for managing the copy-on-write simulation state
// machine.
class C10_API CopyOnWriteState {
 public:
  // Gets the current storage generation.
  auto storage_generation() -> std::uint64_t;

  // Bumps the generation if simulator is non-null. If non-null,
  // simulator must be the result of a previous call to
  // simulate_copy_on_write on this instance.
  auto maybe_bump(CopyOnWriteSimulator* maybe_simulator) -> void;

  // Simulates a copy on write.
  //
  // maybe_simulator comes from the tensor that would be receiving a
  // copy on write. If non-null, then the result gets the generation
  // number from this value.
  auto simulate_copy_on_write(CopyOnWriteSimulator* maybe_simulator)
      -> intrusive_ptr<CopyOnWriteSimulator>;

 private:
  // Guards all the state.
  std::mutex mtx_;
  // How many writes have been applied to the storage.
  std::uint64_t physical_generation_ = 0;
  // The simulator to use for any tensors that don't have one. This
  // situation is common, and will be true for tensors and views
  // thereof created before any copy on writes.
  intrusive_ptr<CopyOnWriteSimulator> default_simulator_;
};

} // namespace c10::impl
