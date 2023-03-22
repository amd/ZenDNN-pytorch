#include <c10/core/impl/copy_on_write_state.h>

#include <c10/util/Exception.h>

namespace c10::impl {

auto CopyOnWriteState::maybe_bump(CopyOnWriteSimulator* maybe_simulator)
    -> void {
  std::lock_guard<std::mutex> lock(mtx_);
  if (default_simulator_ == nullptr) {
    TORCH_INTERNAL_ASSERT(maybe_simulator == nullptr);
    return;
  }

  CopyOnWriteSimulator& simulator =
      maybe_simulator != nullptr ? *maybe_simulator : *default_simulator_;

  std::uint64_t physical_generation = ++physical_generation_;
  std::uint64_t shadow_generation = simulator.bump_storage_generation();

  TORCH_INTERNAL_ASSERT(shadow_generation <= physical_generation);
  if (shadow_generation != physical_generation) {
    TORCH_WARN_ONCE(
        "You have written through to both aliases created by calling "
        "reshape(). In the future, reshape() will never create a view but will "
        "instead return a lazily copied tensor. If you wish to preserve the "
        "aliasing properties, you should rewrite your reshape() as a view().");
  }
}

auto CopyOnWriteState::storage_generation() -> std::uint64_t {
  std::lock_guard<std::mutex> lock(mtx_);
  return physical_generation_;
}

auto CopyOnWriteState::simulate_copy_on_write(CopyOnWriteSimulator* simulator)
    -> intrusive_ptr<CopyOnWriteSimulator> {
  std::lock_guard<std::mutex> lock(mtx_);
  if (simulator != nullptr) {
    TORCH_INTERNAL_ASSERT(default_simulator_ != nullptr);
    return make_intrusive<CopyOnWriteSimulator>(
        simulator->storage_generation());
  }

  if (default_simulator_ == nullptr) {
    TORCH_INTERNAL_ASSERT(physical_generation_ == 0);
    default_simulator_ =
        std::make_unique<CopyOnWriteSimulator>(physical_generation_);
  }

  return make_intrusive<CopyOnWriteSimulator>(physical_generation_);
}

} // namespace c10::impl
