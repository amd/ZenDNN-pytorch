#include <c10/core/impl/cow/state_machine.h>

#include <c10/util/Exception.h>

namespace c10::impl {

auto cow::StateMachine::maybe_bump(cow::ShadowStorage* maybe_shadow_storage)
    -> void {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!default_shadow_storage_.has_value()) {
    // Any created shadow storage should be bound to the physical
    // storage that it was created from. Hence, there should only be a
    // shadow storage on a tensor if its own storage created it. We
    // don't check for the specific matching of the storage and shadow
    // storage, but we do check that the presence relationship holds.
    //
    // TODO: This check should probably be here, but it is actually
    // triggering in the StaticRuntime.autogen_inner test.
    //
    // TORCH_INTERNAL_ASSERT(maybe_shadow_storage == nullptr);
    return;
  }

  std::uint64_t physical_generation = ++physical_generation_;
  std::uint64_t shadow_generation = maybe_shadow_storage != nullptr
      ? maybe_shadow_storage->bump_generation()
      : default_shadow_storage_->bump_generation();

  TORCH_INTERNAL_ASSERT(shadow_generation <= physical_generation);
  if (shadow_generation != physical_generation) {
    TORCH_WARN_ONCE(
        "You have written through to both aliases created by calling "
        "reshape(). In the future, reshape() will never create a view but will "
        "instead return a lazily copied tensor. If you wish to preserve the "
        "aliasing properties, you should rewrite your reshape() as a view().");
  }
}

auto cow::StateMachine::physical_generation() -> std::uint64_t {
  std::lock_guard<std::mutex> lock(mtx_);
  return physical_generation_;
}

auto cow::StateMachine::simulate_lazy_copy(cow::ShadowStorage* shadow_storage)
    -> intrusive_ptr<cow::ShadowStorage> {
  std::lock_guard<std::mutex> lock(mtx_);
  if (shadow_storage != nullptr) {
    TORCH_INTERNAL_ASSERT(default_shadow_storage_.has_value());
    return make_intrusive<cow::ShadowStorage>(shadow_storage->generation());
  }

  if (!default_shadow_storage_.has_value()) {
    TORCH_INTERNAL_ASSERT(physical_generation_ == 0);
    default_shadow_storage_.emplace(physical_generation_);
  }

  return make_intrusive<cow::ShadowStorage>(
      default_shadow_storage_->generation());
}

} // namespace c10::impl
