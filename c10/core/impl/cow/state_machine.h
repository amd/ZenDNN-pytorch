#pragma once

#include <c10/core/impl/cow/shadow_storage.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <mutex>

namespace c10::impl::cow {

// Responsible for managing the copy-on-write simulation state
// machine.
class C10_API StateMachine {
 public:
  // Gets the current generation of the physical storage.
  auto physical_generation() -> std::uint64_t;

  // Bumps the generation if the shadow storage is non-null. If
  // non-null, shadow storage must be the result of a previous call to
  // simulate_copy_on_write on this instance.
  auto maybe_bump(cow::ShadowStorage* maybe_shadow_storage) -> void;

  // Simulates a lazy copy a tensor that owns the shadow storage.
  //
  // maybe_shadow_storage comes from the tensor that is being lazily
  // copied. This may be null if this is the first lazy copy taking
  // place, or if the lazy copy is being performed on a tensor that
  // was part of the original tensors that share a view.
  //
  // The generation of the output will come from:
  // 1) maybe_shadow_storage->generation(), if non-null.
  // 2) this->default_shadow_storage_->generation(), if it is set.
  // 3) physical_generation_, i.e. 0, if this is the first lazy copy.
  auto simulate_lazy_copy(cow::ShadowStorage* maybe_shadow_storage)
      -> intrusive_ptr<cow::ShadowStorage>;

 private:
  // Guards all the state.
  std::mutex mtx_;
  // How many writes have been applied to the storage.
  std::uint64_t physical_generation_ = 0;
  // The shadow storage to use for any tensors that don't have
  // one. This situation is common, and will be true for tensors and
  // views thereof created before any copy on writes.
  //
  // Note: this would like to be std::optional, but it can't be
  // because of torchdeploy incompatibility.
  //
  // See https://github.com/pytorch/multipy/issues/312
  c10::optional<cow::ShadowStorageNonIntrusive> default_shadow_storage_;
};

} // namespace c10::impl::cow
