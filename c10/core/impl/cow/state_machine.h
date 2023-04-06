#pragma once

#include <c10/core/impl/cow/shadow_storage.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>

namespace c10::impl::cow {

// Responsible for managing the copy-on-write simulation state
// machine.
//
// This type manages the following transition:

// 1) A tensor is created. It has a StateMachine instance that is in
//    the StateId::initial state. There are no shadow storages. This tensor and
//    any views created from it will have a null shadow storage.
//
// 2) A lazy copy is created, e.g. from Tensor::reshape(). This
//    transitions from StateId::initializing and then finally to
//    StateId::active. At this point a StateMachine::Impl has been
//    allocated on the heap.
//
//    The Tensor that was the source of the lazy copy, and any views
//    created from it, will still have a null shadow storage pointer,
//    but its actual shadow storage will be in
//    Impl::default_shadow_storage.
class C10_API StateMachine {
 public:
  /** @see cow::ShadowStorage::Generation */
  using Generation = cow::ShadowStorage::Generation;

  // Constructs an instance in the "initial" state.
  StateMachine();

  // Gets the current generation of the physical storage if
  // initialized.
  auto physical_generation() -> c10::optional<Generation>;

  // Bumps the generation if the shadow storage is non-null. If
  // non-null, shadow storage must be the result of a previous call to
  // simulate_copy_on_write on this instance.
  //
  // This will be called whenever we track a write to a Tensor. We
  // will call this on the Tensor receiving the mutation, and it will
  // propagate through its storage to here. That tensor will send its
  // shadow storage here, but it will be null if it was the original
  // tensor or a view derived from it.
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
  // How many writes have been applied to the storage.
  //
  // -1 is used as a sentinel to indicate that this has not been
  // initialized. We do this so that in the common path where there
  // are no outstanding lazy copies, we only need to pay the cost of
  // an uncontested atomic read.
  std::atomic<Generation> physical_generation_;
  // The shadow storage to use for any tensors that don't have
  // one. This situation is common, and will be true for tensors and
  // views thereof created before any copy on writes.
  cow::ShadowStorage default_shadow_storage_;
};

} // namespace c10::impl::cow
