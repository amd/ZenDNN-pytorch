#pragma once

#include <c10/core/impl/cow/shadow_storage.h>
#include <c10/macros/Macros.h>
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

  // Responsible for cleaning up any implementation details.
  ~StateMachine();

  // Gets the current generation of the physical storage.
  auto physical_generation() -> Generation;

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
  class Impl;

  /** Gets the underlying Impl, returning null if uninitialized. */
  auto maybe_get_impl() -> Impl*;
  /** Gets the underlying Impl, initializing if uninitialized. */
  auto ensure_initialized() -> Impl&;

  /// Gets the Impl, invoking the callback if uninitialized.
  ///
  /// on_initial_func should return c10::optional<Impl*>. Return:
  ///  * nullopt if we should continue in the loop, e.g. waiting for
  ///    another thread to initialize
  ///  * nullptr if we should return no impl to the user
  ///  * a pointer to an implementation, if it exists
  template <typename OnInitialFunc>
  auto get_impl(const OnInitialFunc& on_initial_func) -> Impl*;

  // Identifies where a storage instance is in the lifecycle of
  // tracking simulated lazy copies.
  enum class StateId {
    // This is the initial state. It may not be returned to and
    // indicates that the storage has never had a lazy copy requested
    // of it.
    //
    // INVARIANT: maybe_get_impl() == nullptr
    initial = 2,

    // A transition state between initial and active. Any visitor that
    // sees this state will spin until it sees the active state.
    initializing = 1,

    // This is transitioned to when the first copy on write takes
    // place.
    //
    // Subtle! This has the value of 0 because we are using the
    // alignment bits of a pointer to hold this state id. In the case
    // where we have an actual impl, the state id will be active,
    // hence the alignment bits will be 0.
    active = 0,

    // TODO Consider adding an "inactive" state. An inactive state
    //      would occur when the last outstanding lazy copy goes away
    //      or has been materialized. At that point, we may stop
    //      tracking generations.
    //
    //      We may want to do this because we've already warned or
    //      because we only had a temporary lazy copy that shouldn't
    //      affect the performance of the program for the remainder of
    //      its lifetime.
    //
    //      DANGER! There could be outstanding shadow storages. For
    //      example, if the last set of tensors sharing a view
    //      remaining was not using the default shadow storage, it
    //      will continue holding the instances that they share.
    //
    //      There are a few potential solutions to this. One is for
    //      the state machine to be the exclusive owner of the shadow
    //      storages and for tensors to only hold weak pointers to
    //      them.
    //
    //      Another is for tensors to hold a storage id and an index
    //      and those can be invalidated by increasing the storage id
    //      whenever we transition to StateId::inactive. A nice
    //      property of this is that it would generalize the design
    //      decision around the "default shadow storage".
  };

  // Encapsulates the inline storage of the state machine. This is
  // morally equivalent to an untagged union of a
  // cow::StateMachine::StateId and a cow::StateMachine::Impl*.
  class Rep {
   public:
    // Initializes the state to a state_id.
    //
    // ensures(as_state_id() == state_id)
    Rep(StateId state_id);

    // Initializes the state to a state_id.
    //
    // ensures(&as_stat_machine_impl() == &impl)
    // ensures(as_state_id() == StateId::active)
    explicit Rep(Impl& impl);

    // Bitwise copy-constructor, required for std::atomic<>.
    Rep(Rep const& that) = default;

    // Gets the state id. Always valid.
    auto as_state_id() const noexcept -> StateId;
    // Gets the address to the state machine implementation.
    //
    // requires(as_state_id() == StateId::active)
    auto as_state_machine_impl() -> Impl&;

    // Bitwise equality. Required for atomic exchange.
    auto operator==(const Rep& that) const noexcept -> bool;

   private:
    // Holds the StateId in the three low bits. If the StateId is
    // active, then this also holds the address of a pointer to an
    // Impl.
    std::uintptr_t rep_;
  };

  // The current representation of the storage instance.
  std::atomic<Rep> state_rep_;
};

} // namespace c10::impl::cow
