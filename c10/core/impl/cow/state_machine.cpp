#include <c10/core/impl/cow/state_machine.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <mutex>

namespace c10::impl {

class cow::StateMachine::Impl {
 public:
  /** @see cow::StateMachine::maybe_bump */
  auto maybe_bump(cow::ShadowStorage* maybe_shadow_storage) -> void;

  /** @see cow::StateMachine::simulate_lazy_copy */
  auto simulate_lazy_copy(cow::ShadowStorage* maybe_shadow_storage)
      -> intrusive_ptr<cow::ShadowStorage>;

 private:
  friend class cow::StateMachine;

  // Guards all the state.
  std::mutex mtx_;
  // How many writes have been applied to the storage.
  cow::StateMachine::Generation physical_generation_ = 0;
  // The shadow storage to use for any tensors that don't have
  // one. This situation is common, and will be true for tensors and
  // views thereof created before any copy on writes.
  cow::ShadowStorageNonIntrusive default_shadow_storage_{physical_generation_};
};

cow::StateMachine::StateMachine() : state_rep_(StateId::initial) {}

cow::StateMachine::~StateMachine() {
  // If we created an impl, we are responsible for clenaing this up
  // here.
  Rep state_rep = state_rep_;
  switch (state_rep.as_state_id()) {
    case StateId::initial:
      // Nothing to do.
      break;

    case StateId::initializing:
      TORCH_INTERNAL_ASSERT(false); // we should not race with the destructor

    case StateId::active:
      delete &state_rep.as_state_machine_impl();
  }
}

auto cow::StateMachine::physical_generation() -> Generation {
  Impl* impl = maybe_get_impl();
  return impl != nullptr ? impl->physical_generation_ : 0;
}

auto cow::StateMachine::maybe_bump(cow::ShadowStorage* maybe_shadow_storage)
    -> void {
  Impl* impl = maybe_get_impl();
  if (impl == nullptr) {
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

  impl->maybe_bump(maybe_shadow_storage);
}

auto cow::StateMachine::simulate_lazy_copy(
    cow::ShadowStorage* maybe_shadow_storage)
    -> intrusive_ptr<cow::ShadowStorage> {
  return ensure_initialized().simulate_lazy_copy(maybe_shadow_storage);
}

auto cow::StateMachine::maybe_get_impl() -> cow::StateMachine::Impl* {
  return get_impl([] { return nullptr; });
}

auto cow::StateMachine::ensure_initialized() -> cow::StateMachine::Impl& {
  return *get_impl([this]() -> c10::optional<cow::StateMachine::Impl*> {
    Rep initial_rep(StateId::initial);
    Rep initializing_rep(cow::StateMachine::StateId::initializing);
    // If we fail to exchange, it means that someone else is
    // initializing.
    if (!state_rep_.compare_exchange_strong(initial_rep, initializing_rep)) {
      return c10::nullopt;
    }

    auto impl = std::make_unique<cow::StateMachine::Impl>();
    Rep new_rep(*impl);
    TORCH_INTERNAL_ASSERT(
        new_rep.as_state_id() == cow::StateMachine::StateId::active);
    // We won the race to initializing: we will be the only thread
    // that may transition to active.
    TORCH_INTERNAL_ASSERT(
        state_rep_.compare_exchange_strong(initializing_rep, new_rep));
    return c10::make_optional(impl.release()); // owned by state_rep_ now
  });
}

template <typename OnInitialFunc>
auto cow::StateMachine::get_impl(const OnInitialFunc& on_initial_func)
    -> StateMachine::Impl* {
  bool seen_initializing = false;
  do {
    Rep state_rep = state_rep_;
    switch (state_rep.as_state_id()) {
      case StateId::initial: {
        TORCH_INTERNAL_ASSERT(!seen_initializing);
        c10::optional<Impl*> impl = on_initial_func();
        if (!impl.has_value()) {
          continue;
        }
        return *impl;
      }

      case StateId::initializing:
        seen_initializing = true;
        // Wait for initialization.
        continue;

      case StateId::active:
        return &state_rep.as_state_machine_impl();
    }
  } while (true);
}

cow::StateMachine::Rep::Rep(cow::StateMachine::StateId state_id)
    : rep_(static_cast<std::uintptr_t>(state_id)) {
  TORCH_INTERNAL_ASSERT(as_state_id() == state_id);
}

namespace {

// Implementations of std::bit_cast() from C++ 20.
//
// This is a less sketchy version of reinterpret_cast.
//
// See https://en.cppreference.com/w/cpp/numeric/bit_cast for more
// information as well as the source of our implementations.
template <class To, class From>
std::enable_if_t<
    sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_trivially_constructible_v<To>,
      "This implementation additionally requires "
      "destination type to be trivially constructible");

  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

} // namespace

cow::StateMachine::Rep::Rep(Impl& impl)
    : rep_(bit_cast<std::uintptr_t>(&impl)) {
  TORCH_INTERNAL_ASSERT(as_state_id() == StateId::active);
}

auto cow::StateMachine::Rep::as_state_id() const noexcept
    -> cow::StateMachine::StateId {
  return StateId{static_cast<int>(rep_ & 0b111)};
}

auto cow::StateMachine::Rep::as_state_machine_impl()
    -> cow::StateMachine::Impl& {
  TORCH_INTERNAL_ASSERT(as_state_id() == StateId::active);
  TORCH_INTERNAL_ASSERT((rep_ & 0b111) == 0); // no alignment bits are modified
  return *bit_cast<cow::StateMachine::Impl*>(rep_);
}

auto cow::StateMachine::Rep::operator==(const Rep& that) const noexcept
    -> bool {
  return rep_ == that.rep_;
}

namespace {

// Applies a function to whichever shadow storage is active.
//
// Note that we templatize on the shadow storage types because they
// may or may not be const. The bare types will always be
// ShadowStorage and ShadowStorageNonIntrusive.
template <typename ShadowStorage, typename DefaultShadowStorage, typename Func>
auto apply_to_active_shadow_storage(
    ShadowStorage* shadow_storage,
    DefaultShadowStorage& default_shadow_storage,
    const Func& func) {
  if (shadow_storage != nullptr) {
    return func(*shadow_storage);
  }
  return func(default_shadow_storage);
}

} // namespace

auto cow::StateMachine::Impl::maybe_bump(
    cow::ShadowStorage* maybe_shadow_storage) -> void {
  std::lock_guard<std::mutex> lock(mtx_);
  Generation physical_generation = ++physical_generation_;
  Generation shadow_generation = apply_to_active_shadow_storage(
      maybe_shadow_storage, default_shadow_storage_, [](auto& shadow_storage) {
        return shadow_storage.bump_generation();
      });

  TORCH_INTERNAL_ASSERT(shadow_generation <= physical_generation);
  if (shadow_generation != physical_generation) {
    TORCH_WARN_ONCE(
        "You have written through to both aliases created by calling "
        "reshape(). In the future, reshape() will never create a view but will "
        "instead return a lazily copied tensor. If you wish to preserve the "
        "aliasing properties, you should rewrite your reshape() as a view().");
  }
}

auto cow::StateMachine::Impl::simulate_lazy_copy(
    cow::ShadowStorage* maybe_shadow_storage)
    -> intrusive_ptr<cow::ShadowStorage> {
  // We grab the lock here unconditionally. No need to check the
  // current state first.
  std::lock_guard<std::mutex> lock(mtx_);
  return make_intrusive<cow::ShadowStorage>(apply_to_active_shadow_storage(
      maybe_shadow_storage,
      default_shadow_storage_,
      [](auto const& shadow_storage) { return shadow_storage.generation(); }));
}

} // namespace c10::impl
