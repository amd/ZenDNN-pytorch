#include <c10/core/impl/cow/try_ensure.h>

#include <c10/core/StorageImpl.h>
#include <c10/core/impl/cow/context.h>
#include <c10/core/impl/cow/deleter.h>
#include <c10/util/Exception.h>
#include <c10/util/UniqueVoidPtr.h>

#include <memory>
#include <mutex>

namespace c10::impl {

namespace {

// A std::unique_ptr deleter that is expected to never execute.
struct AssertFalseDeleter {
  template <typename T>
  auto operator()(T* /* ptr */) const -> void {
    TORCH_INTERNAL_ASSERT(false); // expected to never be called
  }
};

// Wraps a DataPtr with a copy-on-write DataPtr.
auto make_data_ptr(at::DataPtr const& data_ptr, cow::Context& ctx)
    -> at::DataPtr {
  return at::DataPtr(
      data_ptr.get(), &ctx, cow::delete_context, data_ptr.device());
}

} // namespace

auto C10_API cow::try_ensure(StorageImpl& storage)
    -> c10::intrusive_ptr<StorageImpl> {
  at::DataPtr& data_ptr = storage.mutable_data_ptr();

  // There are three possible circumstances:
  //
  // 1) the storage does not already have a copy on write context. In
  //    this case there can be no blind aliases to the storage impl:
  //    they all will be public aliases and the user is expected to
  //    synchronize manually.
  //
  //    No locking is required in this case.
  //
  // 2) the storage has a context that is not the copy on write
  //    context. This is not supported, so we just return null.
  //
  //    No locking is required in this case.
  //
  // 3) there is already a copy on write context on the storage. There
  //    is a potential race condition with a blind alias (i.e. an
  //    alias that the user is not required to synchronize
  //    with). Because our input storage is bound to a live reference
  //    to the data, we know that it isn't going away. A blind alias
  //    could be copying from it right now, but we will grab the
  //    context's mutex to protect us.
  //
  //    We do not need to lock in this case either, because we're just
  //    wrapping a context that we know isn't going away.

  DataPtr new_data_ptr;

  if (data_ptr.get() == data_ptr.get_context()) {
    // Case 1) We have a simple data pointer: wrap it.
    std::unique_ptr<void, DeleterFnPtr> original_ctx = data_ptr.move_context();
    TORCH_INTERNAL_ASSERT(original_ctx.get() == data_ptr.get());

    std::unique_ptr<cow::Context, AssertFalseDeleter> ctx(
        new cow::Context(std::move(original_ctx)));
    // Update this storage to the new copy on write context.
    storage.set_data_ptr_noswap(make_data_ptr(data_ptr, *ctx));

    // Save this for the result.
    ctx->increment_refcount();
    new_data_ptr = make_data_ptr(data_ptr, *ctx.release());
  } else if (data_ptr.get_deleter() != cow::delete_context) {
    // Case 2) There is a context and it's not copy-on-write. Nothing
    // we can do here.
    return nullptr;
  } else {
    // Case 3): there is already a copy on write context. Just return a
    // new storage impl.
    TORCH_INTERNAL_ASSERT(data_ptr.get_deleter() == cow::delete_context);
    // Already a copy-on-write storage.
    auto& ctx = *data_ptr.cast_context<cow::Context>(cow::delete_context);
    ctx.increment_refcount();
    new_data_ptr = make_data_ptr(data_ptr, ctx);
  }

  return make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      storage.sym_nbytes(),
      std::move(new_data_ptr),
      storage.allocator(),
      storage.resizable());
}

} // namespace c10::impl
