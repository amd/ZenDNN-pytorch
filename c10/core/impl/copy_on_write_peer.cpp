#include <c10/core/impl/copy_on_write_peer.h>

#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/copy_on_write_simulator.h>
#include <c10/util/Exception.h>

#include <cassert>

namespace c10::impl {

/* static */ auto CopyOnWritePeer::get_generation(Storage const& storage)
    -> std::uint32_t {
  assert(storage);
  return storage.unsafeGetStorageImpl()->generation_;
}

/* static */ auto CopyOnWritePeer::bump_copy_on_write_generation(
    TensorImpl& tensor) -> void {
  if (!tensor.has_storage()) {
    return;
  }

  StorageImpl& storage = *tensor.storage().unsafeGetStorageImpl();
  if (tensor.copy_on_write_simulator_->storage_generation() !=
      storage.generation_) {
    TORCH_WARN_ONCE(
        "You have written through to both aliases created by calling "
        "reshape(). In the future, reshape() will never create a view but will "
        "instead return a lazily copied tensor. If you wish to preserve the "
        "aliasing properties, you should rewrite your reshape() as a view().");
  }
  tensor.copy_on_write_simulator_->bump_storage_generation();
  ++storage.generation_;
}

/* static */ auto CopyOnWritePeer::get_simulator(TensorImpl const& tensor)
    -> CopyOnWriteSimulator const& {
  return *tensor.copy_on_write_simulator_;
}

} // namespace c10::impl
