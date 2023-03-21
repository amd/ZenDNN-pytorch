#include <c10/core/impl/copy_on_write_spy.h>

#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/copy_on_write_simulator.h>
#include <c10/util/Exception.h>

#include <cassert>

namespace c10::impl {

/* static */ auto CopyOnWriteSpy::get_generation(Storage const& storage)
    -> std::uint64_t {
  assert(storage);
  return storage.unsafeGetStorageImpl()
      ->copy_on_write_state_.storage_generation();
}

/* static */ auto CopyOnWriteSpy::get_simulator(TensorImpl const& tensor)
    -> CopyOnWriteSimulator const* {
  return tensor.copy_on_write_simulator_.get();
}

} // namespace c10::impl
