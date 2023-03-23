#include <c10/core/impl/cow/spy.h>

#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

#include <cassert>

namespace c10::impl::cow {

/* static */ auto Spy::get_generation(Storage const& storage) -> std::uint64_t {
  assert(storage);
  return storage.unsafeGetStorageImpl()
      ->copy_on_write_state_.storage_generation();
}

/* static */ auto Spy::get_simulator(TensorImpl const& tensor)
    -> Simulator const* {
  return tensor.copy_on_write_simulator_.get();
}

} // namespace c10::impl::cow
