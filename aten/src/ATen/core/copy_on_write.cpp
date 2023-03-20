#include <ATen/core/copy_on_write.h>

#include <ATen/core/TensorBase.h>
#include <c10/core/impl/copy_on_write_peer.h>

namespace at {

auto simulate_materialize_copy_on_write(TensorBase const& tensor) -> void {
  if (!tensor.has_storage()) {
    return;
  }

  c10::impl::CopyOnWritePeer::bump_copy_on_write_generation(*tensor.unsafeGetTensorImpl());
}

} // namespace at
