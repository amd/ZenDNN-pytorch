#include <c10/core/StorageImpl.h>

namespace c10 {

intrusive_ptr<impl::CopyOnWriteSimulator> StorageImpl::simulate_copy_on_write(
    impl::CopyOnWriteSimulator* simulator) {
  return copy_on_write_state_.simulate_copy_on_write(simulator);
}

void StorageImpl::maybe_bump_copy_on_write_generation(
    impl::CopyOnWriteSimulator* simulator) {
  copy_on_write_state_.maybe_bump(simulator);
}

} // namespace c10
