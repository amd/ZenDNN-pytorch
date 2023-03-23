#include <c10/core/impl/cow/simulator.h>

#include <cassert>
#include <limits>

namespace c10::impl::cow {

template <bool intrusive>
SimulatorImpl<intrusive>::SimulatorImpl(
    std::uint64_t storage_generation) noexcept
    : storage_generation_(storage_generation) {}

template <bool intrusive>
auto SimulatorImpl<intrusive>::storage_generation() noexcept -> std::uint64_t {
  std::lock_guard<std::mutex> lock(mtx_);
  return storage_generation_;
}

template <bool intrusive>
auto SimulatorImpl<intrusive>::bump_storage_generation() noexcept
    -> std::uint64_t {
  std::lock_guard<std::mutex> lock(mtx_);
  assert(storage_generation_ != std::numeric_limits<std::uint64_t>::max());
  return ++storage_generation_;
}

template class SimulatorImpl<true>;
template class SimulatorImpl<false>;

} // namespace c10::impl::cow
