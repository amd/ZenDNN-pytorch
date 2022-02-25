#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <type_traits>
#include <vector>

#include <c10/macros/Macros.h>

namespace torch {
namespace profiler {
namespace impl {

// ============================================================================
// == AppendOnlyList ==========================================================
// ============================================================================
//   During profiling we have a very predictable access pattern: we only append
// to the end of the container. We can specialize and outperform both
// std::vector (which must realloc) and std::deque (which performs a double
// indirection), and this class of operation is sufficiently important to the
// profiling hot path to warrant specializing:
//   https://godbolt.org/z/rTjozf1c4
//   https://quick-bench.com/q/mmfuu71ogwaiULDCJyHdKnHZms4    (Prototype #1, int)
//   https://quick-bench.com/q/5vWDW6jjdXVdoffev2zst8D09no    (Prototype #1, int pair)
//   https://quick-bench.com/q/IfEkfAQMeJSNBA52xtMP6Agcl-Q    (Prototype #2, int pair)
//   https://quick-bench.com/q/wJV2lKmuXL4XyGJzcI5hs4gEHFg    (Prototype #3, int pair)
//   https://quick-bench.com/q/xiO8ZaBEkYRYUA9dFrMuPLlW9fo    (Full impl, int pair)
// AppendOnlyList has 2x lower emplace overhead compared to more generic STL
// containers.
template <typename array_t>
class AppendOnlyListBase {
 public:
  using T = typename array_t::value_type;
  static constexpr size_t N = std::tuple_size<array_t>::value;
  static_assert(N > 0, "Block cannot be empty.");
  static_assert(std::is_base_of<std::array<T, N>, array_t>::value);

  AppendOnlyListBase() : buffer_end_{buffer_.before_begin()} {}
  AppendOnlyListBase (const AppendOnlyListBase&) = delete;
  AppendOnlyListBase& operator= (const AppendOnlyListBase&) = delete;

  size_t size() const {
    return n_blocks_ * N + (size_t)(next_ - end_);
  }

  template <class... Args>
  void emplace_back(Args... args) {
    maybe_grow();
    *next_++ = {args...};
  }

  void clear() {
    buffer_.clear();
    buffer_end_ = buffer_.begin();
    n_blocks_ = 0;
    next_ = nullptr;
    end_ = nullptr;
  }

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;

    Iterator(std::forward_list<array_t>& buffer, const size_t size)
      : block_{buffer.begin()}, size_{size} {}

    // End iterator.
    Iterator() = default;

    reference operator*() const {
      TORCH_INTERNAL_ASSERT(current_ptr() != nullptr);
      return *current_ptr();
    }

    pointer operator->() const {
      TORCH_INTERNAL_ASSERT(current_ptr() != nullptr);
      return current_ptr();
    }

    // Prefix increment
    Iterator& operator++() {
      if (!(++current_ % N)) {
        block_++;
      }
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

    friend bool operator== (const Iterator& a, const Iterator& b) { return a.current_ptr() == b.current_ptr(); };
    friend bool operator!= (const Iterator& a, const Iterator& b) { return a.current_ptr() != b.current_ptr(); };

    std::pair<array_t*, size_t> address() const {
      if (current_ >= size_){
        return {nullptr, 0};
      }
      return {&(*block_), current_ % N};
    }

   private:
    T* current_ptr() const {
      auto a = address();
      return a.first->data() + a.second;
    }

    typename std::forward_list<array_t>::iterator block_;
    size_t current_ {0};
    size_t size_ {0};
  };

  Iterator begin() { return Iterator(buffer_, size()); }
  Iterator end()   { return Iterator(); }
  // TODO: cbegin and cend()

 // TODO: make private
 protected:
  void maybe_grow() {
    if (C10_UNLIKELY(next_ == end_)) {
      buffer_end_ = buffer_.emplace_after(buffer_end_);
      n_blocks_++;
      next_ = buffer_end_->data();
      end_ = next_ + N;
    }
  }

  std::forward_list<array_t> buffer_;
  typename std::forward_list<array_t>::iterator buffer_end_;
  size_t n_blocks_ {0};
  T* next_ {nullptr};
  T* end_ {nullptr};
};

template <typename T, size_t N>
class AppendOnlyList : public AppendOnlyListBase<std::array<T, N>> {};

// ============================================================================
// == IncrementalCounter ======================================================
// ============================================================================
//   Often we need to store values which may fall in a large range, but in
// practice are small increments from a prior value. (e.g. timestamps or
// sequence numbers.) Storing such values is doubly wasteful: first, we waste
// most of the bits most of the time. Second, including large values in small
// structs wastes space due to alignment. Consider the following:
// ```
// struct Foo {
//   uint64_t x_;
//   enum Y : uint8_t {kFirst, kSecond} y_;
// };
// static_assert(sizeof(Foo) == 16);
// ```
//   We not only pay eight bytes for `x_`, but `Foo` is now also 8 byte aligned
// which means `y_` bumps it up to 16. This means that for every bit we shave
// off of `x_`, we reduce the size of `Foo` by two bits.
//
//   An obvious concern, however, is that we have to encode large values down
// to small ones, and even if our assumption of small deltas holds such
// encoding is not free. Fortunately, in practice it seems that as long as
// the rate of overflow is low this numeric compression does not incur a
// meaningful slowdown:
//   https://godbolt.org/z/sxs9d5f8v
//   https://quick-bench.com/q/lLi5I9UAdlAEY7mfHGLkIDY77EM
template <typename small_t, typename large_t, bool monotonic>
struct IncrementalCounter {
  static_assert(std::is_integral<large_t>::value, "Counter can only encode integers.");
  static_assert(std::is_integral<small_t>::value, "Counter can only encode integers.");
  static_assert(sizeof(large_t) > sizeof(small_t), "IncrementalCounter should compress values.");
  static_assert(
    monotonic || (std::is_signed<large_t>::value && std::is_signed<small_t>::value), 
    "Because we track deltas, integer types must be signed for non-monotonic functions.");

  static constexpr large_t x0 = std::numeric_limits<large_t>::min();
  static constexpr small_t OVERFLOW = std::numeric_limits<small_t>::max();
  static constexpr small_t UNDERFLOW = std::numeric_limits<small_t>::min();

  small_t encode(const large_t x) {
    large_t dx = x - prior_x_;
    prior_x_ = x;
    if (C10_UNLIKELY((dx >= OVERFLOW) | (!monotonic && (dx <= UNDERFLOW)))) {
      fallback_.emplace_back(x);
      dx = OVERFLOW;
    }
    return dx;
  }

  auto makeDecoder() {
    return [fallback_it=fallback_.begin(), x=x0](small_t xi) mutable {
      auto x_old = x;
      x = (xi == OVERFLOW) ? *(fallback_it++) : x + xi;
      TORCH_INTERNAL_ASSERT(!monotonic || x >= x_old, x, " < ", x_old);
      return x;
    };
  }

 private:
  large_t prior_x_ {x0};
  AppendOnlyList<large_t, 16> fallback_;
};

} // namespace impl
} // namespace profiler
} // namespace torch
