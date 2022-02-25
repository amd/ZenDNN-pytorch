#pragma once

#include <array>
#include <deque>
#include <type_traits>
#include <utility>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/record_function.h>
#include <c10/util/python_stub.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/graph.h>
#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
using interned_t = uint32_t;

namespace impl {
// ============================================================================
// == Timestamp encoding ======================================================
// ============================================================================
//   When we profile a program we expect to see many calls in rapid succession.
// However, in rare cases gaps are long; we might be waiting on data, a network
// call, or a long running kernel. We have to represent these gaps (after all,
// when we profile we're generally looking for gaps), however it is very
// wasteful to use large numbers (e.g. 64 bit integers) to represent a long
// sequence of small numbers. (Particularly since it would interfere with bit
// packing for the core `Event` type.) So instead we store incremental times
// and decode them back during post processing.
class IncrementalClock {
 public:
  uint16_t operator()() {
    return counter_.encode(getApproximateTime());
  }

  auto makeDecoder() {
    return counter_.makeDecoder();
  }

 private:
  IncrementalCounter<uint16_t, approx_time_t, /*monotonic=*/true> counter_;
};

// ============================================================================
// == EventQueue: AppendOnlyList with correlation IDs =========================
// ============================================================================
enum class EventTag : uint8_t {
  //                  CompressedEvent.payload_
  OpCall = 0, //        fn.scope()
  BITPACK_OpCall_1, //  ...
  BITPACK_OpCall_2, //  ...
  BITPACK_OpCall_3, //  ...
  BITPACK_OpCall_4, //  ...
  BITPACK_OpCall_5, //  ...
  BITPACK_OpCall_6, //  ...
  BITPACK_OpCall_7, //  ...
  PyCall, //            tid (for Python thread.)
  PyModuleCall, //      tid (for Python thread.)
  PyCCall, //           tid (for Python thread.)
  OpReturn, //          (Unused.)
  PyReturn, //          tid (for Python thread. Possibly redundant.)
  PyCReturn, //         tid (for Python thread. Possibly redundant.)
  Allocate, //          Device Type
};

struct WrappedEventTag {
  EventTag get() const {
    switch (tag_) {
      case EventTag::OpCall:
      case EventTag::BITPACK_OpCall_1:
      case EventTag::BITPACK_OpCall_2:
      case EventTag::BITPACK_OpCall_3:
      case EventTag::BITPACK_OpCall_4:
      case EventTag::BITPACK_OpCall_5:
      case EventTag::BITPACK_OpCall_6:
      case EventTag::BITPACK_OpCall_7:
        return EventTag::OpCall;
      default:
        return tag_;
    }
  }

  static constexpr auto raw_op_call = (uint8_t)EventTag::OpCall;
  static constexpr uint8_t bwd_mask = 0b001;
  static constexpr uint8_t schema_mask = 0b010;
  static constexpr uint8_t seq_nr_mask = 0b100;

  static EventTag packOpCall(bool is_bwd, bool has_schema, bool has_seq_nr) {
    auto low_bits = (is_bwd * bwd_mask) |
                    (has_schema * schema_mask) |
                    (has_seq_nr * seq_nr_mask);
    return static_cast<EventTag>(raw_op_call + low_bits);
  }

  bool decode(const uint8_t mask) const {
    TORCH_INTERNAL_ASSERT(get() == EventTag::OpCall, "Mask: ", (int)mask);
    auto low_bits = (uint8_t)tag_ - raw_op_call;
    return low_bits & mask;
  }

  bool is_bwd() const {
    return decode(bwd_mask);
  }

  bool has_schema() const {
    return decode(schema_mask);
  }

  bool has_sequence_number() const {
    return decode(seq_nr_mask);
  }

  EventTag tag_;
};

struct CompressedEvent {
  // ThreadLocalSubqueue only needs three bytes to store what it needs
  // (`tag_` & `dt_`) but because of alignment on `dt_` we will still pay for
  // the fourth byte. Therefore we expose an opaque `payload_` field that is
  // used to stash tag-specific data.
  WrappedEventTag tag_;
  uint8_t payload_;
  uint16_t dt_; // Time since last event. (Or max_value if overflow.)
};

struct UncompressedEvent {
  UncompressedEvent(WrappedEventTag tag, correlation_id_t correlation_id, time_t t, uint64_t tid)
    : tag_{tag}, correlation_id_{correlation_id}, t_{t}, tid_{tid} {}

  WrappedEventTag tag_;
  correlation_id_t correlation_id_;
  time_t t_;
  uint64_t tid_;
  c10::optional<torch::profiler::graph::Metadata> metadata_;

  // Used to recreate the call graph.
  c10::optional<std::weak_ptr<UncompressedEvent>> begin_raw_event_;
  c10::optional<uint64_t> calling_tid_;
  c10::optional<std::shared_ptr<torch::profiler::graph::Event>> final_event_;
  bool finished_{false};
  
  // Tag specific events
  // OpCall
  c10::optional<uint64_t> forward_tid_;

  // OpReturn
  c10::optional<std::vector<torch::profiler::graph::ValueSummary>> outputs_;

  // PyCall, PyModuleCall, PyCCall, PyReturn, PyCReturn
  c10::optional<uint8_t> python_tid_;

  // Allocate
  c10::optional<c10::Device> alloc_device_;
  
  uint64_t calling_thread() {
    return calling_tid_.has_value() ? *calling_tid_ : tid_;
  }

  bool operator<(const UncompressedEvent& other) const {
    return t_ < other.t_;
  }
};

using raw_event_ptr_t = std::shared_ptr<UncompressedEvent>;

static constexpr size_t EventArraySize = 1024;
struct EventArray : public std::array<CompressedEvent, EventArraySize> {
  EventArray() : block_start_(nextCorrelationIDRange() * size()) {}
  static correlation_id_t nextCorrelationIDRange();

  correlation_id_t block_start_;
};

} // namespace impl
} // namespace profiler
} // namespace torch
namespace std {
template <>
struct tuple_size<torch::profiler::impl::EventArray>
    : std::integral_constant<size_t, torch::profiler::impl::EventArraySize> {};
} // namespace std
namespace torch {
namespace profiler {
namespace impl {

class EventQueue : public AppendOnlyListBase<EventArray> {
 public:
  // We need to assign a *unique* correlation ID to each op call so that we can
  // correlate with Kineto events. (e.g. GPU kernels) We encode the correlation
  // ID as: `block_start_ + position_in_block_`. This means that we only have
  // to do an atomic increment once per block creation, and we do not have to
  // explicitly store correlation_id since it is implicitly encoded just by
  // storing the event.
  correlation_id_t lastCorrelationID();
  static correlation_id_t correlationIDFor(const EventQueue::Iterator& i);
};

// ============================================================================
// == Input / Output tracking =================================================
// ============================================================================
class InputOutputEncoder {
 public:
  void push(const std::vector<c10::IValue>& values);
  std::deque<std::vector<torch::profiler::graph::ValueSummary>> materialize();

 private:
  enum class Tag {
    Tensor = 0,
    UndefinedTensor,
    TensorListBegin, // TODO: generalize to other lists.
    Scalar,
    Other,
    TERMINATOR
  };

  struct TensorMetadata {
    void* ptr_;
    c10::ScalarType dtype_;
    c10::DeviceType device_type_;
    c10::DeviceIndex device_index_;
    uint32_t dim_;
  };

  void push(const at::Tensor& t);
  void push(const c10::Scalar s);

  AppendOnlyList<Tag, 1024> tags_;
  AppendOnlyList<TensorMetadata, 1024> tensor_metadata_;
  AppendOnlyList<int64_t, 1024> tensor_sizes_;
  AppendOnlyList<c10::Scalar, 1024> scalars_;
};

// ============================================================================
// == SubqueueData: Storage implementation for ThreadLocalSubqueue encoding ===
// ============================================================================
struct Alloc {
  void* ptr_;
  int64_t alloc_size_ : 56;
  c10::DeviceIndex device_index_;
  int64_t total_allocated_;
  int64_t total_reserved_;
};

static_assert(sizeof(c10::DeviceType) == 1, "c10::DeviceType has grown.");
static_assert(
  std::is_same<c10::DeviceIndex, int8_t>::value,
  "Profiler must be updated to reflect the change in c10::DeviceIndex.");
static_assert(sizeof(Alloc) == 32, "Alloc is misaligned.");

struct TORCH_API SubqueueOpaqueFields {
  // All events
  EventQueue events_;
  IncrementalClock clock_;

  // PyTorch ops (TODO: pack this a bit better.)
  AppendOnlyList<c10::ErasedOperatorDefPtr, 1024> op_handles_;
  AppendOnlyList<std::string, 128> op_names_;
  AppendOnlyList<uint64_t, 1024> forward_thread_ids_;
  AppendOnlyList<int64_t, 1024> sequence_numbers_;
  AppendOnlyList<correlation_id_t, 64> return_corr_ids_;
  InputOutputEncoder op_inputs_and_outputs_;

  // Python functions
  AppendOnlyList<std::pair<interned_t, interned_t>, 1024> py_call_metadata_;

  // Allocations
  AppendOnlyList<Alloc, 1024> allocations_;
};

// ============================================================================
// == Python interface: (libtorch_python -> libtorch) =========================
// ============================================================================
struct PyCodeDescriptions {
  struct Module {
    size_t id_;
    at::StringView cls_name_;
  };

  std::vector<torch::profiler::graph::PyFrameState> frame_states_;
  std::vector<Module> modules_;
  std::vector<at::StringView> c_function_names_;
};

class RecordQueue;
using PyTracerStartFn = void (*)(torch::profiler::impl::RecordQueue*);
using PyTracerStopFn = PyCodeDescriptions (*)();

TORCH_API void registerPyFunctions(PyTracerStartFn start, PyTracerStopFn stop);

} // namespace impl
} // namespace profiler
} // namespace torch
