#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <ATen/core/function_schema.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/overloaded.h>
#include <c10/util/variant.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/kineto_shim.h>

namespace torch {
namespace profiler {
namespace graph {
using namespace torch::profiler::impl;

enum class EventType : uint8_t {
  TorchOp = 0,
  PythonFunction,
  PythonNNModule,
  PythonCFunction,
  Allocation,
  UserAnnotation,
  GPUMemcpy,
  GPUMemset,
  ConcurrentKernal,
  CudaRuntime
  // TODO: PythonTorchBinding from `torch.ops`, rather than representing
  //       `torch.foo` calls with PythonCFunction.
  // TODO: CUDA_OOM
};

// ============================================================================
// == Metadata: EventType specific fields =====================================
// ============================================================================
struct Event;

struct TORCH_API MetadataBase {
  virtual ~MetadataBase() {};
  std::shared_ptr<const Event> event() const {
    return event_.lock();
  }
  std::weak_ptr<const Event> event_; // Owns this metadata
};

//  ----------------------------------
// |  PyTorch Ops                     |
//  ----------------------------------
struct TORCH_API TensorSummary {
  TensorSummary(void* ptr, c10::Device device, c10::ScalarType dtype, std::vector<int64_t> sizes)
    : ptr_{ptr}, device_{device}, dtype_{dtype}, sizes_{sizes} {}

  void* ptr_;
  c10::Device device_;
  c10::ScalarType dtype_;
  std::vector<int64_t> sizes_;

  // We need the entire graph before we can assign these.
  int64_t identity_{-1};
};

using TensorSummaryPtr = std::shared_ptr<TensorSummary>;
struct TORCH_API UndefinedTensorSummary {};
using TensorListSummary =
    std::vector<c10::variant<TensorSummaryPtr, UndefinedTensorSummary>>;
struct TORCH_API OtherSummary {};

using ValueSummary = c10::variant<
    TensorSummaryPtr,
    UndefinedTensorSummary,
    TensorListSummary,
    c10::Scalar,
    OtherSummary>;

struct TORCH_API TorchOp : public MetadataBase {
  TorchOp(const std::string& op_name, const at::RecordScope scope)
    : op_name_{op_name}, scope_{scope} {}

  TorchOp(const c10::FunctionSchema& schema, const at::RecordScope scope)
    : op_name_{schema.name()}, scope_{scope}, schema_{schema} {}

  std::string op_name_;
  at::RecordScope scope_;
  c10::optional<int64_t> sequence_number_;
  c10::optional<c10::FunctionSchema> schema_;
  c10::optional<std::vector<ValueSummary>> inputs_;
  c10::optional<std::vector<ValueSummary>> outputs_;
  // TODO: Map to convert pointers into Tensor identities and versions.
  // TODO: Map fwd and bwd pass.
};

//  ----------------------------------
// |  Python                          |
//  ----------------------------------
struct TORCH_API PyFrameState {
  int line_no_;
  at::StringView filename_;
  at::StringView funcname_;
};

struct TORCH_API PyCallBase : public MetadataBase {
  PyCallBase(PyFrameState caller, uint8_t python_tid)
      : caller_{caller}, python_tid_{python_tid} {}

  PyFrameState caller_;
  uint8_t python_tid_;
};

struct TORCH_API PyCall : public PyCallBase {
  PyCall(PyFrameState function, PyFrameState caller, uint8_t python_tid)
      : PyCallBase(caller, python_tid), function_{function} {}

  PyFrameState function_;
};

struct TORCH_API PyNNModuleForwardCall : public PyCallBase {
  PyNNModuleForwardCall(
      size_t id,
      at::StringView cls_name,
      PyFrameState caller,
      uint8_t python_tid)
      : PyCallBase(caller, python_tid), id_{id}, cls_name_{cls_name} {}

  size_t id_;
  at::StringView cls_name_;
};

struct TORCH_API PyCCall : public PyCallBase {
  PyCCall(at::StringView name, PyFrameState caller, uint8_t python_tid)
      : PyCallBase(caller, python_tid), name_{name} {}

  at::StringView name_;
};

//  ----------------------------------
// |  Memory                          |
//  ----------------------------------
struct TORCH_API Allocation : public MetadataBase {
  Allocation(
      void* ptr,
      int64_t alloc_size,
      int64_t total_allocated,
      int64_t total_reserved)
      : ptr_{ptr},
        alloc_size_{alloc_size},
        total_allocated_{total_allocated},
        total_reserved_{total_reserved} {}

  void* ptr_;
  int64_t alloc_size_;
  int64_t total_allocated_;
  int64_t total_reserved_;
};

//  ----------------------------------
// |  Kineto (external events)        |
//  ----------------------------------
struct TORCH_API KinetoEvent : public MetadataBase {
  KinetoEvent(std::string name, EventType type) : name_{name}, type_{type} {}

  std::string name_;
  EventType type_;
};

using Metadata = c10::variant<
    TorchOp,
    PyCall,
    PyNNModuleForwardCall,
    PyCCall,
    Allocation,
    KinetoEvent>;

// ============================================================================
// == Event: universal fields =================================================
// ============================================================================
struct TORCH_API Event : std::enable_shared_from_this<Event> {
  [[nodiscard]] static std::shared_ptr<Event> create(
      const c10::Device device,
      const uint64_t tid,
      const Metadata metadata) {
    auto out = std::shared_ptr<Event>(new Event(device, tid, metadata));
    c10::visit([out](auto& m) { m.event_ = out; }, out->metadata_);
    return out;
  }

  EventType type() const;
  void setParent(std::shared_ptr<Event>& parent);

  bool operator<(const Event& other) const {
    return start_time_ < other.start_time_;
  }

  static constexpr time_t UNSET_TIME = std::numeric_limits<time_t>::min();

  time_t start_time_{UNSET_TIME};
  time_t end_time_{UNSET_TIME};
  c10::Device device_;
  correlation_id_t correlation_id_{no_correlation_id};
  uint64_t tid_;
  Metadata metadata_;
  std::weak_ptr<const Event> parent_;
  std::vector<std::shared_ptr<const Event>> children_;

 private:
  Event(const c10::Device device, const uint64_t tid, const Metadata metadata)
      : device_{device}, tid_{tid}, metadata_{metadata} {}
};

// ============================================================================
// == Graph ===================================================================
// ============================================================================
class TORCH_API Graph {
 public:
  Graph() = default;
  Graph(std::deque<std::shared_ptr<Event>>&& events);
  std::vector<std::shared_ptr<const Event>>& head_nodes() {
    return head_nodes_;
  }

  void emitKinetoHostEvents(
      torch::profiler::impl::kineto::TraceWrapper& cpu_trace);

  void addKinetoDeviceEvents(
      torch::profiler::impl::kineto::ActivityTraceWrapper& trace);

  void finalize();

 private:
  friend std::string toString(const Graph& g);
  std::vector<std::shared_ptr<const Event>> head_nodes_;
  std::deque<std::shared_ptr<Event>> all_nodes_;
};

TORCH_API EventType toEventType(const Metadata& m);
TORCH_API std::string toString(const EventType& e);
TORCH_API std::string toString(const TensorSummary& s);
TORCH_API std::string toString(const UndefinedTensorSummary& s);
TORCH_API std::string toString(const TensorListSummary& s);
TORCH_API std::string toString(const OtherSummary& s);
TORCH_API std::string toString(const ValueSummary& s);
TORCH_API std::string toString(const TorchOp& s);
TORCH_API std::string toString(const PyFrameState& s);
TORCH_API std::string toString(const PyCall& c);
TORCH_API std::string toString(const PyNNModuleForwardCall& c);
TORCH_API std::string toString(const PyCCall& c);
TORCH_API std::string toString(const Allocation& a);
TORCH_API std::string toString(const KinetoEvent& e);
TORCH_API std::string toString(const Metadata& m);
TORCH_API std::string toString(const Event& e);
TORCH_API std::string toString(const Graph& g);

} // namespace graph
} // namespace profiler
} // namespace torch
