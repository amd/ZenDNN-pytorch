#include <torch/csrc/profiler/graph.h>

#include <algorithm>
#include <deque>
#include <sstream>
#include <type_traits>

#ifdef USE_KINETO
#include <libkineto.h>
#endif

#include <fmt/format.h>

#include <ATen/core/Formatting.h>
#include <c10/macros/Macros.h>
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/kineto_shim.h>

namespace torch {
namespace profiler {
namespace graph {

namespace {
auto ptr_sort(
    const std::shared_ptr<const Event>& a,
    const std::shared_ptr<const Event>& b) {
  TORCH_INTERNAL_ASSERT(a);
  TORCH_INTERNAL_ASSERT(b);
  return *a < *b;
}
} // namespace

void Event::setParent(std::shared_ptr<Event>& parent) {
  TORCH_INTERNAL_ASSERT(parent.get() != nullptr);
  parent_ = parent;
  parent->children_.push_back(shared_from_this());
  TORCH_INTERNAL_ASSERT(parent->children_.back().get() != nullptr);
  std::stable_sort(
    parent->children_.begin(),
    parent->children_.end(),
    &ptr_sort);
}

Graph::Graph(std::deque<std::shared_ptr<Event>>&& events)
    : all_nodes_{std::move(events)} {
  std::stable_sort(all_nodes_.begin(), all_nodes_.end(), &ptr_sort);
  for (const auto node : all_nodes_) {
    if (node->parent_.lock().get() == nullptr) {
      head_nodes_.push_back(node);
    }
  }
}

void Graph::emitKinetoHostEvents(torch::profiler::impl::kineto::TraceWrapper& cpu_trace) {
  for (const auto& node : all_nodes_) {
    if (node->type() == EventType::TorchOp) {
      auto m = c10::get<torch::profiler::graph::TorchOp>(node->metadata_);
      cpu_trace.addCPUActivity(
        m.op_name_,
        torch::profiler::impl::kineto::kineto_ids(),  // TODO: this is a placeholder.
        node->correlation_id_,
        
        // Kineto uses us rather than ns.
        node->start_time_ / 1000,
        node->end_time_ / 1000);
    }
  }
}

void Graph::addKinetoDeviceEvents(torch::profiler::impl::kineto::ActivityTraceWrapper& trace) {
#ifdef USE_KINETO
  ska::flat_hash_map<correlation_id_t, std::shared_ptr<Event>> node_map;
  for (auto node : all_nodes_ ) {
    if (node->correlation_id_ != no_correlation_id) {
      auto it = node_map.insert({node->correlation_id_, node});
      TORCH_INTERNAL_ASSERT(it.second);
    }
  }

  const ska::flat_hash_map<libkineto::ActivityType, EventType> type_map {
    {libkineto::ActivityType::USER_ANNOTATION,   EventType::UserAnnotation},
    {libkineto::ActivityType::GPU_MEMCPY,        EventType::GPUMemcpy},
    {libkineto::ActivityType::GPU_MEMSET,        EventType::GPUMemset},
    {libkineto::ActivityType::CONCURRENT_KERNEL, EventType::ConcurrentKernal},
    {libkineto::ActivityType::CUDA_RUNTIME,      EventType::CudaRuntime}
  };

  const auto& events = *(trace.get()->activities());
  for (const auto& ev_ptr : events) {
    const auto& activity = *ev_ptr;
    const auto type_it = type_map.find(activity.type());
    if (type_it != type_map.end()) {
      correlation_id_t correlation_id = activity.linkedActivity()
        ? activity.linkedActivity()->correlationId() : no_correlation_id;
      // `nextCorrelationIDRange` starts at 1, so correlation_id is zero only
      // if linkedActivity was default constructed.
      correlation_id = correlation_id ? correlation_id : no_correlation_id;

      auto e = Event::create(
        c10::Device(
          torch::autograd::profiler::deviceTypeFromActivity(activity.type()),
          activity.deviceId()),
        /*tid=*/0,    // TODO: placeholder
        KinetoEvent(activity.name(), type_it->second)
      );
      e->start_time_ = activity.timestamp() * 1000;
      e->end_time_ = e->start_time_ + activity.duration() * 1000;   
      e->correlation_id_ = correlation_id;

      all_nodes_.push_back(e);
      auto it = node_map.find(correlation_id);
      if (it == node_map.end()) {
        head_nodes_.push_back(e);
      } else {
        e->setParent(it->second);
      }
    }
  }
#endif // USE_KINETO
}

namespace {
struct PtrUse {
  time_t t_;
  c10::Device device_;
  c10::variant<TensorSummaryPtr, Allocation> value_;
};

template <typename T>
void apply_if(auto& x, auto&& f) {
  if (c10::holds_alternative<T>(x)) {
    f(c10::get<T>(x));
  }
}
} // namespace

void Graph::finalize() {
  // Generally Tidy up.
  std::stable_sort( head_nodes_.begin(), head_nodes_.end(), &ptr_sort);
  std::stable_sort(all_nodes_.begin(), all_nodes_.end(), &ptr_sort);

  // Assign identities to Tensors
  std::deque<PtrUse> uses;
  auto push_tensors = [&](c10::optional<std::vector<ValueSummary>> values, time_t time) {
    auto push_if_tensor = [&](auto& value) {
      apply_if<TensorSummaryPtr>(value, [&](auto& t) {
        uses.push_back({time, t->device_, t});
      });
    };

    for (auto& value : values.value_or(std::vector<ValueSummary>())) {
      push_if_tensor(value);
      apply_if<TensorListSummary>(value, [&](auto& list) {
        for (auto& list_i : list) {
          push_if_tensor(list_i);
        }
      });
    }
  };

  for (const auto& node : all_nodes_) {
    apply_if<TorchOp>(node->metadata_, [&](const auto& op) {
      push_tensors(op.inputs_, node->start_time_);
      push_tensors(op.outputs_, node->end_time_);
    });
    apply_if<Allocation>(node->metadata_, [&](const auto& alloc) {
      uses.push_back({node->start_time_, node->device_, alloc});
    });
  }

  std::stable_sort(
    uses.begin(),
    uses.end(),
    [](const auto& a, const auto& b) { return a.t_ < b.t_; });

  using key_t = std::pair<void*, c10::Device>;
  int64_t id_counter {0};
  ska::flat_hash_map<key_t, int64_t, c10::hash_pair> ids;

  for (auto& use : uses) {
    apply_if<Allocation>(use.value_, [&](const auto& alloc) {
      ids.erase(std::make_pair(alloc.ptr_, use.device_));
    });
    apply_if<TensorSummaryPtr>(use.value_, [&](auto& t) {
      const auto key = std::make_pair(t->ptr_, t->device_);
      const auto it = ids.find(key);
      if (it == ids.end()) {
        ids[key] = id_counter++;
      }
      t->identity_ = ids.at(key);
    });
  }
}

EventType toEventType(const Metadata& m) {
  return c10::visit(
      c10::overloaded(
          [](const TorchOp&) { return EventType::TorchOp; },
          [](const PyCall&) { return EventType::PythonFunction; },
          [](const PyNNModuleForwardCall&) {
            return EventType::PythonNNModule;
          },
          [](const PyCCall&) { return EventType::PythonCFunction; },
          [](const Allocation&) { return EventType::Allocation; },
          [](const KinetoEvent& e) { return e.type_; }),
      m);
}

EventType Event::type() const {
  return toEventType(metadata_);
};

std::string toString(const EventType& e) {
  switch (e) {
    case EventType::TorchOp:          return "TorchOp";
    case EventType::PythonFunction:   return "PythonFunction";
    case EventType::PythonNNModule:   return "PythonNNModule";
    case EventType::PythonCFunction:  return "PythonCFunction";
    case EventType::Allocation:       return "Allocation";
    case EventType::UserAnnotation:   return "UserAnnotation";
    case EventType::GPUMemcpy:        return "GPUMemcpy";
    case EventType::GPUMemset:        return "GPUMemset";
    case EventType::ConcurrentKernal: return "ConcurrentKernal";
    case EventType::CudaRuntime:      return "CudaRuntime";
  }
  return "EventType: ???";
}

std::string toString(const TensorSummary& s) {
  return fmt::format(
    "Tensor( {}, {}, ID={} ({}), [{}] )",
    c10::toString(s.dtype_),
    s.device_.str(),
    s.identity_,
    s.ptr_,
    fmt::join(s.sizes_, ", "));
}

std::string toString(const UndefinedTensorSummary& s) {
  return "Undefined Tensor";
}

std::string toString(const TensorListSummary& s) {
  std::vector<std::string> elements;
  for (const auto& i : s) {
    elements.push_back(c10::visit([](const auto& x){ return toString(x); }, i));
  }
  return fmt::format("[{}]", fmt::join(elements, ", "));
}

std::string toString(const OtherSummary& s) {
  return "Other";
}

std::string toString(const ValueSummary& s) {
  return c10::visit(
      c10::overloaded(
          [](const std::shared_ptr<TensorSummary>& s) {
            return fmt::format("Tensor: {}", toString(s));
          },
          [](const UndefinedTensorSummary& s) { return toString(s); },
          [](const TensorListSummary& s) {
            return fmt::format("TensorList: {}", toString(s));
          },
          [](const c10::Scalar& s) {
            return fmt::format("Scalar: {}", c10::toString(s));
          },
          [](const OtherSummary& s) { return toString(s); }),
      s);
}

std::string toString(const TorchOp& s) {
  return fmt::format("TorchOp: {}", s.op_name_);
}

std::string toString(const PyFrameState& s) {
  return fmt::format(
    "{}:{} {}", 
    s.filename_.str(), s.funcname_.str(), s.line_no_);
}

std::string toString(const PyCall& c) {
  return fmt::format("PyCall: {}", toString(c.function_));
}

std::string toString(const PyNNModuleForwardCall& c) {
  return fmt::format("PyNNModuleForward: {}", c.cls_name_.str());
}

std::string toString(const PyCCall& c) {
  return fmt::format("PyCCall: {}", c.name_.str());
}

std::string toString(const Allocation& a) {
  return fmt::format("Allocation: {} ({} bytes)", fmt::ptr(a.ptr_), a.alloc_size_);
}

TORCH_API std::string toString(const KinetoEvent& e) {
  return fmt::format("{}: {}", toString(e.type_), e.name_);
}

std::string toString(const Metadata& m) {
  return fmt::format(
    "{}: {}",
    toString(toEventType(m)),
    c10::visit([](const auto& i){ return toString(i); }, m));
}

std::string toString(const Event& e) {
  return toString(e.metadata_);
}

std::string toString(const Graph& g) {
  return fmt::format("Profiler Graph ({})", g.all_nodes_.size());
}


} // namespace graph
} // namespace profiler
} // namespace torch
