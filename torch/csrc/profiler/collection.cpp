#include <torch/csrc/profiler/collection.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <forward_list>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {
namespace {
PyTracerStartFn py_start_fn {nullptr};
PyTracerStopFn py_stop_fn {nullptr};

void py_start(impl::RecordQueue* record_queue) {
  TORCH_INTERNAL_ASSERT(py_start_fn != nullptr);
  py_start_fn(record_queue);
}

auto py_stop() {
  TORCH_INTERNAL_ASSERT(py_stop_fn != nullptr);
  return py_stop_fn();
}

// See `RecordQueue::getSubqueue()` for an overview of this cache.
struct SubQueueThreadCache {
  uint32_t key_;
  ThreadLocalSubqueue* ref_;
};

std::atomic<uint32_t> queue_id_{0};
thread_local SubQueueThreadCache sub_queue_cache_{0, nullptr};
}  // namespace

correlation_id_t EventArray::nextCorrelationIDRange() {
  // NB: Zero is used as the null state for correlation_ids so that default
  //     initialized values are not interpreted as real IDs. To maintain this
  //     invariant, the block range counter must begin at one.
  static std::atomic<correlation_id_t> corr_id_block_{1};
  return corr_id_block_.fetch_add(1, std::memory_order_relaxed);
}

correlation_id_t EventQueue::lastCorrelationID() {
  // We subtract one because `next_` was incremented in `emplace_back`.
  return buffer_end_->block_start_ + (next_ - 1 - buffer_end_->data());
}

correlation_id_t EventQueue::correlationIDFor(const EventQueue::Iterator& i) {
  switch (i->tag_.get()) {
    case EventTag::OpCall:
    case EventTag::OpReturn:
      {
        auto a = i.address();
        return a.first->block_start_ + a.second;
      }
    default:
      return no_correlation_id;
  }
}

void InputOutputEncoder::push(const std::vector<c10::IValue>& values) {
  for (const auto& value : values) {
    if (value.isTensor()) {
      push(value.toTensor());
    } else if (value.isScalar()) {
      push(value.toScalar());
    } else if (value.isTensorList()) {
      tags_.emplace_back(Tag::TensorListBegin);
      for (const auto& t : value.toTensorList()) {
        push(t);
      }
      tags_.emplace_back(Tag::TERMINATOR);
    } else {
      tags_.emplace_back(Tag::Other);
    }
  }
  tags_.emplace_back(Tag::TERMINATOR);
}

void InputOutputEncoder::push(const at::Tensor& t) {
  if (t.defined()) {
    tags_.emplace_back(Tag::Tensor);
    const auto device = t.device();
    const auto& sizes = t.sizes();
    const auto dim = sizes.size();
    TORCH_CHECK(
      dim <= std::numeric_limits<uint32_t>::max(),
      "Cannot profile Tensors of size > uint32 max. Got dim: ", dim);

    tensor_metadata_.emplace_back(
      (void*)t.unsafeGetTensorImpl(),
      t.scalar_type(),
      device.type(),
      device.index(),
      dim
    );
    
    for (const auto i : sizes) {
      tensor_sizes_.emplace_back(i);
    }
  } else {
    tags_.emplace_back(Tag::UndefinedTensor);
  }
}

void InputOutputEncoder::push(const c10::Scalar s) {
  tags_.emplace_back(Tag::Scalar);
  scalars_.emplace_back(s);
}

std::deque<std::vector<torch::profiler::graph::ValueSummary>> InputOutputEncoder::materialize() {
  using namespace torch::profiler::graph;
  auto tag_it = tags_.begin();
  auto tensor_metadata_it = tensor_metadata_.begin();
  auto tensor_size_it = tensor_sizes_.begin();
  auto scalars_it = scalars_.begin();

  auto decode_tensor = [&tag_it, &tensor_metadata_it, &tensor_size_it]() {
    TORCH_INTERNAL_ASSERT(*tag_it == Tag::Tensor);
    auto metadata = *tensor_metadata_it++;
    std::vector<int64_t> sizes;
    for (const auto _ : c10::irange(metadata.dim_)) {
      sizes.push_back(*tensor_size_it++);
    }

    return std::make_shared<TensorSummary>(
      metadata.ptr_,
      c10::Device(metadata.device_type_, metadata.device_index_),
      metadata.dtype_,
      sizes
    );
  };

  std::deque<std::vector<ValueSummary>> out;
  out.emplace_back();
  while (tag_it != tags_.end()) {
    switch (*tag_it) {
      case Tag::Tensor:
        out.back().push_back(decode_tensor());
        break;

      case Tag::UndefinedTensor:
        out.back().push_back(UndefinedTensorSummary());
        break;

      case Tag::TensorListBegin:
        {
          TensorListSummary list;
          while (*(++tag_it) != Tag::TERMINATOR) {
            if (*tag_it == Tag::Tensor) {
              list.push_back(decode_tensor());
            } else {
              TORCH_INTERNAL_ASSERT(*tag_it == Tag::UndefinedTensor);
              list.push_back(UndefinedTensorSummary());
            }
          }
          out.back().push_back(std::move(list));
        }
        break;

      case Tag::Scalar:
        out.back().push_back(*scalars_it++);
        break;

      case Tag::Other:
        out.back().push_back(OtherSummary());
        break;

      case Tag::TERMINATOR:
        out.emplace_back();
        break;

      default:
        break;
    }
    tag_it++;
  }

  TORCH_INTERNAL_ASSERT(out.back().empty());
  out.pop_back();
  TORCH_INTERNAL_ASSERT(tensor_metadata_it == tensor_metadata_.end());
  TORCH_INTERNAL_ASSERT(tensor_size_it == tensor_sizes_.end());
  TORCH_INTERNAL_ASSERT(scalars_it == scalars_.end());
  return out;
}

// ============================================================================
// == SubQueue: Thread local (lock free) storage ==============================
// ============================================================================
ThreadLocalSubqueue::ThreadLocalSubqueue(uint64_t tid, RecordQueue* parent)
  : tid_(tid), parent_(parent) {}

void ThreadLocalSubqueue::pushEvent(EventTag tag, uint8_t payload) {
  events_.emplace_back(WrappedEventTag{tag}, payload, clock_());
}

std::unique_ptr<at::ObserverContext> ThreadLocalSubqueue::recordOpCall(
    const at::RecordFunction& fn) {
  const auto forward_tid = fn.forwardThreadId();
  const bool is_bwd = forward_tid > 0;
  const auto& handle = fn.operator_handle();
  const bool has_schema = handle && handle.hasSchema();
  const auto sequence_number = fn.seqNr();
  const bool has_seq_nr = sequence_number != -1;
  pushEvent(
    WrappedEventTag::packOpCall(
      /*is_bwd=*/is_bwd,
      /*has_schema=*/has_schema,
      /*has_seq_nr=*/has_seq_nr),
    /*payload_= */(uint8_t)fn.scope());

  has_schema ? op_handles_.emplace_back(handle)
             : op_names_.emplace_back(fn.name());

  if (parent_->config().report_input_shapes) {
    op_inputs_and_outputs_.push(fn.inputs());
  }

  if (is_bwd) {
    forward_thread_ids_.emplace_back(forward_tid);
  }
  
  if (has_seq_nr) {
    sequence_numbers_.emplace_back(sequence_number);
  }

  auto corr_id = events_.lastCorrelationID();
  torch::profiler::impl::kineto::pushCorrelationId(corr_id);
  torch::profiler::impl::kineto::recordThreadInfo();
  return std::make_unique<KinetoObserverContext>(corr_id);
}

void ThreadLocalSubqueue::recordOpReturn(
    const at::RecordFunction& fn, const at::ObserverContext* ctx) {
  const auto kineto_ctx = static_cast<const KinetoObserverContext*>(ctx);
  pushEvent(EventTag::OpReturn);
  return_corr_ids_.emplace_back(kineto_ctx->correlation_id_);

  if (parent_->config().report_input_shapes) {
    op_inputs_and_outputs_.push(fn.outputs());
  }
  torch::profiler::impl::kineto::popCorrelationId();
  torch::profiler::impl::kineto::recordThreadInfo();
}

void ThreadLocalSubqueue::recordPyCall(
    const bool is_module,
    const uint8_t tid,
    const interned_t caller,
    const interned_t metadata) {
  pushEvent(
    is_module ? EventTag::PyModuleCall
              : EventTag::PyCall,
    /*payload=*/tid);
  py_call_metadata_.emplace_back(caller, metadata);
}

void ThreadLocalSubqueue::recordPyCCall(
    const uint8_t tid,
    const interned_t caller,
    const interned_t metadata) {
  pushEvent(EventTag::PyCCall, /*payload=*/tid);
  py_call_metadata_.emplace_back(caller, metadata);
}

void ThreadLocalSubqueue::recordPyReturn(
    const uint8_t tid,
    const bool is_c_call) {
  pushEvent(
    is_c_call ? EventTag::PyCReturn
              : EventTag::PyReturn,
    /*payload=*/tid);
}

void ThreadLocalSubqueue::recordMemoryUsage(
    void* ptr,
    int64_t alloc_size,
    int64_t total_allocated,
    int64_t total_reserved,
    c10::Device device) {
  pushEvent(
    EventTag::Allocate,
    /*payload=*/static_cast<uint8_t>(device.type()));
  allocations_.emplace_back(
    ptr, alloc_size, device.index(), total_allocated, total_reserved);
  torch::profiler::impl::kineto::recordThreadInfo();
}

RecordQueue::RecordQueue(const ProfilerConfig& config)
  : id_(++queue_id_), config_(config), stopped_{false} {
  if (config.with_stack) {
    py_start(this);
  }
}

RecordQueue::~RecordQueue() {
  if (config_.with_stack && !stopped_) {
    stopped_ = true;
    py_stop();
  }
}
	
ThreadLocalSubqueue* RecordQueue::getSubqueue() {
  // In the most common case, a thread will want to write to the same sub-queue
  // that it wrote to last call. The only time that isn't true is if:
  //  A) The profiler context has ended and we are in a new one.
  //  B) Two profilers are active in different TLS contexts, and this thread
  //     is a worker helping with intra-op parallelism.
  // Since we expect this to be the OVERWHELMINGLY common case (>99%), we add a
  // special thread_local cache so that we can skip the overall `flat_hash_map`
  // (and corresponding lock).
  if (id_ == sub_queue_cache_.key_) {
    return sub_queue_cache_.ref_;
  }

  const auto tid = at::RecordFunction::currentThreadId();
  std::lock_guard<std::mutex> guard(sub_queue_mutex_);
  auto it = sub_queues_.find(tid);
  if (it == sub_queues_.end()) {
    it = sub_queues_.emplace(
      tid, std::make_unique<ThreadLocalSubqueue>(tid, this)).first;
  }

  sub_queue_cache_ = SubQueueThreadCache{id_, it->second.get()};
  return it->second.get();
}

std::deque<raw_event_ptr_t> ThreadLocalSubqueue::postProcess(
    const torch::profiler::impl::PyCodeDescriptions py_code_descriptions,
    const std::function<time_t(approx_time_t)> time_converter) {
  auto event_it = events_.begin();
  auto clock_decoder = clock_.makeDecoder();
  auto op_def_it = op_handles_.begin();
  auto op_name_it = op_names_.begin();
  auto return_corr_id_it = return_corr_ids_.begin();
  auto fwd_thread_it = forward_thread_ids_.begin();
  auto sequence_number_it = sequence_numbers_.begin();

  auto py_metadata_it = py_call_metadata_.begin();
  auto& py_frame_states = py_code_descriptions.frame_states_;
  auto& py_c_function_names = py_code_descriptions.c_function_names_;
  auto& py_modules = py_code_descriptions.modules_;

  auto inputs_and_outputs = op_inputs_and_outputs_.materialize();
  const auto report_input_shapes = parent_->config().report_input_shapes;
  TORCH_INTERNAL_ASSERT(report_input_shapes || inputs_and_outputs.empty());

  auto alloc_it = allocations_.begin();

  std::deque<raw_event_ptr_t> out;
  while (event_it != events_.end()) {
    const auto tag = event_it->tag_;
    const auto time = time_converter(clock_decoder(event_it->dt_));
    const auto correlation_id = tag.get() != EventTag::OpReturn 
      ? EventQueue::correlationIDFor(event_it)
      : *return_corr_id_it++;

    auto e = std::make_shared<UncompressedEvent>(tag, correlation_id, time, tid_);

    switch (tag.get()) {
      case EventTag::OpCall:
        {
          const auto scope = (at::RecordScope)(event_it->payload_);
          e->metadata_ = tag.has_schema()
            ? torch::profiler::graph::TorchOp {(*op_def_it++).schema(), scope}
            : torch::profiler::graph::TorchOp {*op_name_it++, scope};
          auto& op = c10::get<torch::profiler::graph::TorchOp>(*e->metadata_);
           if (tag.is_bwd()) {
            e->forward_tid_ = *fwd_thread_it++;
          }
          if (tag.has_sequence_number()) {
            op.sequence_number_ = *sequence_number_it++;
          }
          if (report_input_shapes) {
            op.inputs_ = std::move(inputs_and_outputs.front());
            inputs_and_outputs.pop_front();
          }
        }
        break;

      case EventTag::OpReturn:
        if (report_input_shapes) {
          e->outputs_ = std::move(inputs_and_outputs.front());
          inputs_and_outputs.pop_front();
        }
        break;

      case EventTag::PyCall:
        e->metadata_ = torch::profiler::graph::PyCall {
          /*function=   */ py_frame_states.at(py_metadata_it->second),
          /*caller=     */ py_frame_states.at(py_metadata_it->first),
          /*python_tid= */ event_it->payload_
        };
        e->python_tid_ = event_it->payload_;
        py_metadata_it++;
        break;

      case EventTag::PyModuleCall:
        e->metadata_ = torch::profiler::graph::PyNNModuleForwardCall {
          /*id=         */ py_modules.at(py_metadata_it->second).id_,
          /*cls_name=   */ py_modules.at(py_metadata_it->second).cls_name_,
          /*caller=     */ py_frame_states.at(py_metadata_it->first),
          /*python_tid= */ event_it->payload_
        };
        e->python_tid_ = event_it->payload_;
        py_metadata_it++;
        break;

      case EventTag::PyCCall:
        e->metadata_ = torch::profiler::graph::PyCCall {
          /*name=       */ py_c_function_names.at(py_metadata_it->second),
          /*caller=     */ py_frame_states.at(py_metadata_it->first),
          /*python_tid= */ event_it->payload_
        };
        e->python_tid_ = event_it->payload_;
        py_metadata_it++;
        break;

      case EventTag::PyReturn:
      case EventTag::PyCReturn:
        e->python_tid_ = event_it->payload_;
        break;

      case EventTag::Allocate:
        e->metadata_ = torch::profiler::graph::Allocation{
          /*ptr_=             */alloc_it->ptr_,
          /*alloc_size_=      */alloc_it->alloc_size_,
          /*total_allocated_= */alloc_it->total_allocated_,
          /*total_reserved_=  */alloc_it->total_reserved_
        };
        e->alloc_device_ = c10::Device(
          static_cast<c10::DeviceType>(event_it->payload_),
          alloc_it->device_index_);
        alloc_it++;
        break;

      default:
        TORCH_CHECK(false, "Unknown tag: ", (int)tag.get());
        break;
    }

    out.push_back(std::move(e));
    event_it++;
  }

  TORCH_INTERNAL_ASSERT(op_def_it == op_handles_.end());
  TORCH_INTERNAL_ASSERT(op_name_it == op_names_.end());
  TORCH_INTERNAL_ASSERT(return_corr_id_it == return_corr_ids_.end());
  TORCH_INTERNAL_ASSERT(fwd_thread_it == forward_thread_ids_.end())
  TORCH_INTERNAL_ASSERT(py_metadata_it == py_call_metadata_.end());
  TORCH_INTERNAL_ASSERT(inputs_and_outputs.empty());
  TORCH_INTERNAL_ASSERT(alloc_it == allocations_.end())
  TORCH_INTERNAL_ASSERT(sequence_number_it == sequence_numbers_.end());
  return out;
}

// Preprocessing for event replay.
void matchPythonCalls(std::deque<raw_event_ptr_t>& sorted_raw_events) {
  ska::flat_hash_map<uint8_t, std::deque<raw_event_ptr_t>> py_thread_stacks;
  for (auto& i : sorted_raw_events) {
    const auto tag = i->tag_.get();
    if (i->python_tid_.has_value()) {
      // operator[] default construction is intentional. (empty stack)
      auto& stack = py_thread_stacks[*i->python_tid_];

      if (tag == EventTag::PyReturn || tag == EventTag::PyCReturn) {
        TORCH_INTERNAL_ASSERT(!stack.empty());
        TORCH_INTERNAL_ASSERT(!stack.back()->begin_raw_event_.has_value());
        i->begin_raw_event_ = stack.back();
        stack.pop_back();

      } else {
        TORCH_INTERNAL_ASSERT(
          tag == EventTag::PyCall ||
          tag == EventTag::PyModuleCall ||
          tag == EventTag::PyCCall);
        if (!stack.empty() && stack.back()->tid_ != i->tid_) {
          i->calling_tid_ = *(stack.back()->python_tid_);
        }
        stack.push_back(i);
      }
    }
  }
}

void matchTorchCalls(std::deque<raw_event_ptr_t>& sorted_raw_events) {
  ska::flat_hash_map<correlation_id_t, raw_event_ptr_t> op_calls;
  for (auto& i : sorted_raw_events) {
    const auto tag = i->tag_.get();
    if (tag == EventTag::OpCall || tag == EventTag::OpReturn) {
      auto it = op_calls.find(i->correlation_id_);
      TORCH_INTERNAL_ASSERT(i->correlation_id_ != no_correlation_id);

      if (tag == EventTag::OpCall) {
        TORCH_INTERNAL_ASSERT(it == op_calls.end());
        op_calls[i->correlation_id_] = i;

      } else if (it != op_calls.end()) {
        i->begin_raw_event_ = it->second;
        // op_calls.erase(i->correlation_id_);

      } else {
        TORCH_WARN("Unhandled PyTorch op return: ", i->correlation_id_);
      }
    }
  }
}

// Autograd runs on its own threadpool. However it runs:
//   autograd::engine::evaluate_function: FooBackward0
//   FooBackward0
//   ...
// So for backward ops we have to walk the stack until we get to another
// op call and set the calling thread based on the forward thread.
void matchAutograd(std::deque<raw_event_ptr_t>& sorted_raw_events) {
  ska::flat_hash_set<correlation_id_t> finished;
  ska::flat_hash_map<uint64_t, std::deque<raw_event_ptr_t>> event_stacks;
    for (auto& i : sorted_raw_events) {
      const auto tag = i->tag_.get();
      if (tag == EventTag::OpCall) {
        auto& stack = event_stacks[i->tid_];

        // Lazily clear finished ops.
        while (!stack.empty() && finished.find(i->correlation_id_) != finished.end()) {
          stack.pop_back();
        }

        if (!stack.empty() && i->tag_.is_bwd() && !stack.back()->tag_.is_bwd()) {
          TORCH_INTERNAL_ASSERT(stack.back()->metadata_.has_value());
          TORCH_INTERNAL_ASSERT(c10::get<torch::profiler::graph::TorchOp>(
            *stack.back()->metadata_).op_name_
            .find("autograd::engine::evaluate_function:") != std::string::npos);
          stack.back()->calling_tid_ = *(i->forward_tid_);
        }

        stack.push_back(i);

    } else if (tag == EventTag::OpReturn) {
      finished.insert(i->correlation_id_);
    }
  }
}

namespace {
auto ptr_sort(
    const std::shared_ptr<const UncompressedEvent>& a,
    const std::shared_ptr<const UncompressedEvent>& b) {
  TORCH_INTERNAL_ASSERT(a);
  TORCH_INTERNAL_ASSERT(b);
  return *a < *b;
}
} // namespace

std::deque<std::shared_ptr<torch::profiler::graph::Event>> makeTree(
    std::deque<raw_event_ptr_t>& sorted_raw_events) {
  using event_ptr_t = std::shared_ptr<torch::profiler::graph::Event>;
  std::deque<event_ptr_t> events;

  ska::flat_hash_map<uint64_t, std::deque<raw_event_ptr_t>> event_stacks;

  // Stacks are lazily cleared to more robustly handle malformed call hierarchies.
  auto lazy_clear = [&events](std::deque<raw_event_ptr_t>& stack, bool cleanup = false) {
    while (!stack.empty() && (stack.back()->finished_ || cleanup)) {
      TORCH_INTERNAL_ASSERT(stack.back()->final_event_.has_value())
      events.push_back(*(stack.back()->final_event_));
      stack.pop_back();
    }
  };

  auto push_call = [&event_stacks, &lazy_clear](raw_event_ptr_t& raw_event){
    auto& stack = event_stacks[raw_event->tid_];
    lazy_clear(stack);

    TORCH_INTERNAL_ASSERT(raw_event->metadata_.has_value());
    auto e = torch::profiler::graph::Event::create(
      c10::Device(c10::DeviceType::CPU), raw_event->tid_, *raw_event->metadata_);
    e->start_time_ = raw_event->t_;
    e->end_time_ = torch::profiler::graph::Event::UNSET_TIME;
    e->correlation_id_ = raw_event->correlation_id_;
    
    auto& calling_stack = event_stacks[raw_event->calling_thread()];
    lazy_clear(calling_stack);
    if (!calling_stack.empty()) {
      TORCH_INTERNAL_ASSERT(calling_stack.back()->final_event_.has_value());
      e->setParent(*(calling_stack.back()->final_event_));
    }
    raw_event->final_event_ = e;
    stack.push_back(raw_event);
  };

  auto pop_call = [](raw_event_ptr_t& raw_event) {
    if (raw_event->begin_raw_event_.has_value()) {
      auto begin_event = raw_event->begin_raw_event_->lock();
      TORCH_INTERNAL_ASSERT(begin_event->final_event_.has_value());
      auto& e = *(begin_event->final_event_);
      e->end_time_ = raw_event->t_;
      if (raw_event->outputs_.has_value()) {
        TORCH_INTERNAL_ASSERT(raw_event->tag_.get() == EventTag::OpReturn);
        c10::get<torch::profiler::graph::TorchOp>(e->metadata_).outputs_ = *raw_event->outputs_;
      }
      begin_event->finished_ = true;
    }
  };

  auto push_allocation = [&events, &event_stacks](raw_event_ptr_t& raw_event) {
    TORCH_INTERNAL_ASSERT(raw_event->alloc_device_.has_value());
    TORCH_INTERNAL_ASSERT(raw_event->metadata_.has_value());
    auto e = torch::profiler::graph::Event::create(
      *(raw_event->alloc_device_), raw_event->tid_, *(raw_event->metadata_));
    e->start_time_ = raw_event->t_;
    e->end_time_ = e->start_time_;
    TORCH_INTERNAL_ASSERT(raw_event->correlation_id_ == no_correlation_id);
    e->correlation_id_ = no_correlation_id;

    auto& stack = event_stacks[raw_event->tid_];
    if (!stack.empty()) {
      TORCH_INTERNAL_ASSERT(stack.back()->final_event_.has_value());
      e->setParent(*(stack.back()->final_event_));
    }
    events.push_back(e);
  };

  for (auto& i : sorted_raw_events) {
    switch (i->tag_.get()) {
      case EventTag::OpCall:
      case EventTag::PyCall:
      case EventTag::PyModuleCall:
      case EventTag::PyCCall:
        push_call(i);
        break;

      case EventTag::OpReturn:
      case EventTag::PyReturn:
      case EventTag::PyCReturn:
        pop_call(i);
        break;

      case EventTag::Allocate:
        push_allocation(i);
        break;

      default:
        TORCH_CHECK(false, "Unhandled tag: ", (int)i->tag_.get());
    }
  }

  // Handle calls which are still ongoing when profiling ends.
  for (auto& it : event_stacks) {
    lazy_clear(it.second, /*cleanup=*/true);
  }

  std::stable_sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
    TORCH_INTERNAL_ASSERT(a);
    TORCH_INTERNAL_ASSERT(b);
    return *a < *b;
  });
  return events;
}

torch::profiler::graph::Graph RecordQueue::finalize() {
  TORCH_INTERNAL_ASSERT(!stopped_, "postProcess is a destructive operation.")
  stopped_ = true;
  const auto py_code_descriptions_ = config_.with_stack
    ? py_stop()
    : PyCodeDescriptions();

  const auto time_converter = clock_converter_.makeConverter();
  std::deque<std::shared_ptr<UncompressedEvent>> raw_events;
  for (auto& subqueue_it : sub_queues_) {
    auto thread_events = subqueue_it.second->postProcess(
      py_code_descriptions_, time_converter);
    raw_events.insert(raw_events.end(), thread_events.begin(), thread_events.end());
  }

  std::stable_sort(raw_events.begin(), raw_events.end(), &ptr_sort);
  matchPythonCalls(raw_events);
  matchTorchCalls(raw_events);
  matchAutograd(raw_events);
  return torch::profiler::graph::Graph(makeTree(raw_events));
};

void registerPyFunctions(PyTracerStartFn start, PyTracerStopFn stop) {
  py_start_fn = start;
  py_stop_fn = stop;
}

} // namespace impl
} // namespace profiler
} // namespace torch
