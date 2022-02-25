#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/csrc/autograd/profiler_kineto.h>

#include <c10/macros/Export.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/nvtx_observer.h>

#include <ATen/Context.h>

#include <deque>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef USE_KINETO
#include <libkineto.h>

#ifndef _MSC_VER
// TODO: TO be removed, once this properly works from libkineto
// Literal copy-n-paste from third_party/kineto/libkineto/src/WeakSymbols.cpp
extern "C" {
// This function is needed to avoid superfluous dependency on GNU OpenMP library
// when cuPTI is linked statically For more details see
// https://github.com/pytorch/pytorch/issues/51026
__attribute__((weak)) int acc_get_device_type() {
  throw std::runtime_error(
      "Dummy implementation of acc_get_device_type is not supposed to be called!");
}
} // extern "C"
#endif // _MSC_VER
#endif // USE_KINETO

namespace torch {
namespace autograd {
namespace profiler {
namespace {
using torch::profiler::impl::ProfilerThreadLocalStateBase;
using torch::profiler::impl::ActiveProfilerType;

struct KinetoThreadLocalState : public ProfilerThreadLocalStateBase {
  explicit KinetoThreadLocalState(
      const ProfilerConfig& config,
      std::set<torch::profiler::impl::ActivityType> activities)
      : ProfilerThreadLocalStateBase(config),
        start_time_(torch::profiler::impl::getTimeSinceEpoch() / 1000),
        activities_(std::move(activities)),
        record_queue_(config),
        cpu_trace_(start_time_, "PyTorch Profiler") {}
  ~KinetoThreadLocalState() override = default;

  static KinetoThreadLocalState* getTLS() {
    auto tls = ProfilerThreadLocalStateBase::getTLS();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::KINETO);
    return static_cast<KinetoThreadLocalState*>(tls);
  }

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::KINETO;
  }

  bool tracePython() {
    return config().with_stack && activities_.count(ActivityType::CPU);
  }

  void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      int64_t total_allocated,
      int64_t total_reserved,
      c10::Device device) override {
    if (config_.profile_memory && config_.state != ProfilerState::Disabled) {
      getTLS()->record_queue_.getSubqueue()->recordMemoryUsage(
        ptr, alloc_size, total_allocated, total_reserved, device);
    }
  }

  const std::function<void(std::vector<KinetoEvent>&)>&
  getEventPostProcessingCallback() const {
    return event_post_process_cb_;
  }

  void setEventPostProcessingCallback(
      std::function<void(std::vector<KinetoEvent>&)>&& cb) {
    event_post_process_cb_ = std::move(cb);
  }

  torch::profiler::impl::kineto::ActivityTraceWrapper finalizeTrace() {
    auto end_time = torch::profiler::impl::getTimeSinceEpoch() / 1000;
    // Call events post processing callback before finalizing trace, if there is
    // one.
    if (getEventPostProcessingCallback()) {
      getEventPostProcessingCallback()(kineto_events_);
    }

    graph_ = record_queue_.finalize();
    graph_.emitKinetoHostEvents(cpu_trace_);
    // finalizeCPUTrace(cpu_trace_.get());
    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      cpu_trace_.transferCpuTrace(end_time);
    }

    auto trace = torch::profiler::impl::kineto::stopTrace();
    TORCH_CHECK(trace || !torch::profiler::kKinetoAvailable);
    addTraceEvents(trace);
    graph_.addKinetoDeviceEvents(trace);
    graph_.finalize();
    return trace;
  }

  void finalizeCPUTrace(std::unique_ptr<torch::profiler::impl::kineto::trace_t>& cpu_trace) {
#ifndef USE_KINETO
  }
#else // USE_KINETO
    TORCH_INTERNAL_ASSERT(
        cpu_trace->activities.size() == kineto_events_.size());
    // startThreadId_seqNum to pointer of activity.
    // Low-16bits of startThreadId and low-48bits seqNum are concatenated into
    // one uint64_t variable as key.
    std::unordered_map<std::pair<uint64_t, uint64_t>, libkineto::GenericTraceActivity*, c10::hash_pair>
        tidSeq2activity;
    uint64_t fwd_bwd_link_id = 1;

    for (const auto idx : c10::irange(cpu_trace->activities.size())) {
      auto& kineto_event = kineto_events_[idx];
      auto& activity = cpu_trace->activities[idx];

      if (kineto_event.hasShapes()) {
        activity.addMetadata("Input Dims", torch::profiler::impl::shapesToStr(kineto_event.shapes()));
      }
      if (kineto_event.hasStack()) {
        // NB: This is only for the JIT stack. The python stack (if applicable)
        //     is constructed later.
        activity.addMetadata(
            "Call stack", torch::profiler::impl::stacksToStr(kineto_event.stack(), ";"));
      }
      if (kineto_event.hasModuleHierarchy()) {
        activity.addMetadata(
            "Module Hierarchy",
            torch::profiler::impl::stacksToStr(kineto_event.moduleHierarchy(), "."));
      }
      if (kineto_event.hasTypes()) {
        activity.addMetadata("Input type", torch::profiler::impl::dtypesToStr(kineto_event.dtypes()));
      }
      if (!kineto_event.backend().empty()) {
        activity.addMetadata("Backend", "\"" + kineto_event.backend() + "\"");
      }

      // add information about an associated forward op, if a sequence number
      // is available (e.g. during training)
      if (kineto_event.sequenceNr() >= 0) {
        activity.addMetadata(
            "Fwd thread id", std::to_string(kineto_event.fwdThreadId()));
        activity.addMetadata(
            "Sequence number", std::to_string(kineto_event.sequenceNr()));
        generateForwardBackwardLink(
            kineto_event, fwd_bwd_link_id, activity, tidSeq2activity);
      }
    }
  }

  void generateForwardBackwardLink(
      const KinetoEvent& kineto_event,
      uint64_t& fwd_bwd_link_id,
      libkineto::GenericTraceActivity& activity,
      std::unordered_map<std::pair<uint64_t, uint64_t>, libkineto::GenericTraceActivity*, c10::hash_pair>&
          tidSeq2activity) {
    if (kineto_event.fwdThreadId() > 0) {
      // act is backward op.
      std::pair<uint64_t, uint64_t> key = {kineto_event.fwdThreadId(), kineto_event.sequenceNr()};
      auto iter = tidSeq2activity.find(key);
      if (iter != tidSeq2activity.end()) {
        libkineto::GenericTraceActivity* fwd = iter->second;
#ifdef USE_KINETO_UPDATED
        fwd->flow.start = true;
#else
        activity.flow.linkedActivity = fwd; // Only destination side set this,
                                            // to distinguish with start side.
#endif
        activity.flow.id = fwd->flow.id = fwd_bwd_link_id;
        activity.flow.type = fwd->flow.type = libkineto::kLinkFwdBwd;
        ++fwd_bwd_link_id;
      }
    } else if (kineto_event.startThreadId() != 0) {
      // act is forward op.
      std::pair<uint64_t, uint64_t> key = {kineto_event.startThreadId(), kineto_event.sequenceNr()};
      // Assumption: Among all ops with same sequence number,
      // the one with biggest start time is most likely launching backward op.
      auto iter = tidSeq2activity.find(key);
      if (iter == tidSeq2activity.end()) {
        tidSeq2activity[key] = &activity;
      } else {
        // Now the sequence number is only incremented on creating a "Node"
        // object for backward pass, by calling
        // "at::sequence_number::get_and_increment()". Among all ops with same
        // sequence number, the one with biggest startTime is the one launching
        // backward op.
        if (activity.startTime >= iter->second->startTime) {
          tidSeq2activity[key] = &activity;
        }
      }
    }
  }
#endif // USE_KINETO

  void addTraceEvents(torch::profiler::impl::kineto::ActivityTraceWrapper& trace) {
#ifdef USE_KINETO
    const auto& events = *(trace.get()->activities());
    for (const auto& ev_ptr : events) {
      const auto& activity = *ev_ptr;
      // These events are already processed
      if (activity.type() != libkineto::ActivityType::CPU_OP &&
          activity.type() != libkineto::ActivityType::CPU_INSTANT_EVENT &&
          activity.type() != libkineto::ActivityType::USER_ANNOTATION &&
          activity.type() != libkineto::ActivityType::PYTHON_FUNCTION) {
        kineto_events_.emplace_back();
        auto& kineto_event = kineto_events_.back();
        kineto_event.name(activity.name())
            .deviceIndex(activity.deviceId())
            .deviceResourceId(activity.resourceId())
            .startUs(activity.timestamp())
            .durationUs(activity.duration())
            .activityType((uint8_t)activity.type());
        kineto_event.deviceType(deviceTypeFromActivity(activity.type()));
      }
    }
#endif // USE_KINETO
  }

  uint64_t start_time_;
  std::set<torch::profiler::impl::ActivityType> activities_;
  torch::profiler::impl::RecordQueue record_queue_;
  torch::profiler::impl::kineto::TraceWrapper cpu_trace_;
  std::vector<KinetoEvent> kineto_events_;
  torch::profiler::graph::Graph graph_;
  // Optional, if event post-processing is enabled.
  std::function<void(std::vector<KinetoEvent>&)> event_post_process_cb_;
};

void pushProfilingCallbacks(const std::unordered_set<at::RecordScope>& scopes) {
  auto registration_state_ptr = KinetoThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(registration_state_ptr, "Expected profiler state set");
  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn)
              -> std::unique_ptr<at::ObserverContext> {
            auto state_ptr = KinetoThreadLocalState::getTLS();
            if (!state_ptr) {
              return nullptr;
            }
            return state_ptr->record_queue_.getSubqueue()->recordOpCall(fn);
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
            auto state_ptr = KinetoThreadLocalState::getTLS();
            if (!state_ptr) {
              return;
            }
            state_ptr->record_queue_.getSubqueue()->recordOpReturn(fn, ctx_ptr);
          })
          .needsInputs(registration_state_ptr->config().report_input_shapes)
          .needsOutputs(registration_state_ptr->config().report_input_shapes)
          .scopes(scopes));
  registration_state_ptr->setCallbackHandle(handle);
}

} // namespace

void reportBackendEventToActiveKinetoProfiler(
    const int64_t start_time_us,
    const int64_t end_time_us,
    const int64_t debug_handle,
    const at::RecordScope scope,
    const std::string& event_name,
    const std::string& backend_name) {
  auto state_ptr = KinetoThreadLocalState::getTLS();
  if (!state_ptr) {
    return;
  }

  #if 0
  auto ctx_ptr = state_ptr->newOpEvent();
  auto data_ptr = ctx_ptr->data_;
  data_ptr->start_us_ = start_time_us;
  data_ptr->end_us_ = end_time_us;
  data_ptr->correlation_id_ = std::numeric_limits<uint64_t>::max();
  data_ptr->start_thread_id_ = at::RecordFunction::currentThreadId();
  data_ptr->end_thread_id_ = data_ptr->start_thread_id_;
  data_ptr->sequence_number_ = -1;
  data_ptr->forward_thread_id_ = data_ptr->start_thread_id_;
  data_ptr->record_function_scope_ = (uint8_t)scope;
  data_ptr->is_async_ = false;
  data_ptr->debug_handle_ = debug_handle;
  data_ptr->kineto_info_ = torch::profiler::impl::kineto::kineto_ids();
  data_ptr->name_ = event_name;
  data_ptr->backend_ = backend_name;

  /* no support for input shapes now?
  if (config.report_input_shapes) {
    ctx_ptr->shapes = inputSizes(fn);
    ctx_ptr->dtypes = inputTypes(fn);
  }
  */

  torch::profiler::impl::kineto::recordThreadInfo();
  #endif  // 0
}

void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities) {
  if (config.state == ProfilerState::NVTX) {
    return;
  }
  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
          config.state == ProfilerState::KINETO_GPU_FALLBACK,
      "Supported only in Kineto profiler");
  torch::profiler::impl::kineto::prepareTrace(
      /*cpuOnly=*/!at::hasCUDA(), activities);
}

void enableProfilerWithEventPostProcess(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    std::function<void(std::vector<KinetoEvent>&)>&& cb,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(
      config.state != ProfilerState::NVTX,
      "NVTX does not support post processing callback.");
  enableProfiler(config, activities, scopes);
  auto state_ptr = KinetoThreadLocalState::getTLS();
  state_ptr->setEventPostProcessingCallback(std::move(cb));
}

void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(!profilerEnabled(), "Profiler is already enabled on this thread");
  if (config.state == ProfilerState::NVTX) {
    torch::profiler::impl::pushNVTXCallbacks(config, scopes);
    return;
  }

  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK);
  TORCH_CHECK(
      !activities.empty(), "No activities specified for Kineto profiler");

  auto state = std::make_shared<KinetoThreadLocalState>(config, activities);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

  // if (state->tracePython()) {
  //   python_tracer::call(python_tracer::Command::kStartOne);
  // }

  if (activities.count(ActivityType::CPU)) {
    pushProfilingCallbacks(scopes);
  }

  torch::profiler::impl::kineto::startTrace();
}

std::unique_ptr<ProfilerResult> disableProfiler() {
  // all the DebugInfoBase objects are scope based and supposed to use
  // DebugInfoGuard
  auto state =
      c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);

  auto state_ptr = static_cast<ProfilerThreadLocalStateBase*>(state.get());
  const auto& config = state_ptr->config();
  TORCH_CHECK(
      state_ptr &&
          (config.state == ProfilerState::KINETO ||
           config.state == ProfilerState::KINETO_GPU_FALLBACK ||
           config.state == ProfilerState::NVTX),
      "Can't disable Kineto profiler when it's not running");

  if (state_ptr->hasCallbackHandle()) {
    at::removeCallback(state_ptr->callbackHandle());
  }

  if (state_ptr->config().state == ProfilerState::NVTX) {
    return std::make_unique<ProfilerResult>();
  }

  auto kineto_state_ptr = static_cast<KinetoThreadLocalState*>(state_ptr);
  auto trace = kineto_state_ptr->finalizeTrace();
  auto out = std::make_unique<ProfilerResult>(
      kineto_state_ptr->start_time_,
      std::move(kineto_state_ptr->kineto_events_),
      std::move(kineto_state_ptr->graph_),
      std::move(trace));
  return out;
}

int64_t KinetoEvent::cudaElapsedUs() const {
  if (!cuda_event_start_ || !cuda_event_end_) {
    return -1;
  }
  try {
    return (int64_t)torch::profiler::impl::cudaStubs()->elapsed(&cuda_event_start_, &cuda_event_end_);
  } catch (std::exception& e) {
    LOG(WARNING) << "Failed to measure time between two CUDA events. "
                 << e.what();
  }
  return -1;
}

ProfilerResult::ProfilerResult(
    uint64_t start_time,
    std::vector<KinetoEvent> events,
    torch::profiler::graph::Graph graph,
    torch::profiler::impl::kineto::ActivityTraceWrapper trace)
    : trace_start_us_(start_time),
      events_(std::move(events)),
      graph_(std::move(graph)),
      trace_(std::move(trace)) {}
ProfilerResult::ProfilerResult() = default;
ProfilerResult::~ProfilerResult() = default;

void ProfilerResult::save(const std::string& path) {
  trace_.save(path);
}

} // namespace profiler
} // namespace autograd
} // namespace torch
