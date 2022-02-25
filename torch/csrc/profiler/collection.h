#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <ATen/record_function.h>
#include <c10/util/Optional.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/python_stub.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/collection-inl.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/util.h>
	
namespace torch {
namespace profiler {
namespace impl {

struct KinetoObserverContext : public at::ObserverContext {
  KinetoObserverContext(uint64_t correlation_id)
    : correlation_id_(correlation_id) {}
  correlation_id_t correlation_id_;
};

class RecordQueue;

class TORCH_API ThreadLocalSubqueue : public SubqueueOpaqueFields {
 public:
  ThreadLocalSubqueue(uint64_t tid, RecordQueue* parent);

  std::unique_ptr<at::ObserverContext> recordOpCall(
    const at::RecordFunction& fn);

  void recordOpReturn(
    const at::RecordFunction& fn,
    const at::ObserverContext* ctx);
  
  void recordPyCall(
    const bool is_module,
    const uint8_t tid,
    const interned_t caller,
    const interned_t metadata);

  void recordPyCCall(
    const uint8_t tid,
    const interned_t caller,
    const interned_t metadata);

  void recordPyReturn(
    const uint8_t tid,
    const bool is_c_call);

  void recordMemoryUsage(
    void* ptr,
    int64_t alloc_size,
    int64_t total_allocated,
    int64_t total_reserved,
    c10::Device device);

 private:
  friend class RecordQueue;
  void pushEvent(EventTag tag, uint8_t payload = 0);
  std::deque<raw_event_ptr_t> postProcess(
    const PyCodeDescriptions py_code_descriptions,
    const std::function<time_t(approx_time_t)> time_converter);

  uint64_t tid_;
  RecordQueue* parent_;
};

class TORCH_API RecordQueue {
 public:
  explicit RecordQueue(const ProfilerConfig& config);
  ~RecordQueue();

  const ProfilerConfig& config() { return config_; }
  torch::profiler::graph::Graph finalize();

  ThreadLocalSubqueue* getSubqueue();

 private:
  uint32_t id_;
  ProfilerConfig config_;
  ska::flat_hash_map<uint64_t, std::unique_ptr<ThreadLocalSubqueue>> sub_queues_;
  ApproximateClockToUnixTimeConverter clock_converter_;
  bool stopped_;

  std::mutex sub_queue_mutex_;
};
  	
} // namespace impl
} // namespace profiler
} // namespace torch
