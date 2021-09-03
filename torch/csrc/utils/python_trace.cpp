#include <torch/csrc/utils/python_trace.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <Python.h>
#include <frameobject.h>

#include <ATen/record_function.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace profiler {
namespace impl {

namespace {
int64_t now() {
  using namespace std::chrono;
  return time_point_cast<nanoseconds>(system_clock::now())
      .time_since_epoch()
      .count();
}
} // namespace

// ----------------------------------------------------------------------------
// -- Python interpreter state ------------------------------------------------
// ----------------------------------------------------------------------------
struct PyEvent {
  PyEvent(int event, PyCodeObject* f_code)
      : event_(event), time_ns_(now()), f_code_(f_code) {}

  int event_;
  int64_t time_ns_;
  PyCodeObject* f_code_;
};

std::vector<PyEvent> py_events_;

int py_profile_fn(
    PyObject* obj,
    PyFrameObject* frame,
    int event,
    PyObject* arg) {
  py_events_.emplace_back(event, frame->f_code);
  return 0;
}

// ----------------------------------------------------------------------------
// -- Pytorch op dispatch state -----------------------------------------------
// ----------------------------------------------------------------------------
struct TensorMetadata {
  int64_t hash_;
};

struct IntMetadata {
  int64_t value_;
};

std::atomic<int64_t> other_id_counter_{0};
struct OtherMetadata {
  OtherMetadata() : id_(other_id_counter_++) {}
  int64_t id_;
};

using Metadata = c10::variant<TensorMetadata, IntMetadata, OtherMetadata>;

Metadata to_metadata(const c10::IValue& i) {
  if (i.isTensor()) {
    return TensorMetadata{/*hash_=*/i.hash().toInt()};
  } else if (i.isInt()) {
    return IntMetadata{/*value_=*/i.toInt()};
  }

  return OtherMetadata();
}

struct TraceObserverContext : public at::ObserverContext {
  TraceObserverContext()
      : py_events_size_(py_events_.size()), enter_ns_(now()) {}
  size_t py_events_size_;
  int64_t enter_ns_;
};

struct ObserverEvent {
  ObserverEvent(
      const char* name,
      const std::vector<c10::IValue>& in,
      const std::vector<c10::IValue>& out,
      const TraceObserverContext* ctx)
      : name_(name),
        py_events_size_(ctx->py_events_size_),
        enter_ns_(ctx->enter_ns_),
        exit_ns_(now()) {
    for (const auto& i : in) {
      inputs_.push_back(to_metadata(i));
    }
    for (const auto& o : out) {
      outputs_.push_back(to_metadata(o));
    }
  }

  std::string name_;
  std::vector<Metadata> inputs_;
  std::vector<Metadata> outputs_;
  size_t py_events_size_;
  int64_t enter_ns_;
  int64_t exit_ns_;
};

std::vector<ObserverEvent> observer_events_;

std::unique_ptr<at::ObserverContext> onFunctionEnter(
    const at::RecordFunction& fn) {
  return std::make_unique<TraceObserverContext>();
}

void onFunctionExit(
    const at::RecordFunction& fn,
    at::ObserverContext* ctx_ptr) {
  auto* trace_ctx_ptr = static_cast<TraceObserverContext*>(ctx_ptr);
  TORCH_INTERNAL_ASSERT(trace_ctx_ptr != nullptr);

  ObserverEvent e{fn.name().str(), fn.inputs(), fn.outputs(), trace_ctx_ptr};
  observer_events_.push_back(e);
}

c10::optional<at::CallbackHandle> handle_;

void enable_py_tracing() {
  TORCH_CHECK(
      !handle_.has_value(),
      "enable_py_tracing() was called, but tracing is already enabled.");
  auto callback = at::RecordFunctionCallback(onFunctionEnter, onFunctionExit)
                      .needsInputs(true)
                      .needsOutputs(true);
  handle_ = at::addGlobalCallback(callback);
  PyEval_SetProfile(py_profile_fn, NULL);
}

std::tuple<std::vector<ObserverEvent>, std::vector<PyEvent>>
disable_py_tracing() {
  TORCH_CHECK(
      handle_.has_value(),
      "disable_py_tracing() was called, but tracing is not active.");
  at::removeCallback(handle_.value());
  handle_.reset();
  PyEval_SetProfile(NULL, NULL);
  std::vector<ObserverEvent> observer_out(observer_events_);
  observer_events_.clear();

  std::vector<PyEvent> py_out(py_events_);
  py_events_.clear();

  return {observer_out, py_out};
}

void register_trace(py::module& py_module) {
  py::module trace_module = py_module.def_submodule("_trace");

  // TODO: see about actually retaining the pyobject in some circumstances.
  py::class_<PyEvent>(trace_module, "PyEvent")
      .def_property_readonly(
          "event", [](const PyEvent& self) { return self.event_; })
      .def_property_readonly(
          "time", [](const PyEvent& self) { return self.time_ns_; })
      .def_property_readonly("f_code_id", [](const PyEvent& self) {
        // NOTE: 
        //  This isn't safe, because f_code could have been collected 
        //  since the profile was collected. However for an initial hack we
        //  just YOLO cast, and we'll deal with safety later.
        return py::cast<py::object>((PyObject*)(self.f_code_));
      });

  py::class_<TensorMetadata>(trace_module, "TensorMetadata")
      .def_property_readonly(
          "hash", [](const TensorMetadata& self) { return self.hash_; });

  py::class_<IntMetadata>(trace_module, "IntMetadata")
      .def_property_readonly(
          "value", [](const IntMetadata& self) { return self.value_; });

  py::class_<OtherMetadata>(trace_module, "OtherMetadata")
      .def_property_readonly(
          "id", [](const OtherMetadata& self) { return self.id_; });

  py::class_<ObserverEvent>(trace_module, "ObserverEvent")
      .def_property_readonly(
          "name", [](const ObserverEvent& self) { return self.name_; })
      .def_property_readonly(
          "inputs", [](const ObserverEvent& self) { return self.inputs_; })
      .def_property_readonly(
          "outputs", [](const ObserverEvent& self) { return self.outputs_; })
      .def_property_readonly(
          "py_events_size",
          [](const ObserverEvent& self) { return self.py_events_size_; })
      .def_property_readonly(
          "enter_time",
          [](const ObserverEvent& self) { return self.enter_ns_; })
      .def_property_readonly(
          "exit_time", [](const ObserverEvent& self) { return self.exit_ns_; });

  trace_module.def("_enter_module_trace", &enable_py_tracing);

  trace_module.def("_exit_module_trace", &disable_py_tracing);
}

} // namespace impl
} // namespace profiler
} // namespace torch
