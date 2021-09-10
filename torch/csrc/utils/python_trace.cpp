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

namespace {

c10::optional<at::CallbackHandle> handle_;

int64_t now() {
  using namespace std::chrono;
  return time_point_cast<nanoseconds>(system_clock::now())
      .time_since_epoch()
      .count();
}

// ----------------------------------------------------------------------------
// -- Python interpreter state ------------------------------------------------
// ----------------------------------------------------------------------------
// https://docs.python.org/3/c-api/init.html#profiling-and-tracing

PyObject* module_call_code;

void py_init() {
  // NB: We cannot call this from `register_trace` because at that time
  // `torch.nn` is not done setting up, so instead we lazily initialize.
  if (module_call_code == nullptr) {
    module_call_code = py::module::import("torch.nn")
                           .attr("Module")
                           .attr("__call__")
                           .attr("__code__")
                           .ptr();
  }
}

py::object get_self(PyFrameObject* frame) {
  // By default `f_locals` is NULL as a performance optimization. (so the
  // interpreter can use an interal data type rather that a dict) We have
  // to tell it to materialize the dict, and then return to the fast path
  // when we're done.
  PyFrame_FastToLocals(frame);
  py::object self = py::cast<py::dict>(frame->f_locals)["self"];
  PyFrame_LocalsToFast(frame, 0);
  return self;
}

// ----------------------------------------------------------------------------
// -- Event timeline ----------------------------------------------------------
// ----------------------------------------------------------------------------
struct TorchOpCallRecord {
  int64_t id_;
};
struct TorchOpReturnRecord {
  int64_t id_;
};
struct PyModuleCallRecord {
  PyModuleCallRecord(PyFrameObject* frame) : self_(get_self(frame)) {}
  py::object self_;
};
struct PyCallRecord {
  PyCallRecord(PyFrameObject* frame)
      : f_code_(py::cast<py::object>((PyObject*)(frame->f_code))) {}
  py::object f_code_;
};
struct PyCCallRecord {
  PyCCallRecord(PyObject* f) : f_(py::cast<py::object>(f)) {}
  py::object f_;
};
struct PyReturnRecord {};
struct PyOtherRecord {
  int what_;
};

using Record = c10::variant<
    TorchOpCallRecord,
    TorchOpReturnRecord,
    PyModuleCallRecord,
    PyCallRecord,
    PyCCallRecord,
    PyReturnRecord,
    PyOtherRecord>;

Record to_pyrecord(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg) {
  switch (what) {
    case PyTrace_CALL:
      if ((PyObject*)(frame->f_code) == module_call_code) {
        return PyModuleCallRecord(frame);
      }
      return PyCallRecord(frame);

    case PyTrace_C_CALL:
      return PyCCallRecord(arg);

    case PyTrace_RETURN:
    case PyTrace_C_RETURN:
    case PyTrace_EXCEPTION:
    case PyTrace_C_EXCEPTION:
      return PyReturnRecord();

    default:
      break;
  }
  return PyOtherRecord{what};
}

// ----------------------------------------------------------------------------
// -- Op inputs and outputs ---------------------------------------------------
// ----------------------------------------------------------------------------
struct TensorMetadata {
  int64_t hash_;
  uintptr_t id_;
};

struct IntMetadata {
  int64_t value_;
};

std::atomic<int64_t> metadata_other_id_counter_{0};
struct OtherMetadata {
  OtherMetadata() : id_(metadata_other_id_counter_++) {}
  int64_t id_;
};

using Metadata = c10::variant<TensorMetadata, IntMetadata, OtherMetadata>;
Metadata to_metadata(const c10::IValue& i) {
  if (i.isTensor()) {
    void* t_ptr = THPVariable_Wrap(i.toTensor());
    auto t_id = (uintptr_t)t_ptr;
    return TensorMetadata{
        /*hash_=*/i.hash().toInt(),
        /*id_=*/t_id};
  } else if (i.isInt()) {
    return IntMetadata{/*value_=*/i.toInt()};
  }

  return OtherMetadata();
}

// ----------------------------------------------------------------------------
// -- Collection --------------------------------------------------------------
// ----------------------------------------------------------------------------
struct Event {
  Event(Record record) : time_ns_(now()), record_(record) {}

  static std::vector<Event>& events() {
    static std::vector<Event> events;
    return events;
  }

  int64_t time_ns_;
  Record record_;
};

std::atomic<int64_t> torch_record_id_counter_{0};
struct TraceObserverContext : public at::ObserverContext {
  TraceObserverContext()
      : record_id_(torch_record_id_counter_++), enter_ns_(now()) {
    TorchOpCallRecord r{record_id_};
    Event::events().emplace_back(r);
  }
  int64_t record_id_;
  int64_t enter_ns_;
};

struct ObserverEvent {
  ObserverEvent(
      const char* name,
      const std::vector<c10::IValue>& in,
      const std::vector<c10::IValue>& out,
      const TraceObserverContext* ctx)
      : name_(name),
        record_id_(ctx->record_id_),
        enter_ns_(ctx->enter_ns_),
        exit_ns_(now()) {
    for (const auto& i : in) {
      inputs_.push_back(to_metadata(i));
    }
    for (const auto& o : out) {
      outputs_.push_back(to_metadata(o));
    }
    TorchOpReturnRecord r{record_id_};
    Event::events().emplace_back(r);
  }

  static std::vector<ObserverEvent>& events() {
    static std::vector<ObserverEvent> events;
    return events;
  }

  std::string name_;
  std::vector<Metadata> inputs_;
  std::vector<Metadata> outputs_;
  int64_t record_id_;
  int64_t enter_ns_;
  int64_t exit_ns_;
};

std::unique_ptr<at::ObserverContext> onFunctionEnter(
    const at::RecordFunction& fn) {
  return std::make_unique<TraceObserverContext>();
}

void onFunctionExit(
    const at::RecordFunction& fn,
    at::ObserverContext* ctx_ptr) {
  auto* trace_ctx_ptr = static_cast<TraceObserverContext*>(ctx_ptr);
  TORCH_INTERNAL_ASSERT(trace_ctx_ptr != nullptr);
  ObserverEvent::events().emplace_back(
      fn.name().str(), fn.inputs(), fn.outputs(), trace_ctx_ptr);
}

int py_profile_fn(
    PyObject* obj,
    PyFrameObject* frame,
    int event,
    PyObject* arg) {
  Event::events().emplace_back(to_pyrecord(obj, frame, event, arg));
  return 0;
}

void enable_py_tracing() {
  TORCH_CHECK(
      !handle_.has_value(),
      "enable_py_tracing() was called, but tracing is already enabled.");
  py_init();
  auto callback = at::RecordFunctionCallback(onFunctionEnter, onFunctionExit)
                      .needsInputs(true)
                      .needsOutputs(true);
  handle_ = at::addGlobalCallback(callback);
  PyEval_SetProfile(py_profile_fn, NULL);
}

std::tuple<std::vector<ObserverEvent>, std::vector<Event>> disable_py_tracing() {
  TORCH_CHECK(
      handle_.has_value(),
      "disable_py_tracing() was called, but tracing is not active.");
  at::removeCallback(handle_.value());
  handle_.reset();
  PyEval_SetProfile(NULL, NULL);
  std::vector<ObserverEvent> observer_out(ObserverEvent::events());
  ObserverEvent::events().clear();

  std::vector<Event> events_out(Event::events());
  Event::events().clear();

  return {observer_out, events_out};
}

} // namespace

namespace torch {
namespace profiler {
namespace impl {

void register_trace(py::module& py_module) {
  py::module trace_module = py_module.def_submodule("_trace");

  py::class_<TorchOpCallRecord>(trace_module, "TorchOpCallRecord")
      .def_readonly("id", &TorchOpCallRecord::id_);
  py::class_<TorchOpReturnRecord>(trace_module, "TorchOpReturnRecord")
      .def_readonly("id", &TorchOpReturnRecord::id_);
  py::class_<PyModuleCallRecord>(trace_module, "PyModuleCallRecord")
      .def_readonly("self", &PyModuleCallRecord::self_);
  py::class_<PyCallRecord>(trace_module, "PyCallRecord")
      .def_readonly("f_code", &PyCallRecord::f_code_);
  py::class_<PyCCallRecord>(trace_module, "PyCCallRecord")
      .def_readonly("f", &PyCCallRecord::f_);
  py::class_<PyReturnRecord>(trace_module, "PyReturnRecord");
  py::class_<PyOtherRecord>(trace_module, "PyOtherRecord")
      .def_readonly("what", &PyOtherRecord::what_);

  py::class_<Event>(trace_module, "Event")
      .def_readonly("time", &Event::time_ns_)
      .def_readonly("record", &Event::record_);

  py::class_<TensorMetadata>(trace_module, "TensorMetadata")
      .def_readonly("hash", &TensorMetadata::hash_)
      .def_readonly("id", &TensorMetadata::id_);
  py::class_<IntMetadata>(trace_module, "IntMetadata")
      .def_readonly("value", &IntMetadata::value_);
  py::class_<OtherMetadata>(trace_module, "OtherMetadata")
      .def_readonly("id", &OtherMetadata::id_);

  py::class_<ObserverEvent>(trace_module, "ObserverEvent")
      .def_readonly("name", &ObserverEvent::name_)
      .def_readonly("inputs", &ObserverEvent::inputs_)
      .def_readonly("outputs", &ObserverEvent::outputs_)
      .def_readonly("id", &ObserverEvent::record_id_)
      .def_readonly("enter_time", &ObserverEvent::enter_ns_)
      .def_readonly("exit_time", &ObserverEvent::exit_ns_);

  trace_module.def("_enter_module_trace", &enable_py_tracing);
  trace_module.def("_exit_module_trace", &disable_py_tracing);
}

} // namespace impl
} // namespace profiler
} // namespace torch
