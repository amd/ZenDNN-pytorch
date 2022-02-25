#include <torch/csrc/autograd/profiler_python.h>

#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Python.h>
#include <frameobject.h>

#include <fmt/format.h>

#include <c10/macros/Macros.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/graph.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace pybind11 { namespace detail {
  template <typename... Ts>
  struct VISIBILITY_HIDDEN type_caster<c10::variant<Ts...>> : variant_caster<c10::variant<Ts...>> {};

  template <> struct type_caster<at::StringView> {
  public:
    PYBIND11_TYPE_CASTER(at::StringView, _("at::StringView"));

    static handle cast(at::StringView s, return_value_policy /* policy */, handle /* parent */) {
      return py::cast(s.str()).inc_ref();
    }
  };

  template <> struct type_caster<c10::Scalar> {
  public:
    PYBIND11_TYPE_CASTER(c10::Scalar, _("c10::Scalar"));

    static handle cast(c10::Scalar src, return_value_policy /* policy */, handle /* parent */) {
      #define CAST_TYPE(type, name)                   \
        case c10::ScalarType::name:                   \
          return py::cast(src.to##name()).inc_ref();

      switch (src.type()) {
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(CAST_TYPE)
        default:
          return py::none();
      }
      #undef CAST_TYPE
    }
  };
}} // namespace pybind11::detail

namespace torch {
namespace profiler {
namespace python_tracer {
namespace {
using torch::profiler::impl::ThreadLocalSubqueue;
using torch::profiler::impl::RecordQueue;
using torch::profiler::graph::PyFrameState;

// ============================================================================
// == Core data types =========================================================
// ============================================================================

// PyObject that allows different threads to record events without colliding.
// It is passed as the second argument when enabling tracing via
// `PyEval_SetProfile`.
struct TraceContext {
  PyObject_HEAD

  // It is wasteful to store an entire PyThreadState* in RawEvent. So
  // instead, we map thread ids down to a compact space that we can store in
  // a single byte.
  uint8_t thread_id_;
  PyThreadState* thread_state_;
};

// CPython boilerplate to define `TraceContext` as a proper python object.
static PyTypeObject TraceContextType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "TraceContext",             /* tp_name */
  sizeof(TraceContext),       /* tp_basicsize */
  0,                          /* tp_itemsize */
  nullptr,                    /* tp_dealloc */
  0,                          /* tp_vectorcall_offset */  // NOLINT: modernize-use-nullptr
  nullptr,                    /* tp_getattr */
  nullptr,                    /* tp_setattr */
  nullptr,                    /* tp_reserved */
  nullptr,                    /* tp_repr */
  nullptr,                    /* tp_as_number */
  nullptr,                    /* tp_as_sequence */
  nullptr,                    /* tp_as_mapping */
  nullptr,                    /* tp_hash  */
  nullptr,                    /* tp_call */
  nullptr,                    /* tp_str */
  nullptr,                    /* tp_getattro */
  nullptr,                    /* tp_setattro */
  nullptr,                    /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,         /* tp_flags */
  "Python tracer TLS",        /* tp_doc */
  nullptr,                    /* tp_traverse */
  nullptr,                    /* tp_clear */
  nullptr,                    /* tp_richcompare */
  0,                          /* tp_weaklistoffset */
  nullptr,                    /* tp_iter */
  nullptr,                    /* tp_iternext */
  nullptr,                    /* tp_methods */
  nullptr,                    /* tp_members */
  nullptr,                    /* tp_getset */
  nullptr,                    /* tp_base */
  nullptr,                    /* tp_dict */
  nullptr,                    /* tp_descr_get */
  nullptr,                    /* tp_descr_set */
  0,                          /* tp_dictoffset */
  nullptr,                    /* tp_init */
  nullptr,                    /* tp_alloc */
  PyType_GenericNew,          /* tp_new */
  nullptr                     /* tp_free */
};

// ============================================================================
// == Tracing implementation ==================================================
// ============================================================================
constexpr size_t max_py_threads = std::numeric_limits<uint8_t>::max() + 1;

class PythonTracer final {
 public:
  static int pyProfileFn(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg);

  int record(
    TraceContext* ctx,
    PyFrameObject* frame,
    int what,
    PyObject* arg);

  static PythonTracer& singleton();
  bool active();
  void start(const size_t max_threads, RecordQueue* record_queue);
  torch::profiler::impl::PyCodeDescriptions stop();

 private:
  PythonTracer();
  void recordPyCall(
    ThreadLocalSubqueue* subqueue,
    const uint8_t tid,
    PyFrameObject* frame);

  // Does not extend the lifetime of the function.
  interned_t internFunction(PyFrameObject* frame);
  interned_t internCFunction(PyObject* arg);

  // Extends the lifetime of `self`.
  interned_t internModule(PyFrameObject* frame);

  using DescriptionKey = std::pair</*f_code=*/PyCodeObject*, /*f_lasti=*/int>;
  struct DescriptionValue {
    PyFrameState function_;
    interned_t i_;
  };

  struct CFunctionValue {
    at::StringView name_;
    interned_t i_;
  };

  struct ModuleValue {
    at::StringView cls_name_;
    void* cls_ptr_;
    interned_t i_;
  };

  PyObject* module_call_code_;
  std::vector<std::string> path_prefixes_;
  std::vector<TraceContext*> trace_contexts_;
  RecordQueue* record_queue_;

  ska::flat_hash_map<DescriptionKey, DescriptionValue, c10::hash_pair> function_descriptions_;
  ska::flat_hash_map<void*, CFunctionValue> c_functions_;
  ska::flat_hash_map<void*, ModuleValue> modules_;
};

PythonTracer& PythonTracer::singleton() {
  static PythonTracer singleton_;
  return singleton_;
}

PythonTracer::PythonTracer() : record_queue_{nullptr} {
  path_prefixes_ = py::module::import("torch.profiler.python_tracer")
    .attr("_prefix_regex")().cast<std::vector<std::string>>();

  module_call_code_ = py::module::import("torch.nn")
    .attr("Module")
    .attr("__call__")
    .attr("__code__")
    .ptr();
}

bool PythonTracer::active() {
  return record_queue_ != nullptr;
}

void PythonTracer::start(const size_t max_threads, RecordQueue* record_queue) {
  pybind11::gil_scoped_acquire gil;
  TORCH_CHECK(!active(), "PythonTracer is already active")
  TORCH_CHECK(record_queue != nullptr, "Invalid record_queue")
  TORCH_CHECK(!trace_contexts_.size(), "PythonTracer should not have active contexts");
  TORCH_CHECK(max_threads > 0, "max_threads must be positive, got ", max_threads);
  TORCH_CHECK(
    max_threads <= max_py_threads,
    "max_threads must be less than or equal to ", max_py_threads);

  record_queue_ = record_queue;

  // Loop over all threads within the current interpreter. We will need to
  // register a trace function with each thread. We set the current thread to
  // position zero to ensure that it is traced, and so we can restore the
  // thread state after registration.
  std::vector<PyThreadState*> thread_states { PyThreadState_Get() };
  if (max_threads > 1) {
    auto thread_state = thread_states[0];
    while (thread_state != nullptr) {
      if (thread_state != thread_states[0]) {
        thread_states.push_back(thread_state);
      }
      thread_state = PyThreadState_Next(thread_state);
    }

    if (thread_states.size() > max_threads) {
      std::cout << "Warning: can only trace " << max_threads << " threads. "
                << thread_states.size() << " are currently active." << std::endl;
      thread_states.resize(max_threads);
    }
  }

  // Register the tracer in each thread.
  for (const auto i : c10::irange(thread_states.size())) {
    PyThreadState* thread_state = thread_states.at(i);
    PyThreadState_Swap(thread_state);

    auto ctx = (TraceContext*) TraceContextType.tp_alloc(&TraceContextType, 0);
    ctx->thread_id_ = (uint8_t)i;
    ctx->thread_state_ = thread_state;
    trace_contexts_.push_back(ctx);

    // When we begin profiling there are already frames on the Python
    // interpreter stack. To ensure a complete trace, we must push calls
    // to all the prior frames onto our event stack. (We stop at depth=128)
    std::vector<PyFrameObject*> current_stack;
    auto frame = PyEval_GetFrame();
    size_t depth = 0;  // Make sure we can't infinite loop.
    while (frame != nullptr && depth <= 128) {
      current_stack.push_back(frame);
      frame = frame->f_back;
      depth++;
    }
    auto* subqueue = record_queue_->getSubqueue();
    for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
      recordPyCall(subqueue, ctx->thread_id_, *it);
    }

    // Note:
    //   This profile will not compose with other CPython profilers, and
    //   cannot be round tripped via `sys.settrace(sys.gettrace())`
    PyEval_SetProfile(PythonTracer::pyProfileFn, (PyObject*)ctx);
  }

  // Restore the thread state to its initial value.
  PyThreadState_Swap(thread_states.at(0));
};

torch::profiler::impl::PyCodeDescriptions PythonTracer::stop() {
  pybind11::gil_scoped_acquire gil;
  TORCH_INTERNAL_ASSERT(active(), "PythonTracer is not running.")
  record_queue_ = nullptr;

  PyThreadState* initial_thread_state = PyThreadState_Get();
  for (const auto i : trace_contexts_) {
    PyThreadState_Swap(i->thread_state_);
    PyEval_SetProfile(nullptr, nullptr);
  }
  PyThreadState_Swap(initial_thread_state);

  torch::profiler::impl::PyCodeDescriptions out;

  out.frame_states_.resize(function_descriptions_.size() + 1);

  // The last frame has no caller.
  out.frame_states_[0] = {
    /*line_no_=  */ -1,
    /*filename_= */ at::StringView(),
    /*funcname_= */ at::StringView()
  };

  for (auto& it : function_descriptions_) {
    TORCH_INTERNAL_ASSERT(it.second.i_ > 0);
    TORCH_INTERNAL_ASSERT(it.second.i_ <= function_descriptions_.size());
    out.frame_states_[it.second.i_] = it.second.function_;
  }

  out.c_function_names_.resize(c_functions_.size());
  for (auto& it : c_functions_) {
    TORCH_INTERNAL_ASSERT(it.second.i_ <= c_functions_.size());
    out.c_function_names_[it.second.i_] = it.second.name_;
  }

  ska::flat_hash_map<void*, size_t> module_ids;
  ska::flat_hash_map<void*, size_t> module_counter;

  // TODO: can we avoid lifetime extending modules?
  out.modules_.resize(modules_.size());
  for (auto& it : modules_) {
    auto self = it.first;
    auto cls_ptr = it.second.cls_ptr_;

    auto id_it = module_ids.find(self);
    if (id_it == module_ids.end()) {
      id_it = module_ids.emplace(self, module_counter[cls_ptr]++).first;
    }

    TORCH_INTERNAL_ASSERT(it.second.i_ <= modules_.size());
    out.modules_[it.second.i_] = {id_it->second, it.second.cls_name_};
  }
  
  for (auto i : trace_contexts_) {
    Py_DECREF((PyObject*) i);
  }

  // Clear internal storage.
  function_descriptions_.clear();
  c_functions_.clear();
  modules_.clear();
  trace_contexts_.clear();

  return out;
}

// NB:
//  `frame->f_lasti` will advance as the interpreter progresses through the
//  code object. Thus, we need to call `internFunction` when we record the
//  call rather than the return. (Otherwise we would get the line with the
//  return stmt.)
interned_t PythonTracer::internFunction(PyFrameObject* frame) {
  if (frame == nullptr) {
    return 0;
  }

  const auto& it = function_descriptions_.find({ frame->f_code, frame->f_lasti });
  if (C10_UNLIKELY(it == function_descriptions_.end())) {
    DescriptionKey key {frame->f_code, frame->f_lasti};
    PyFrameState function {
      /*line_no_=  */ PyCode_Addr2Line(frame->f_code, frame->f_lasti),
      /*filename_= */ at::StringView(THPUtils_unpackString(frame->f_code->co_filename)),
      /*funcname_= */ at::StringView(THPUtils_unpackString(frame->f_code->co_name))};
    DescriptionValue value {function, function_descriptions_.size() + 1};
    function_descriptions_.emplace(key, value);
    return value.i_;

  }
  return it->second.i_;
}

void PythonTracer::recordPyCall(
    ThreadLocalSubqueue* subqueue,
    const uint8_t tid,
    PyFrameObject* frame) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame != nullptr);
  const auto caller = internFunction(frame->f_back);
  const bool is_module = ((PyObject*)(frame->f_code) == module_call_code_);
  const auto metadata = is_module ? internModule(frame) : internFunction(frame);
  subqueue->recordPyCall(is_module, tid, caller, metadata);
}

interned_t PythonTracer::internCFunction(PyObject* arg) {
  TORCH_INTERNAL_ASSERT(PyCFunction_Check(arg));
  auto key = (void*)PyCFunction_GetFunction(arg);

  const auto& it = c_functions_.find(key);
  if (C10_UNLIKELY(it == c_functions_.end())) {
    interned_t i = c_functions_.size();
    c_functions_.emplace(key, CFunctionValue{at::StringView(py::repr(arg)), i});
    return i;
  }
  return it->second.i_;
}

interned_t PythonTracer::internModule(PyFrameObject* frame) {
  // By default, CPython stores locals in a "fast" format, with an array
  // of names and an array of values. Consequently, frame->f_locals is
  // NULL since the interpreter has no need to populate it.
  //
  // If these arrays were part of the public API then we could very
  // quickly access `self`. Unfortunately they are not, and moreover are
  // not stable across versions. As a result, we are forced to call
  // `PyFrame_FastToLocals` which forces the interpreter to materialize
  // the full dict of locals.
  PyFrame_FastToLocals(frame);
  auto self = PyDict_GetItemString(frame->f_locals, "self");
  PyFrame_LocalsToFast(frame, 0);
  const auto& it = modules_.find((void*)self);
  if (C10_UNLIKELY(it == modules_.end())) {
    auto cls = py::handle(self).attr("__class__");
    ModuleValue value {
      at::StringView(py::str(cls.attr("__name__"))),
      (void*)(cls.ptr()),
      modules_.size()
    };
    modules_.emplace((void*)self, value);
    return value.i_;
  }
  return it->second.i_;
}

int PythonTracer::record(
    TraceContext* ctx,
    PyFrameObject* frame,
    int what,
    PyObject* arg) {
  auto* subqueue = record_queue_->getSubqueue();
  const auto tid = ctx->thread_id_;

  switch (what) {
    case PyTrace_CALL:
      recordPyCall(subqueue, tid, frame);
      break;

    case PyTrace_C_CALL:
      subqueue->recordPyCCall(
        tid,
        internFunction(frame->f_back),
        internCFunction(arg));
      break;

    case PyTrace_EXCEPTION:
    case PyTrace_RETURN:
      subqueue->recordPyReturn(tid, /*is_c_call=*/false);
      break;

    case PyTrace_C_EXCEPTION:
    case PyTrace_C_RETURN:
      subqueue->recordPyReturn(tid, /*is_c_call=*/true);
      break;
  }
  return 0;
}

// ============================================================================
// == API =====================================================================
// ============================================================================
int PythonTracer::pyProfileFn(
    PyObject* obj, PyFrameObject* frame, int what, PyObject* arg) {
  return PythonTracer::singleton().record(
    reinterpret_cast<TraceContext*>(obj), frame, what, arg);
}

void start(torch::profiler::impl::RecordQueue* record_queue) {
  PythonTracer::singleton().start(256, record_queue);
}

torch::profiler::impl::PyCodeDescriptions stop() {
  return PythonTracer::singleton().stop();
}
} // namespace

void init(py::module_ m) {
  pybind11::gil_scoped_acquire gil;
  TORCH_CHECK(PyType_Ready(&TraceContextType) == 0);
  torch::profiler::impl::registerPyFunctions(&start, &stop);

  using namespace torch::profiler::graph;
  auto m_graph = m.def_submodule("graph", "TorchTidy graph");

  py::enum_<EventType>(m_graph, "EventType")
    .value("TorchOp", EventType::TorchOp)
    .value("PythonFunction", EventType::PythonFunction)
    .value("PythonNNModule", EventType::PythonNNModule)
    .value("PythonCFunction", EventType::PythonCFunction)
    .value("Allocation", EventType::Allocation)
    .value("UserAnnotation", EventType::UserAnnotation)
    .value("GPUMemcpy", EventType::GPUMemcpy)
    .value("GPUMemset", EventType::GPUMemset)
    .value("ConcurrentKernal", EventType::ConcurrentKernal)
    .value("CudaRuntime", EventType::CudaRuntime);

  py::class_<TensorSummary, std::shared_ptr<TensorSummary>>(m_graph, "TensorSummary")
    .def("__repr__", [](const TensorSummary& s) { return toString(s); });
  py::class_<UndefinedTensorSummary>(m_graph, "UndefinedTensorSummary")
    .def("__repr__", [](const UndefinedTensorSummary& s){ return toString(s); });
  py::class_<OtherSummary>(m_graph, "OtherSummary")
    .def("__repr__", [](const OtherSummary& s){ return toString(s); });

  py::class_<TorchOp>(m_graph, "TorchOp")
    .def_readonly("schema", &TorchOp::schema_)
    .def_readonly("inputs", &TorchOp::inputs_)
    .def_readonly("outputs", &TorchOp::outputs_);
  py::class_<PyCall>(m_graph, "PyCall");
  py::class_<PyNNModuleForwardCall>(m_graph, "PyNNModuleForwardCall");
  py::class_<PyCCall>(m_graph, "PyCCall");
  py::class_<Allocation>(m_graph, "Allocation");
  py::class_<KinetoEvent>(m_graph, "KinetoEvent");

  py::class_<Event, std::shared_ptr<Event>>(m_graph, "Event")
    .def("__repr__", [](const Event& e) { return toString(e); })
    .def_property_readonly("type", &Event::type)
    .def_property_readonly("parent", [](const Event& e) {
      auto out = e.parent_.lock();
      return out.get() == nullptr ? py::none() : py::cast(out);
    })
    .def_readonly("children", &Event::children_)
    .def_readonly("metadata", &Event::metadata_)
    .def_readonly("start_time", &Event::start_time_)
    .def_property_readonly("end_time", [](const Event& e) {
      return e.end_time_ == torch::profiler::graph::Event::UNSET_TIME
        ? py::none() : py::cast(e.end_time_);
    })
    .def_property_readonly("duration", [](const Event& e) {
      return e.end_time_ == torch::profiler::graph::Event::UNSET_TIME
        ? py::none() : py::cast(e.end_time_ - e.start_time_);
    })
    .def_readonly("tid", &Event::tid_);

  py::class_<Graph>(m_graph, "Graph")
    .def("__repr__", [](const Graph& g) { return toString(g); })
    .def_property_readonly("head_nodes", &Graph::head_nodes);
}

} // namespace python_tracer
} // namespace profiler
} // namespace torch
