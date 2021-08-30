import torch


class TraceContext:

    def __enter__(self):
        torch._C._trace._enter_module_trace()
        self.observer_events = None
        self.py_events = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.observer_events, self.py_events = torch._C._trace._exit_module_trace()
        # print(self.observer_events)
        # for i in self.observer_events:
        #     print(i.name)
        #     for e in i.elements:
        #         print("", e.is_input, e.tag, e.id)
        #     print()
        # print()
        # for i in self.py_events:
        #     print(i.event, i.f_code_id)
