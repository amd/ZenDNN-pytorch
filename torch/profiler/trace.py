"""
$ python -c "from torch.profiler.trace import example; example()"
"""
import torch


class TraceContext:

    def __enter__(self):
        torch._C._trace._enter_module_trace()
        self.observer_events = None
        self.py_events = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.observer_events, self.py_events = torch._C._trace._exit_module_trace()

        # Just print out fields as proof of concept.
        print(self.observer_events)
        for i in self.observer_events:
            print(i.name)
            print(i.py_events_size)
            for j in i.inputs:
                print("in: ", j)
            for j in i.outputs:
                print("out:", j)
            print()
        print()
        for i in self.py_events:
            print(i.event, i.f_code_id)


def example() -> None:
    class MyModule(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.relu = torch.nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.linear(x)
            return self.relu(y)


    m = MyModule()
    with TraceContext():
        x = torch.ones((2, 2))
        m(x)
