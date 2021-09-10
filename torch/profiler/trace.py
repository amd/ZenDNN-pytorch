"""
$ python -c "from torch.profiler.trace import example; example()"
"""
import dataclasses
import textwrap
import typing

import torch


CALL_CLASSES = (
    torch._C._trace.TorchOpCallRecord,
    torch._C._trace.PyModuleCallRecord,
    torch._C._trace.PyCallRecord,
    torch._C._trace.PyCCallRecord,
)


RETURN_CLASSES = (
    torch._C._trace.TorchOpReturnRecord,
    torch._C._trace.PyReturnRecord,
)


@dataclasses.dataclass(repr=False)
class Node:
    parent: typing.Optional["Node"]
    context: typing.Any
    children: typing.List["Node"]

    def __repr__(self) -> str:
        """Very hacky recursive method to render call tree."""
        result = []
        if self.parent is None:
            result.append("Node:")
        elif isinstance(self.context, torch._C._trace.PyModuleCallRecord):
            result.extend(["", f"NN Module: {self.context.self.__class__.__name__}"])
        elif isinstance(self.context, torch._C._trace.PyCallRecord):
            f_code = self.context.f_code
            result.append(f"Python: {f_code.co_name}, {f_code.co_filename} ")
        elif isinstance(self.context, torch._C._trace.PyCCallRecord):
            result.append(f"Python (C extension): {self.context.f.__qualname__}")
        elif isinstance(self.context, torch._C._trace.ObserverEvent):
            result.append(self.context.name)
        else:
            result.append(repr(self.context))

        for child in self.children:
            result.append(textwrap.indent(repr(child), "  "))
        return "\n".join(result)


class TraceContext:

    def __enter__(self):
        self.observer_events = None
        self.py_events = None
        torch._C._trace._enter_module_trace()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.observer_events, self.events = torch._C._trace._exit_module_trace()

        observer_event_map = {e.id: e for e in self.observer_events}

        # The first event is returning from `TraceContext.__enter__`, and the 
        # last two are calling into `TraceContext.__enter__` and 
        # `torch._C._trace._exit_module_trace`. For now parsing is a bit
        # brittle, so we manually prune these values.
        self.events = self.events[1:-2]

        root = Node(parent=None, context=None, children=[])

        current: Node = root
        for event in self.events:
            if isinstance(event.record, RETURN_CLASSES):
                current = current.parent
                assert current is not None
            elif isinstance(event.record, CALL_CLASSES):
                ctx = event.record
                if isinstance(ctx, torch._C._trace.TorchOpCallRecord):
                    ctx = observer_event_map[ctx.id]
                node = Node(parent=current, context=ctx, children=[])
                current.children.append(node)
                current = node
            else:
                print(event, event.record)
                raise ValueError

        assert current is root
        print(root)


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
