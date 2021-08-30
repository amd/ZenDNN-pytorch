import time

import torch
import torch.profiler.trace


def time_code():
    st = time.time()
    x = torch.ones((1,))
    for _ in range(100):
        x += torch.ones((1,))
    print(f"{(time.time() - st) * 1e3:.2f} ms")


for _ in range(2):
    for _ in range(3):
        time_code()
    print()


for _ in range(3):
    with torch.profiler.trace.TraceContext():
        time_code()
