"""
In last exercise, we constructed a conv2d operation with NNC IR which helped us
understand the loop-level implementation of a DL operator.  In this exercise,
we will start to see the power of DL compilers: loop-level optimizations. We
will try to speedup the conv2d operation we concstructed in last exercise from
scratch.
"""

import numpy as np
from numpy import median
import timeit

# We import the conv2d operation we constructed in last exercise
print("From Exercise 6:")
from exercise_6 import *

print("\n\n============================== Exercise 7 ======================================\n")

# First, we need to use the llvm in the background; make sure that it is
# enabled in your build.
LLVM_ENABLED = torch._C._llvm_enabled()
if not LLVM_ENABLED:
    exit(-1)

# In this exercise, we only focus on single thread.
torch.set_num_threads(1)

# Let's define utility functions to check the correctness of our optimized
# conv2d, and compare its performance to torch.conv2d.
def correctness_check():
    codegen = te.construct_codegen('llvm', stmt, [te.BufferArg(x) for x in [Pimage, Pweight, Pconv]])
    out = torch.zeros(out_shape)
    codegen.call([image, weight, out])
    torch.testing.assert_allclose(out, out_ref)

def run_conv_ref(image):
    with torch.no_grad():
        out_ref = conv2d_ref(image)

def compare_perf():
    codegen = te.construct_codegen('llvm', stmt, [te.BufferArg(x) for x in [Pimage, Pweight, Pconv]])
    out = torch.zeros(out_shape)
    # measure performance
    repeat, times = 1000, 50
    time_ori = median(timeit.repeat(lambda: run_conv_ref(image), number=times, repeat=repeat))
    time_opt = median(timeit.repeat(lambda: codegen.call([image, weight, out]), number=times, repeat=repeat))
    speedup = time_ori / time_opt
    print(f"original: {time_ori*1000:5.3f}us, compiler: {time_opt*1000:5.3f}us, speedup: {speedup:4.2f}")

# Before we start optimizing the conv2d constructed with NNC IR, let's first
# measure its performance and compare it to Pytorch eager.
loopnest = te.LoopNest(init_stmt, [Pconv])
loopnest.prepare_for_codegen()
stmt = te.simplify(loopnest.root_stmt())

correctness_check()
compare_perf()
# It passed the correctness check as we already tested in last exercise. The
# performance is, however, less than 50% of torch.conv2d.
"""
original: 3.085us, compiler: 6.896us, speedup: 0.45
"""

#########################################################
########### Vectorization ###############################
#########################################################

# Vectorization utilizes data parallelism on instruction level and has shown
# good performance improvement in many cases. Let's apply it on our conv2d to
# see if it helps!
loopnest = te.LoopNest(init_stmt, [Pconv])
# Vectorize the inner most loop.
loopnest.vectorize_inner_loops()
loopnest.prepare_for_codegen()
stmt = te.simplify(loopnest.root_stmt())
print("Vectorizated Conv2d:")
print(stmt)

"""
{
  for (int g = 0; g < 16; g++) {
    for (int h = 0; h < 28; h++) {
      for (int w = 0; w < 28; w++) {
        for (int r = 0; r < 3; r++) {
          for (int s_tail_tail = 0; s_tail_tail < 3; s_tail_tail++) {
            if ((((s_tail_tail + 2 * w) - 1>=0 ? 1 : 0) & ((s_tail_tail + 2 * w) - 1<56 ? 1 : 0)) & (((r + 2 * h) - 1>=0 ? 1 : 0) & ((r + 2 * h) - 1<56 ? 1 : 0))) {
              conv[(w + 28 * h) + 784 * g] = (conv[(w + 28 * h) + 784 * g]) + (image[((((s_tail_tail + 112 * h) + 2 * w) + 3136 * g) + 56 * r) - 57]) * (weight[(s_tail_tail + 3 * r) + 9 * g]);
            }
          }
        }
      }
    }
  }
}

"""
# Unfortunately, as shown above, the inner most loop is not vectorized because
# it is enclosed by a conditional check and there are only 4 iterations.
correctness_check()
compare_perf()
"""
original: 3.109us, compiler: 6.900us, speedup: 0.45
"""
# As expected, the performance number does not change. Vectorization, in our
# case, does not help.

#########################################################
########### Slicing #####################################
#########################################################

# As we can oberserve from the stmt, there are two computation branches in the
# operation: 1) mul-add when the conditional check is true; 2) no-op
# when the conditional check is false. These two computation patterns interleve
# as we iterate on "h" and "w". How about we separate the two into different
# stmts so for each stmt, it always follows one branch? This will avoid
# instruction prefetch misses and make the memory accesses more efficient for
# the mul-add computation.
loopnest = te.LoopNest(init_stmt, [Pconv])
# Obtain the loop handles of "h" and "w"
loops = loopnest.get_all_loopnests_for(Pconv)
loop_h, loop_w = loops[0][2], loops[0][3]
print("Loop h:")
print(loop_h)
"""
for (int h = 0; h < 28; h++) {
  for (int w = 0; w < 28; w++) {
    for (int r = 0; r < 3; r++) {
      for (int s_tail_tail = 0; s_tail_tail < 3; s_tail_tail++) {
        if ((((s_tail_tail + 2 * w) - 1>=0 ? 1 : 0) & ((s_tail_tail + 2 * w) - 1<56 ? 1 : 0)) & (((r + 2 * h) - 1>=0 ? 1 : 0) & ((r + 2 * h) - 1<56 ? 1 : 0))) {
          conv[(w + 28 * h) + 784 * g] = (conv[(w + 28 * h) + 784 * g]) + (image[((((s_tail_tail + 112 * h) + 2 * w) + 3136 * g) + 56 * r) - 57]) * (weight[(s_tail_tail + 3 * r) + 9 * g]);
        }
      }
    }
  }
}
"""

print("Loop w:")
print(loop_w)
"""
for (int w = 0; w < 28; w++) {
  for (int r = 0; r < 3; r++) {
    for (int s_tail_tail = 0; s_tail_tail < 3; s_tail_tail++) {
      if ((((s_tail_tail + 2 * w) - 1>=0 ? 1 : 0) & ((s_tail_tail + 2 * w) - 1<56 ? 1 : 0)) & (((r + 2 * h) - 1>=0 ? 1 : 0) & ((r + 2 * h) - 1<56 ? 1 : 0))) {
        conv[(w + 28 * h) + 784 * g] = (conv[(w + 28 * h) + 784 * g]) + (image[((((s_tail_tail + 112 * h) + 2 * w) + 3136 * g) + 56 * r) - 57]) * (weight[(s_tail_tail + 3 * r) + 9 * g]);
      }
    }
  }
}
"""

te.LoopNest.slice_head(loop_h, 2)
print("Sliced 2 iterations off from the head of Loop h:")
print(loop_h)
"""
for (int h = Min(0 + 2, 28, 1); h < 28; h++) {
  for (int w = 0; w < 28; w++) {
    for (int r = 0; r < 3; r++) {
      for (int s_tail_tail = 0; s_tail_tail < 3; s_tail_tail++) {
        if ((((s_tail_tail + 2 * w) - 1>=0 ? 1 : 0) & ((s_tail_tail + 2 * w) - 1<56 ? 1 : 0)) & (((r + 2 * h) - 1>=0 ? 1 : 0) & ((r + 2 * h) - 1<56 ? 1 : 0))) {
          conv[(w + 28 * h) + 784 * g] = (conv[(w + 28 * h) + 784 * g]) + (image[((((s_tail_tail + 112 * h) + 2 * w) + 3136 * g) + 56 * r) - 57]) * (weight[(s_tail_tail + 3 * r) + 9 * g]);
        }
      }
    }
  }
}
"""

te.LoopNest.slice_head(loop_w, 2)
print("Sliced 2 iterations off from the head of Loop w:")
print(loop_w)
"""
for (int w = Min(0 + 2, 28, 1); w < 28; w++) {
  for (int r = 0; r < 3; r++) {
    for (int s_tail_tail = 0; s_tail_tail < 3; s_tail_tail++) {
      if ((((s_tail_tail + 2 * w) - 1>=0 ? 1 : 0) & ((s_tail_tail + 2 * w) - 1<56 ? 1 : 0)) & (((r + 2 * h) - 1>=0 ? 1 : 0) & ((r + 2 * h) - 1<56 ? 1 : 0))) {
        conv[(w + 28 * h) + 784 * g] = (conv[(w + 28 * h) + 784 * g]) + (image[((((s_tail_tail + 112 * h) + 2 * w) + 3136 * g) + 56 * r) - 57]) * (weight[(s_tail_tail + 3 * r) + 9 * g]);
      }
    }
  }
}
"""

loopnest.prepare_for_codegen()
stmt = te.simplify(loopnest.root_stmt())
print("Sliced Conv2d:")
print(stmt)
"""
{
  for (int g = 0; g < 16; g++) {
    for (int h = 0; h < 2; h++) {
      for (int w = 0; w < 28; w++) {
        for (int r = 0; r < 3; r++) {
          for (int s_tail_tail = 0; s_tail_tail < 3; s_tail_tail++) {
            if ((((s_tail_tail + 2 * w) - 1>=0 ? 1 : 0) & ((s_tail_tail + 2 * w) - 1<56 ? 1 : 0)) & (((r + 2 * h) - 1>=0 ? 1 : 0) & ((r + 2 * h) - 1<56 ? 1 : 0))) {
              conv[(w + 28 * h) + 784 * g] = (conv[(w + 28 * h) + 784 * g]) + (image[((((s_tail_tail + 112 * h) + 2 * w) + 3136 * g) + 56 * r) - 57]) * (weight[(s_tail_tail + 3 * r) + 9 * g]);
            }
          }
        }
      }
    }
    for (int h = 2; h < 28; h++) {
      for (int w = 0; w < 2; w++) {
        for (int r = 0; r < 3; r++) {
          for (int s_tail_tail = 0; s_tail_tail < 3; s_tail_tail++) {
            if ((((s_tail_tail + 2 * w) - 1>=0 ? 1 : 0) & ((s_tail_tail + 2 * w) - 1<56 ? 1 : 0)) & (((r + 2 * h) - 1>=0 ? 1 : 0) & ((r + 2 * h) - 1<56 ? 1 : 0))) {
              conv[(w + 28 * h) + 784 * g] = (conv[(w + 28 * h) + 784 * g]) + (image[((((s_tail_tail + 112 * h) + 2 * w) + 3136 * g) + 56 * r) - 57]) * (weight[(s_tail_tail + 3 * r) + 9 * g]);
            }
          }
        }
      }
      for (int w = 2; w < 28; w++) {
        for (int r = 0; r < 3; r++) {
          for (int s_tail_tail = 0; s_tail_tail < 3; s_tail_tail++) {
            if ((((s_tail_tail + 2 * w) - 1>=0 ? 1 : 0) & ((s_tail_tail + 2 * w) - 1<56 ? 1 : 0)) & (((r + 2 * h) - 1>=0 ? 1 : 0) & ((r + 2 * h) - 1<56 ? 1 : 0))) {
              conv[(w + 28 * h) + 784 * g] = (conv[(w + 28 * h) + 784 * g]) + (image[((((s_tail_tail + 112 * h) + 2 * w) + 3136 * g) + 56 * r) - 57]) * (weight[(s_tail_tail + 3 * r) + 9 * g]);
            }
          }
        }
      }
    }
  }
}
"""

# As shown above, the stmt is separated into 3 sub-stmts: 1) 0 < h < 2 and 0 <
# w < 28, 2) 2 <= h < 28 and 0 < w < 2, 3) 2 <= h < 28 and 2 <= w < 28. For
# sub-stmt 1) and 2), there are no operations, and for sub-stmt 3), we performs
# mul-add.

# Let's check its correctness and see if it helps.
correctness_check()
compare_perf()
"""
original: 3.114us, compiler: 4.085us, speedup: 0.76
"""
# As shown, it improves the performance by 1.68x.

# There is still a large space to improve our conv2d, as its performance is
# only 76% of torch.conv2d. As a more advanced exercise, Feel free to try more
# loop-level transformations to
# further imporve it.
# Here is the list of NNC's loop-level transformations:
# https://github.com/pytorch/pytorch/blob/a319bce58d66d12cd71fbf54a145cf7b7563eebc/torch/csrc/jit/tensorexpr/tensorexpr_init.cpp#L454,
# and see their descriptions in this file:
# https://github.com/pytorch/pytorch/blob/a319bce58d66d12cd71fbf54a145cf7b7563eebc/torch/csrc/jit/tensorexpr/loopnest.h#L26
