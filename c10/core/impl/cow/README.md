Copy-on-Write Storage
=====================

Motivation
----------
PyTorch inherited from NumPy an optimization on the reshape function
that produces a view as an output if it can be represented as
such. This complicates the work of the PyTorch compilation stack
because, it needs to understand the representation of the input in
order to understand if the output will be a copy or an alias.

The compilation stack would rather not be concerned with such details,
motivating the Stride Agnostic PyTorch project.

To address reshape specifically, we wish to simplify the
implementation to *always* copy, but copy lazily upon modification if
we can represent the output as a view.

Backwards imcompatibiility
--------------------------
Changing reshape to produce a copy is a backwards incompatible change,
because users could be relying on the aliasing behavior, intentionally
or not.

For one release, rather than returning copy-on-write tensors, we
instead warn when users have triggered behavior in their program that
relies on the aliasing of the output and input.

The general approach to the warning's non-trivial implementation is
that a write to one view family followed by a read or write to another
view family that copy-on-write aliases the first view family's storage
will trigger a warning. A view family is a set of tensors which share
a storage.

We simulate this behavior by introducing a new concept called a
"shadow storage". Tensors that would be linked by a copy-on-write
operation will continue to share a physical storage, but the copies
will have distinct shadow storages. The shadow storage of a tensor is
inherited by views, so you can say that a view family is defined as
tensors which share a shadow storage.

So how do we detect violations? We have a generation number on both
the storage and the shadow storage. On writes to a tensor, we bump the
generations of its shadow storage and its physical storage. On reads
or writes, we check if the generation numbers match, if not, we warn.

We have a few mechanisms for tracking reads and writes:
 * reads can be tracked by const accesses to the data pointer
 * writes can be tracked by mutable accesses to the data pointer
 * writes may also be tracked via autograd, using the same mechanism
   to bump version numbers

Note that we presently are only checking via autograd, since we don't
have const access to the data pointer, so we would be way too
aggressive if we assumed every access was a real write.

Future work
-----------
* enrich the warning by flagging reads/writes to data pointer after a
  big refactoring
* analyze violations of the warning and make a decision about whether
  we require any coordination about the BC change or if we should just
  let the warning run its course
* implement the actual copy on write
* simplify the compiler stack to no longer concern itself with this
