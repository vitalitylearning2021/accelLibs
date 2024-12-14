---
layout: default
title: A couple of ways to sum vectors in Numba
---

Numba is a powerful Python library designed to accelerate numerical computations by compiling Python code to optimized machine code at runtime. It uses the `LLVM` compiler infrastructure to target CPUs and GPUs, making it an excellent tool for data-intensive tasks. With its straightforward syntax, Numba enables developers to write GPU-accelerated programs in Python, eliminating the need to work directly with low-level languages like C or CUDA.

### Why Use Numba?

- *Ease of Use*: Numba simplifies GPU programming by allowing you to write code in Python without needing to learn CUDA's intricate syntax.
- *Performance*: It delivers significant speedups by compiling functions to GPU kernels that leverage parallelism.
- *Seamless Integration*: Works with `NumPy`, allowing existing `NumPy` codebases to gain GPU acceleration with minimal changes.
- *Cross-Platform*: While it supports GPU computing through CUDA (NVIDIA GPUs), it also accelerates CPU-bound code.

### Key Features for GPU Programming
- *CUDA JIT Compilation*: Numba includes support for CUDA programming with a decorator-based approach, enabling Python functions to run directly on GPUs.
- *Thread Management*: Developers can manage grid and block dimensions to optimize GPU usage.
- *Device-Specific Functions*: Numba provides access to low-level GPU functions like shared memory and synchronization primitives.

### Version 1: writing CUDA kernels as JIT functions

This example is the corresponding example to Version 1 in [Five different ways to sum vectors in PyCUDA](PyCUDA.md). CUDA kernels can be written as Python functions exploiting the JIT mechanism, as illustrated by the code below. 

```python
import numpy as np
from numba import cuda

###################
# iDivUp FUNCTION #
###################
def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = np.int32(a)
    b = np.int32(b)
    return (a // b + 1) if (a % b != 0) else (a // b)

# --- CUDA kernel for vector addition
@cuda.jit
def deviceAdd(d_c, d_a, d_b, N):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tid < N:
        d_c[tid] = d_a[tid] + d_b[tid]

########
# MAIN #
########

# --- Initialize parameters
N = 100000
BLOCKSIZE = 256
gridDim = iDivUp(N, BLOCKSIZE)

# --- Create random vectors on the CPU
h_a = np.random.randn(N).astype(np.float32)
h_b = np.random.randn(N).astype(np.float32)
h_c = np.empty_like(h_a)

# --- Allocate GPU device memory
d_a = cuda.to_device(h_a)
d_b = cuda.to_device(h_b)
d_c = cuda.device_array_like(h_a)

# --- Warmup execution
deviceAdd[gridDim, BLOCKSIZE](d_c, d_a, d_b, N)

# --- Measure execution time
start = cuda.event()
end = cuda.event()

start.record()
deviceAdd[gridDim, BLOCKSIZE](d_c, d_a, d_b, N)
end.record()
end.synchronize()

secs = cuda.event_elapsed_time(start, end) * 1e-3
print(f"Processing time = {secs:.6f}s")

# --- Copy results from device to host
d_c.copy_to_host(h_c)

# --- Verify the result
if np.array_equal(h_c, h_a + h_b):
    print("Test passed!")
else:
    print("Error!")
```

On the GPU, the space for these random vectors is transparently allocated during the mem copy operation by `cuda.to_device` or by an explicit allocation by `cuda.device_array_like`:

```python
d_a = cuda.to_device(h_a)
d_b = cuda.to_device(h_b)
d_c = cuda.device_array_like(h_a)
```

The following rows

```python
@cuda.jit
def deviceAdd(d_c, d_a, d_b, N):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tid < N:
        d_c[tid] = d_a[tid] + d_b[tid]
```

define the deviceAdd `__global__` function appointed to perform the elementwise sum, while 

```python
deviceAdd[gridDim, BLOCKSIZE](d_c, d_a, d_b, N)
```

invokes it. 

The following line

```python
d_c.copy_to_host(h_c)
```

copies the result of the computations to the CPU.

The processing time has been `0.000224s` and thus no penalty is introduced against the [PyCUDA](PyCUDA.md) case.
