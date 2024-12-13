---
layout: default
title: Five different ways to sum vectors in PyCUDA
---

PyCUDA is a very useful tool to embed low level programming on Graphics Processing Units (GPUs) with CUDA in a higher level programming framework provided by Python. It makes available a whole bunch of facilities to 
perform a step-by-step code debugging by checking intermediate variable values using breakpoints, simple prints, plots of vectors or images of matrices. In its interactive shell version, for example using Jupyter notebook, 
PyCUDA coding is even simpler. Jupyter is however just a possibility to be exploited locally, but free on line services also exist, like Kaggle or Google Colaboratory.

Coding with PyCUDA occurs without significant loss of performance due to the use of a high level language. Indeed, acceleration using GPU parallel programming occurs by splitting the code into a sequential or mildly 
parallelizable part to be executed on a (possibly multicore) CPU and into a massively parallelizable part to be executed on GPU. Typically, the CPU is just a controller which schedules the GPU executions so that there is no 
significant penalty when using Python. Opposite to that, GPU coding can be worked out directly using CUDA.

PyCUDA is documented at the PyCUDA homepage which contains also a tutorial, it has a GitHub page where the source code is available and issues are discussed and some useful examples are available . This notwithstanding, 
a didactic getting started guide is missing. Therefore, we decided to contribute to the topic with this post having the aim of providing a smooth introduction to coding in PyCUDA. To this end, we will discuss five different 
ways to implement the classical example by which parallel programming on GPU is taught, namely, the elementwise sum of two vectors, an operation very common in scientific computing.

Basics of CUDA programming and of Python coding will be assumed. CUDA basics prerequisites can be reached by the classical CUDA By Example book.

To form the five examples, we will consider different possibilities offered by PyCUDA, namely, using:

- the `SourceModule` module;
- the `ElementwiseKernel` module;
- the elementwise sum of two `gpuarray`’s.

Different possibilities may have different performance. For this reason, we will assess the performance of each version by the execution times on a Google Colab's Tesla T4 GPU.

---

### Version 1: using `SourceModule`
The module `SourceModule` enables coding GPU processing directly using CUDA `__global__` functions and to execute the kernels by specifying the launch grid.

```python
import numpy as np

# --- PyCUDA initialization
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

###################
# iDivUp FUNCTION #
###################
def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = np.int32(a)
    b = np.int32(b)
    return (a / b + 1) if (a % b != 0) else (a / b)

########
# MAIN #
########

start = cuda.Event()
end   = cuda.Event()

N = 100000

BLOCKSIZE = 256

h_a = np.random.randn(1, N)
h_b = np.random.randn(1, N)

h_a = h_a.astype(np.float32)
h_b = h_b.astype(np.float32)

d_a = cuda.mem_alloc(h_a.nbytes)
d_b = cuda.mem_alloc(h_b.nbytes)
d_c = cuda.mem_alloc(h_a.nbytes)

cuda.memcpy_htod(d_a, h_a)
cuda.memcpy_htod(d_b, h_b)

mod = SourceModule("""
  #include <stdio.h>
  __global__ void deviceAdd(float * __restrict__ d_c, const float * __restrict__ d_a, const float * __restrict__ d_b, const int N)
  {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    d_c[tid] = d_a[tid] + d_b[tid];
  } 
  """)

deviceAdd = mod.get_function("deviceAdd")

blockDim  = (BLOCKSIZE, 1, 1)
gridDim   = (int(iDivUp(N, BLOCKSIZE)), 1, 1)

# --- Warmup execution
deviceAdd(d_c, d_a, d_b, np.int32(N), block = blockDim, grid = gridDim)

start.record()
deviceAdd(d_c, d_a, d_b, np.int32(N), block = blockDim, grid = gridDim)
end.record() 
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))

h_c = np.empty_like(h_a)
cuda.memcpy_dtoh(h_c, d_c)

if np.array_equal(h_c, h_a + h_b):
  print("Test passed!")
else :
  print("Error!")

cuda.Context.synchronize()
```

In the code above, `SourceModule` is imported at the following line:

```python
from pycuda.compiler import SourceModule
```

Then, the `iDivUp` function, which is the analogous of the `iDivUp` function typically used in CUDA/C/C++ codes (see High Performance Programming for Soft Computing, page 103), is defined; it is used to compute the number of blocks in the launch grid:

```python
def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = np.int32(a)
    b = np.int32(b)
    return (a / b + 1) if (a % b != 0) else (a / b)
```

CUDA events (see CUDA By Example), which will be subsequently used to evaluate the execution times, are set as:

```python
start = cuda.Event()
end   = cuda.Event()
```

Later on, the number of vector elements is fixed

```python
N = 100000
```

as well as the size (`BLOCKSIZE`) of each execution block:

```python
BLOCKSIZE = 256
```

The following lines define, through the `numpy` library, the two random CPU vectors (`h_a` and `h_b`) to be transferred to GPU and summed thereon.

```python
# --- Create random vectorson the CPU
h_a = np.random.randn(1, N)
h_b = np.random.randn(1, N)

# --- Set CPU arrays as single precision
h_a = h_a.astype(np.float32)
h_b = h_b.astype(np.float32)
```

On the GPU, the space for these random vectors is allocated by the `mem_alloc` method of the `cuda.driver`:

```python
# --- Allocate GPU device memory
d_a = cuda.mem_alloc(h_a.nbytes)
d_b = cuda.mem_alloc(h_b.nbytes)
d_c = cuda.mem_alloc(h_a.nbytes)
```

Note that global memory space to contain the results of the computations is also allocated. The CPU-to-GPU memory transfers are executed by `memcpy_htod` as:

```python
# --- Memcopy from host to device
cuda.memcpy_htod(d_a, h_a)
cuda.memcpy_htod(d_b, h_b)
```

There also exist other possibilities to implement allocations and copies. One of these is offered by the `gpuArray` class and an example will be illustrated next, while another possibility is to link the CUDA runtime library (`cudart.dll`) and directly use its unwrapped functions, but this latter option is off topic for this post.

The following rows

```python
mod = SourceModule("""
  #include <stdio.h>
  __global__ void deviceAdd(float * __restrict__ d_c, const float * __restrict__ d_a, const float * __restrict__ d_b, const int N)
  {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    d_c[tid] = d_a[tid] + d_b[tid];
  } 
  """)
```

define the deviceAdd `__global__` function appointed to perform the elementwise sum, while 

```python
deviceAdd = mod.get_function("deviceAdd")
```

defines a reference to `deviceAdd`. The launch grid is set as

```python
blockDim  = (BLOCKSIZE, 1, 1)
gridDim   = (iDivUp(N, BLOCKSIZE), 1, 1)
```

while 

```python
deviceAdd(d_c, d_a, d_b, np.int32(N), block = blockDim, grid = gridDim)
```

invokes the relevant `__global__` function. 

The following lines

```python
h_c = np.empty_like(h_a)
cuda.memcpy_dtoh(h_c, d_c)
```

allow the allocation of CPU memory space to store the results and the GPU-to-CPU memory transfers. Finally, rows

```python
if np.array_equal(h_c, h_a + h_b):
  print("Test passed!")
else :
  print("Error!")
```

check whether the GPU computation is correct by comparing the results with an analogous CPU computation.

Line

```python
cuda.Context.synchronize()
```

has no effect in this code, but is kept for convenience. Whenever one decides to test the code into an interactive python shell and to use `printf()` within the `__global__` function, such instructions would enable the flush of the `printf()` buffer. Without those, the `printf()` whould have no effect into an interactive python shell.

The processing time has been `0.000224s`.

---

### Version 2: using `SourceModule` and copying data from host to device on-the-fly
The second version is the same as the foregoing one with the only exception that the copies from host to the device and viceversa are not performed explicitly before the kernel launch, but rather implicitly. 

```python
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

###################
# iDivUp FUNCTION #
###################
def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = np.int32(a)
    b = np.int32(b)
    return (a / b + 1) if (a % b != 0) else (a / b)

########
# MAIN #
########

start = cuda.Event()
end   = cuda.Event()

N = 100000

BLOCKSIZE = 256

h_a = np.random.randn(1, N)
h_b = np.random.randn(1, N)

h_a = h_a.astype(np.float32)
h_b = h_b.astype(np.float32)
h_c = np.empty_like(h_a)

mod = SourceModule("""
__global__ void deviceAdd(float * __restrict__ d_c, const float * __restrict__ d_a, 
                                                    const float * __restrict__ d_b,
                                                    const int N)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N) return;
  d_c[tid] = d_a[tid] + d_b[tid];
}
""")

deviceAdd = mod.get_function("deviceAdd")
blockDim  = (BLOCKSIZE, 1, 1)
gridDim   = (int(iDivUp(N, BLOCKSIZE)), 1, 1)

# --- Warmup execution
deviceAdd(cuda.Out(h_c), cuda.In(h_a), cuda.In(h_b), np.int32(N), block = blockDim, grid = gridDim)

start.record()
deviceAdd(cuda.Out(h_c), cuda.In(h_a), cuda.In(h_b), np.int32(N), block = blockDim, grid = gridDim)
end.record() 
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))

if np.array_equal(h_c, h_a + h_b):
  print("Test passed!")
else :
  print("Error!")

cuda.Context.synchronize()
```

Implicit copies are executed on-the-fly by applying `cuda.In` to the host input arrays and `cuda.Out` to the output host array:

```python
deviceAdd(cuda.Out(h_c), cuda.In(h_a), cuda.In(h_b), np.int32(N), block = blockDim, grid = gridDim)
```

The code is now shorter, but simplicity is paid with execution times. Indeed, memory transfers now affect the computation time which becomes `0.000948s`.

---

### Version 3: using `gpuArray`'s
In the third version, GPU arrays are dealt with by the `gpuarray` class. The elementwise sum is then performed by using the possibility offered by such a class of expressing array operations on the GPU with the classical `numpy` array syntax without explicitly coding a `__global__` function and using `SourceModule`.

```python
import numpy as np

# --- PyCUDA initialization
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

########
# MAIN #
########

start = cuda.Event()
end   = cuda.Event()

N = 100000

h_a = np.random.randn(1, N)
h_b = np.random.randn(1, N)

h_a = h_a.astype(np.float32)
h_b = h_b.astype(np.float32)
h_c = np.empty_like(h_a)

d_a = gpuarray.to_gpu(h_a)
d_b = gpuarray.to_gpu(h_b)

# --- Warmup execution
d_c = (d_a + d_b)

start.record()
d_c = (d_a + d_b)
end.record() 
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))

h_c = d_c.get()

if np.array_equal(h_c, h_a + h_b):
  print("Test passed!")
else :
  print("Error!")

cuda.Context.synchronize()
```

As compared to the first version, we have now a timing penalty since the elementwise execution requires `0.000420s`.

---

### Version 4: using `ElementwiseKernel`
The PyCUDA `ElementwiseKernel` class allows to define snippets of C code to be executed elementwise. Since the `__global__ deviceAdd` function contains operations to be executed elementwise on the involved vectors, we are suggested to replace the use of `SourceModule` with `ElementwiseKernel`.

```python
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

########
# MAIN #
########

start = cuda.Event()
end   = cuda.Event()

N = 100000

h_a = np.random.randn(1, N)
h_b = np.random.randn(1, N)

h_a = h_a.astype(np.float32)
h_b = h_b.astype(np.float32)

d_a = gpuarray.to_gpu(h_a)
d_b = gpuarray.to_gpu(h_b)

In this example, ```d_c``` is explicitly defined. This is necessary for the use of the ```ElementwiseKernel``` module.

d_c = gpuarray.empty_like(d_a)

Load the ```ElementwiseKernel``` module.

from pycuda.elementwise import ElementwiseKernel

lin_comb = ElementwiseKernel(
        "float *d_c, float *d_a, float *d_b, float a, float b",
        "d_c[i] = a * d_a[i] + b * d_b[i]")

# --- Warmup execution
lin_comb(d_c, d_a, d_b, 2, 3)

start.record()
lin_comb(d_c, d_a, d_b, 2, 3)
end.record()
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))

h_c = d_c.get()

if np.array_equal(h_c, 2 * h_a + 3 * h_b):
  print("Test passed!")
else :
  print("Error!")

cuda.Context.synchronize()
```

The code above conceptually represents version 1 with `SourceModule` replaced with `ElementwiseKernel`. Actually, now a linear combination of the involved vectors instead of a simple elementwise sum is performed. The lines 

```python
lin_comb = ElementwiseKernel(
        "float *d_c, float *d_a, float *d_b, float a, float b",
        "d_c[i] = a * d_a[i] + b * d_b[i]",
        "linear_combination")
```

define the elementwise linear combination function `lin_comb` while line 

```python
lin_comb(d_c, d_a, d_b, 2, 3)
```

calls it. In this way, it is also possible to illustrate how passing constant values.

The computation time is `0.000220s` so that, as compared to version 1, `ElementwiseKernel` seems not to give rise to a loss of performance as compared to `SourceModule`.

---

### Version 5: using `SourceModule` while handling vectors by `gpuArray`

With the aim of verifying whether gpuarray is responsible of the increase of the execution times of the previous version, the code below reconsiders version 1 while dealing now the vectors by gpuarray instead of `mem_alloc`.

```python
import numpy as np

# --- PyCUDA initialization
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

###################
# iDivUp FUNCTION #
###################
def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = np.int32(a)
    b = np.int32(b)
    return (a / b + 1) if (a % b != 0) else (a / b)

########
# MAIN #
########

start = cuda.Event()
end   = cuda.Event()

N = 100000

BLOCKSIZE = 256

# --- Create random vectorson the CPU
h_a = np.random.randn(1, N)
h_b = np.random.randn(1, N)

# --- Set CPU arrays as single precision
h_a = h_a.astype(np.float32)
h_b = h_b.astype(np.float32)
h_c = np.empty_like(h_a)

mod = SourceModule("""
__global__ void deviceAdd(float * __restrict__ d_c, const float * __restrict__ d_a, 
                                                    const float * __restrict__ d_b,
                                                    const int N)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N) return;
  d_c[tid] = d_a[tid] + d_b[tid];
}
""")

d_a = gpuarray.to_gpu(h_a)
d_b = gpuarray.to_gpu(h_b)
d_c = gpuarray.zeros_like(d_a)

# --- Define a reference to the __global__ function and call it
deviceAdd = mod.get_function("deviceAdd")
blockDim  = (BLOCKSIZE, 1, 1)
gridDim   = (int(iDivUp(N, BLOCKSIZE)), 1, 1)

# --- Warmup execution
deviceAdd(d_c, d_a, d_b, np.int32(N), block = blockDim, grid = gridDim)

start.record()
deviceAdd(d_c, d_a, d_b, np.int32(N), block = blockDim, grid = gridDim)
end.record() 
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))

h_c = d_c.get()

if np.array_equal(h_c, h_a + h_b):
  print("Test passed!")
else :
  print("Error!")

# --- Flush context printf buffer
cuda.Context.synchronize()
```

The execution time keeps `0.000197s`, comparable to version 1.
