# -*- coding: utf-8 -*-
"""fiveDifferentWays_version4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1r067Ewwq_lP-PmO-UQGsg-Aq-xMfOchW

# PyCUDA installation
"""

!pip install pycuda

"""---

# Version 4: using ```ElementwiseKernel```

The first part of the code is the usual one.
"""

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

"""In this example, ```d_c``` is explicitly defined. This is necessary for the use of the ```ElementwiseKernel``` module."""

d_c = gpuarray.empty_like(d_a)

"""Load the ```ElementwiseKernel``` module."""

from pycuda.elementwise import ElementwiseKernel

"""The ```ElementwiseKernel``` enables to define only the kernel instructions to be elementwise executed within the kernel. Here, to generalize Version #1, a general linear combination between ```d_a``` and ```d_b``` is considered. A reference to the elementwise kernel is defined in ```lin_comb```.

"""

lin_comb = ElementwiseKernel(
        "float *d_c, float *d_a, float *d_b, float a, float b",
        "d_c[i] = a * d_a[i] + b * d_b[i]")

"""Invoke the ```lin_comb``` function."""

# --- Warmup execution
lin_comb(d_c, d_a, d_b, 2, 3)

start.record()
lin_comb(d_c, d_a, d_b, 2, 3)
end.record()
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))

"""The last part is as usual."""

h_c = d_c.get()

if np.array_equal(h_c, 2 * h_a + 3 * h_b):
  print("Test passed!")
else :
  print("Error!")

cuda.Context.synchronize()