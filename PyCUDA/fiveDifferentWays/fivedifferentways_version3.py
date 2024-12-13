# -*- coding: utf-8 -*-
"""fiveDifferentWays_version3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JKss1wqi1cp2-Zfb0OrXSciReXoFVcZW

# PyCUDA installation
"""

!pip install pycuda

"""---

# Version #3: using ```gpuArrays```

The following initial code portion is the same as for Version #1 and #2.
"""

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

"""This version uses the ```gpuarray``` class. In this way, it is possible to allocate and move host memory space to device by ```gpuarray.to_gpu()```, perform the sum of the two arrays by simply using ```d_c = (d_a + d_b)``` and finally to move the result to host by the ```.get()``` method. There is no explicit declaration of ```d_c``` which automatically occurs during the execution of the ```d_c = (d_a + d_b)``` instruction.

"""

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

"""This last part is the same as for the previous versions."""

if np.array_equal(h_c, h_a + h_b):
  print("Test passed!")
else :
  print("Error!")

cuda.Context.synchronize()