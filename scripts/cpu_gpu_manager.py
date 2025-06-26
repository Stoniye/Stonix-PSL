import numpy as np_cpu
from public_variables import USE_CUDA

USE_GPU_ENABLED = False
np = np_cpu

if USE_CUDA:
    try:
        import cupy as np_gpu
        if np_gpu.cuda.is_available():
            np = np_gpu
            USE_GPU_ENABLED = True
            print("Using CuPy (using GPU with better performance)")
        else:
            print("WARNING: 'USE_CUDA' public variable set, but no CUDA-enabled GPU found or configured. Falling back to NumPy (CPU).")
    except ImportError:
        print("WARNING: 'USE_CUDA' public variable set, but CuPy not found. Falling back to NumPy (CPU).")
else:
    print("Using NumPy (using CPU with worse performance)")