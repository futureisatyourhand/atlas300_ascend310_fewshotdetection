import numpy as np
import ctypes

package_data = np.ones((1,5664), dtype=np.float32)
mylib = ctypes.cdll.LoadLibrary("./infer.so")
rows,cols = package_data.shape
data = package_data.reshape(-1)
data = np.repeat(package_data.reshape(-1), 8)
data = (ctypes.c_float*data.shape[0])(*data)
mylib._Z5inferPfiiiiiii(data, 8, 1, 1, 5664, 8*5664, 8, 6)
