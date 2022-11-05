from numba import jit, cuda
import numpy as np
# to measure exec time
import timeit
import copy

# normal function to run on cpu
def func(a):
    for i in range(10000000):
        a[i]+= 1

# function optimized to run on gpu
@jit(target_backend='cuda')
def func2(a):
    for i in range(10000000):
        a[i]+= 1

if __name__=="__main__":
    n = 10000000
    a = np.ones(n, dtype = np.float64)

    on_cpu = timeit.Timer(stmt='func(a)', globals={
        'func':func,
        'a': a
    })

    on_gpu = timeit.Timer(stmt='func2(a)', globals={
        'func2':func2,
        'a': a
    })

    print('On CPU:', on_cpu.repeat(3, 10))
    print('On GPU:', on_gpu.repeat(3, 10))
