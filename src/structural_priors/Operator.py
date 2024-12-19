import numpy as np
from numba import njit, prange, jit
from contextlib import contextmanager

@jit(forceobj=True)
def power_iteration(operator, input_shape, num_iterations=100):
    """
    Approximate the largest singular value of an operator using power iteration.

    Args:
        operator: The operator with defined forward and adjoint methods.
        num_iterations (int): The number of iterations to refine the approximation.

    Returns:
        float: The approximated largest singular value of the operator.
    """
    
    # Start with a random input of appropriate shape
    input_data = np.random.randn(*input_shape)
    length = len(input_shape)
    
    for i in range(num_iterations):
        # Apply forward operation
        output_data = operator.forward(input_data)
        
        # Apply adjoint operation
        input_data = operator.adjoint(output_data)
        
        # Normalize the result
        if length == 3:
            norm = fast_norm_parallel_3d(input_data)
        elif length == 4:
            norm = fast_norm_parallel_4d(input_data)

        input_data /= norm

        # print iteration on and remove previous line
        print(f'Iteration {i+1}/{num_iterations}', end='\r')
    
    return norm

@njit(parallel=True)
def fast_norm_parallel_3d(arr):
    s = arr.shape
    total = 0.0
    
    for i in prange(s[0]):
        for j in prange(s[1]):
            for k in prange(s[2]):
                total += arr[i, j, k]**2

    return np.sqrt(total)

@njit(parallel=True)
def fast_norm_parallel_4d(arr):
    s = arr.shape
    total = 0.0
    
    for i in prange(s[0]):
        for j in prange(s[1]):
            for k in prange(s[2]):
                for l in prange(s[3]):
                    total += arr[i, j, k, l]**2

    return np.sqrt(total)