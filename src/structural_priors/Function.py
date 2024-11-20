from numbers import Number
import numpy as np
from numba import njit

from cil.framework import BlockDataContainer

@njit
def multiply_array(x, y):

    return np.multiply(x, y)

@njit
def divide_array(x, y, eps = 1e-10):

    return np.divide(x, y + eps)

class Function():

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return self.forward(x)

    def convex_conjugate(self, x):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError

    def proximal(self, x, tau):
        raise NotImplementedError

    def proximal_conjugate(self, x, tau):
        return x - tau * self.proximal(x / tau, 1/tau)
    
    def multiply(self, other):
        if isinstance(other, Number):
            return ScalarScaledFunction(self, other)
        elif isinstance(other, np.ndarray):
            return ArrayScaledFunction(self, other)
    
    def __mul__(self, other):
        return self.multiply(other)
    
    def __rmul__(self, other):
        return self.multiply(other)

    def __add__(self, other):
        return SumFunction([self, other])
    
    def __radd__(self, other):
        return SumFunction([self, other])
    
    def __sub__(self, other):
        return SumFunction([self, -1*other])
    
    def __rsub__(self, other):
        return SumFunction([other, -1*self])   

class SumFunction(Function):

    def __init__(self, functions):
        self.functions = functions

    def forward(self, x):
        return sum(f(x) for f in self.functions)
    
    def convex_conjugate(self, x):
        raise NotImplementedError
    
    def proximal(self, x, step_size):
        # an  approximation of the proximal operator of the sum of functions
        return sum(f.proximal(x, step_size) for f in self.functions)
    
    def gradient(self, x):
        return sum(f.gradient(x) for f in self.functions)
    
    def hessian(self, x):
        return sum(f.hessian(x) for f in self.functions)
        
class ArrayScaledFunction(Function):
    
    def __init__(self, function, weight):
        self.function = function
        self.weight = weight
        
    def __call__(self, x):
        try:
            return np.sum(self.weight * self.function.call_no_sum(x))
        except:
            return np.sum(self.weight * self.function(x))

    def convex_conjugate(self, x):
        # definitely not correct
        return np.sum(self.weight) * self.function.convex_conjugate(self.divide(x,self.weight))

    def gradient(self, x):
        
        return multiply_array(self.function.gradient(x), self.weight)

    def proximal(self, x, tau):
        return self.function.proximal(x, self.weight*tau)

    def proximal_conjugate(self, x, tau):
        
        tmp = self.divide(x, tau)
        if isinstance(tau, Number):
            val = self.function.proximal(tmp, self.weight / tau)
        else:
            val = self.function.proximal(tmp, divide_array(self.weight, tau))
        return x - tau*val
    
    def multiply(self, other):
        return ArrayScaledFunction(self.function, other*self.weight)
    
    ## overloading operations
    def __mul__(self, other):
        return self.multiply(other)
    
    def __rmul__(self, other):
        return self.multiply(other)
    
class ScalarScaledFunction(Function):
    
    def __init__(self, function, scalar):
        self.function = function
        self.scalar = scalar
        
    def __call__(self, x):
        return self.scalar * self.function(x) 

    def convex_conjugate(self, x):
        return self.scalar * self.function.convex_conjugate(x/self.scalar)

    def gradient(self, x):
        
        return self.scalar * self.function.gradient(x)

    def proximal(self, x, tau):
        return self.function.proximal(x, self.scalar*tau)
    
    def proximal_conjugate(self, x, tau):
        return self.function.proximal_conjugate(x, self.scalar*tau)
    
    def multiply(self, other):
        if isinstance(other, Number):
            return ScalarScaledFunction(self.function, self.scalar*other)
        elif isinstance(other, np.ndarray):
            return ArrayScaledFunction(self.function, self.scalar*other)

    ## overloading operations
    def __mul__(self, other):
        return self.multiply(other)
    
    def __rmul__(self, other):
        return self.multiply(other)
    
class BlockFunction(Function):

    def __init__(self, functions,*args, **kwargs):
        
        self.functions = functions

        try:
            self.L = np.max([function.L for function in self.functions])
        except:
            self.L = None

    def __call__(self, x):
        out = 0
        # put last dimension first
        x = np.moveaxis(x, -1, 0)
        for function, arr in zip(self.functions, x):
            out += function(arr)
        return out
    
    def __setitem__(self, key, value):
        self.functions[key] = value

    def __getitem__(self, key):
        return self.functions[key]
    
    def __len__(self):
        return len(self.functions)

    def convex_conjugate(self, x):
        out = 0
        # put last dimension first
        x = np.moveaxis(x, -1, 0)
        for function, arr in zip(self.functions, x):
            out += function.convex_conjugate(arr)
        return out

    def gradient(self, x):
        out = []
        # put last dimension first
        x = np.moveaxis(x, -1, 0)
        for function, arr in zip(self.functions, x):
            out.append(function.gradient(arr))
        res = np.array(out)
        # put first dimension last
        res = np.moveaxis(res, 0, -1)
        return res

    def proximal(self, x, tau):
        out = []
        # put last dimension first
        x = np.moveaxis(x, -1, 0)
        for function, arr in zip(self.functions, x):
            out.append(function.proximal(arr, tau))
        res = np.array(out)
        # put first dimension last
        res = np.moveaxis(res, 0, -1)
        return res

    def proximal_conjugate(self, x, tau):
        out = []
        # put last dimension first
        x = np.moveaxis(x, -1, 0)
        for function, arr in zip(self.functions, x):
            try:
                out.append(function.proximal_conjugate(arr, tau))
            except:
                out.append(arr - tau * function.proximal(arr / tau, 1 / tau))
        res = np.array(out)
        # put first dimension last
        res = np.moveaxis(res, 0, -1)
        return res
    
class SumFunction(Function):

    def __init__(self, functions,*args, **kwargs):
        self.functions = functions

        try:
            self.L = np.max([function.L for function in self.functions])
        except:
            self.L = None
            
    def __call__(self, x):
        out = 0
        for function in self.functions:
            out += function(x)
        return out
    
    def convex_conjugate(self, x):
        out = 0
        for function in self.functions:
            out += function.convex_conjugate(x)
        return out
    
    def gradient(self, x):
        
        out = np.zeros_like(x)
        for function in self.functions:
            out += function.gradient(x)
        return out
    
    def proximal(self, x):
        
        out = np.zeros_like(x)
        for function in self.functions:
            out += function.proximal(x)
        return out
    
    def proximal_conjugate(self, x):
        
        out = np.zeros_like(x)
        for function in self.functions:
            out += function.proximal_conjugate(x)
        return out
    
class OperatorCompositionFunction(Function):
    
    def __init__(self, function, operator):
        self.function = function
        self.operator = operator
        
    def __call__(self, x):
        return self.function(self.operator.direct(x))
    
    def convex_conjugate(self, x):
        return self.function.convex_conjugate(self.operator.direct(x))
    
    def gradient(self, x):
        return self.operator.adjoint(self.function.gradient(self.operator.direct(x)))
    
    def proximal(self, x):
        return self.operator.adjoint(self.function.proximal(self.operator.direct(x)))
    
    def proximal_conjugate(self, x, tau):
        return self.operator.adjoint(self.function.proximal_conjugate(self.operator.direct(x), tau))
    
    def hessian_diag(self, x):
        return self.operator.adjoint(self.function.hessian_diag(self.operator.direct(x)))
    
    def inv_hessian_diag(self, x):
        return self.operator.adjoint(self.function.inv_hessian_diag(self.operator.direct(x)))


def test_proximal(function, shape, num_tests, tau, epsilon=1e-5, print_output=True, conjugate=False):
    """
    Test the proximal mapping of a function numerically.

    Parameters:
    - function: instance of the Function class that has a proximal method.
    - shape: tuple, the shape of the input array to test.
    - num_tests: int, the number of tests to run.
    - tau: float, the parameter for the proximal operator.
    - epsilon: float, the perturbation size for testing around the proximal point.
    """
    for test in range(num_tests):
        # Generate a random input array
        v = np.random.randn(*shape)
        
        # Compute the proximal mapping
        if not conjugate:
            x_star = function.proximal(v, tau)
        else:
            x_star = function.proximal_conjugate(v, tau)
        
        # Compute the objective function value at x_star
        f_x_star = function(x_star) + (1/(2*tau)) * np.sum(x_star - v)**2
        
        # Verify minimization by comparing with objective function values at perturbed points
        perturbations = [epsilon * np.random.randn(*shape) for _ in range(10)]
        for perturbation in perturbations:
            x_perturbed = x_star + perturbation
            f_x_perturbed = function(x_perturbed) + (1/(2*tau)) * np.sum(x_perturbed - v)**2
            
            # Check if the objective function at x_star is less or equal than at x_perturbed
            if f_x_star > f_x_perturbed:
                print(f"Test {test+1}: Proximal mapping verification failed.")
                print(f"Objective function at x_star: {f_x_star}")
                print(f"Objective function at x_perturbed: {f_x_perturbed}")
                break
        else: # This else belongs to the for, not the if; executed if the loop doesn't break
            print(f"Test {test+1}: Proximal mapping verification passed.")


class SIRFBlockFunction(Function):
    
    def __init__(self, functions):
        self.functions = functions
        self.num_subsets = self.get_num_subsets()
        
    def __call__(self, x):
        return sum(f(el) for f, el in zip(self.functions, x.containers))
    
    def get_modality_indices(self, subset_num):
        # Get the subset information from obj_fun

        # num_subsets is a list/tuple with subset counts in modality order [modality0, modality1, ...]
        total_subsets = 0
        for modality, s in enumerate(self.num_subsets):
            if subset_num < total_subsets + s:
                subset_within_modality = subset_num - total_subsets
                break
            total_subsets += s
        else:
            raise ValueError("Subset number exceeds the total number of subsets across modalities.")

        # Print and return the modality and subset within that modality for debugging
        print(f"Modality {modality}, Subset {subset_within_modality}")
        return modality, subset_within_modality
    
    def get_num_subsets(self):
        return  [f.get_num_subsets() for f in self.functions]
        
    def gradient(self, x):
        return self.get_gradient(x)
    
    def get_gradient(self, x):
        return BlockDataContainer(*[f.get_gradient(el) for f, el in zip(self.functions, x.containers)])

    def get_single_subset_gradient(self, x, subset):

        modality, subset = self.get_modality_indices(subset)

        return BlockDataContainer(*[f.get_subset_gradient(el, subset) if i == modality else \
            el.clone().fill(0) for i, f, el in zip(range(len(self.functions)), self.functions, x.containers)])

    def get_subset_gradient(self, x, subset=0):
        if isinstance(self.get_num_subsets(), int):
            return BlockDataContainer(*[f.get_subset_gradient(el, subset) for f, el in zip(self.functions, x.containers)])   
        else:
            return self.get_single_subset_gradient(x, subset)
        
    def get_single_subset_sensitivity(self, subset):

        modality, subset = self.get_modality_indices(subset)

        return BlockDataContainer(*[f.get_subset_sensitivity(subset) if i == modality else \
            f.get_subset_sensitivity(0).get_uniform_copy(0) for i, f, in zip(range(len(self.functions)), self.functions)])
   
    def get_subset_sensitivity(self, subset_num):
        if isinstance(self.get_num_subsets(), int):
            return BlockDataContainer(*[f.get_subset_sensitivity(subset_num) for f in self.functions])
        else:
            return self.get_single_subset_sensitivity(subset_num)