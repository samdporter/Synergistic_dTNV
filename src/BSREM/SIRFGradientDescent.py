import numpy
import os
import sirf.STIR as STIR
from sirf.Utilities import examples_data_path
import numpy as np

from abc import ABC, abstractmethod

from cil.optimisation.algorithms import Algorithm 
from cil.framework import DataContainer, BlockDataContainer

class SIRFGradientDescent(Algorithm):
    def __init__(self, initial, gradient_function,
                 preconditioner, step_size,
                 **kwargs):
        
        super(SIRFGradientDescent, self).__init__(**kwargs)

        self.x = initial
        self.preconditioner = preconditioner
        self.gradient_function = gradient_function
        self.step_size = step_size

        self.subset = -1

        self.configured = True
        
    def update(self):

        # calculate our subset gradient for th eimage update
        g = self.gradient_function.gradient(self)

        # calculate our preconditioner
        precond = self.preconditioner(self)

        # calculate our step size
        ss = self.step_size(self)

        # calculate the update
        x_update = ss * precond * g

        # update the image
        self.x += x_update

        # project the image to be non-negative
        self.x.maximum(0, out=self.x)

    def epoch(self):
        # self-explanatory
        return self.iteration // self.num_subsets
    
    def update_objective(self):

        data_val, prior_val = self.gradient_function(self)

        self.loss.append([data_val + prior_val, data_val, prior_val])

        self.save_ims(self.x, "image")
        self.save_ims(self.preconditioner(self), "preconditioner")
        self.save_ims(self.gradient_function.get_full_gradient(self), "gradient")

    def save_ims(self, x, name):
        # save the image
        if isinstance(x, BlockDataContainer):
            for i, im in enumerate(self.x.containers):
                im.write(name + f'_{i}_iter_{self.iteration}.hs')
        else:
            x.write(name)


### Preconditioners ###

class Preconditioner(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def calculate_preconditioner(self, algorithm):
        pass
    
class BSREMPreconditioner(Preconditioner):
    
    def __init__(self, eps=0):
        self.eps = eps
        self.inv_avg_sens =  None

    def compute_average_sensitivity(self, algorithm):
        # Initialize average sensitivity to zero
        average_sensitivity = algorithm.x.get_uniform_copy(0)
        print('Computing average sensitivity')

        # Accumulate sensitivities from all subsets
        for s in range(algorithm.gradient_function.num_subsets):
            average_sensitivity += algorithm.gradient_function.obj_fun.get_subset_sensitivity(s).maximum(0)

        # Add a small number to avoid division by zero in the preconditioner
        self.inv_avg_sens = average_sensitivity.copy()

        # Compute reciprocal of average sensitivity, avoiding division by zero
        def fill_reciprocal(data):
            data_arr = data.as_array()
            reciprocal_arr = np.reciprocal(data_arr, where=data_arr > 0)
            data.fill(reciprocal_arr)

        # Apply fill_reciprocal to each element if BlockDataContainer, else directly
        if isinstance(self.inv_avg_sens, BlockDataContainer):
            for el in self.inv_avg_sens.containers:
                fill_reciprocal(el)
        else:
            fill_reciprocal(self.inv_avg_sens)

        print('Done computing average sensitivity')
        
    def __call__(self, algorithm):
        return self.calculate_preconditioner(algorithm)
    
    def calculate_preconditioner(self, algorithm):
        if self.inv_avg_sens is None:
            self.compute_average_sensitivity(algorithm)
        return (algorithm.x + self.eps) * self.inv_avg_sens
    
class HarmonicMeanPreconditionerBSREMPrior(Preconditioner):
    def __init__(self, update_interval, preconds):
        self.update_interval = update_interval
        self.precond = None   
        self.preconds = preconds
        
    def __call__(self, algorithm):
        return self.calculate_preconditioner(algorithm) if self.precond is None or algorithm.iteration % self.update_interval == 0 else self.precond
    
    def calculate_preconditioner(self, algorithm):
        a = self.preconds[0].calculate_preconditioner(algorithm)
        b = self.preconds[1].calculate_preconditioner(algorithm)
        self.precond = 2 * a * b / (a + b)
        return self.precond
    
class HessianDiagPreconditionerBSREMPrior(Preconditioner):
    
    def __call__(self, algorithm):
        return self.calculate_preconditioner(algorithm)
    
    def calculate_preconditioner(self, algorithm):
        return algorithm.prior.inv_hessian_diag(algorithm.x).abs()
    
class FixedPreconditioner(Preconditioner):
    
    def __init__(self, preconditioner):
        self.preconditioner = preconditioner
        
    def __call__(self, algorithm):
        return self.calculate_preconditioner(algorithm)
    
    def calculate_preconditioner(self, algorithm):
        return self.preconditioner
    
class ConditionalPreconditioner(Preconditioner):
    
    def __init__(self, preconditioners):
        self.preconditioners = preconditioners
        
    def __call__(self, algorithm):
        return self.calculate_preconditioner(algorithm)
    
    def calculate_preconditioner(self, algorithm):
        # if subset is last one, use prior preconditioner
        if algorithm.gradient_function.subset == algorithm.gradient_function.num_subsets - 1:
            return 1/(algorithm.gradient_function.num_subsets - 1) * self.preconditioners[1].calculate_preconditioner(algorithm)
        # otherwise use BSREM preconditioner
        return self.preconditioners[0].calculate_preconditioner(algorithm)
    
### Step Sizes ###

class StepSize():

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, algorithm):
        pass

class FixedStepSize(StepSize):

    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, algorithm):
        return self.step_size
    
class HarmonicDecayStepSize(StepSize):

    def __init__(self, step_size, decay):
        self.step_size = step_size
        self.decay = decay

    def __call__(self, algorithm):
        return self.step_size / (1 + algorithm.iteration * self.decay)
    
### Gradient Functions ###

class GradientFunction(ABC):
    def __init__(self, obj_fun, prior, sampler,
                 prior_is_subset=False):
        self.obj_fun = obj_fun
        self.prior = prior
        self.sampler = sampler
        self.prior_is_subset = prior_is_subset
        self.full_gradient = None
        self.subset_grads = None  # Used by classes that need to store subset gradients
        self.num_subsets = sum(self.obj_fun.get_num_subsets()) if not self.prior_is_subset else sum(self.obj_fun.get_num_subsets()) + 1
        self.subset = -1

    def __call__(self, algorithm):
        obj_val = self.obj_fun(algorithm.x)
        prior_val = -self.prior(algorithm.x)
        return obj_val, prior_val

    @abstractmethod
    def gradient(self, algorithm):
        pass

    def get_full_gradient(self, algorithm):
        return -self.prior.gradient(algorithm.x) + self.obj_fun.gradient(algorithm.x)

    def subset_gradient(self, x, subset):
        """Compute the gradient for a specific subset and add prior if necessary."""
        if not self.prior_is_subset:
            return self.obj_fun.get_subset_gradient(x, subset) - self.prior.gradient(x) / self.num_subsets
        else:
            if subset == self.num_subsets - 1:
                return -self.prior.gradient(x)
            else:
                return self.obj_fun.get_subset_gradient(x, subset)
            
class SubsetGradientFunction(GradientFunction):
    def __init__(self, obj_fun, prior, sampler, prior_is_subset=False):
        super().__init__(obj_fun, prior, sampler, prior_is_subset)
        

    def gradient(self, algorithm):
        self.subset = self.sampler()
        return self.subset_gradient(algorithm.x, self.subset)

class SVRGGradientFunction(GradientFunction):
    def __init__(self, obj_fun, prior, sampler, update_interval, store_gradients=True, prior_is_subset=False):
        super().__init__(obj_fun, prior, sampler, prior_is_subset)
        self.update_interval = update_interval
        self.store_gradients = store_gradients
        self.store_x = None
        self.counter = -1
        if store_gradients:
            self.subset_grads = []

    def anchor_subset_gradient(self, subset):
        """Retrieve the stored subset gradient in SVRG."""
        return self.subset_grads[subset] if self.store_gradients else self.subset_gradient(self.store_x, subset)
    
    def get_full_gradient(self, algorithm):
        """Compute or reset full gradient and subset gradients as needed."""
        # Initialize full gradient and clear subset gradients
        if self.store_gradients:
            self.full_gradient = algorithm.x.clone().get_uniform_copy(0)
            self.subset_grads = []

            # Calculate subset gradients and accumulate full gradient
            for s in range(sum(self.obj_fun.get_num_subsets())):
                subset_grad = self.subset_gradient(algorithm.x, s)
                self.subset_grads.append(subset_grad)
                self.full_gradient += subset_grad

            # Add prior gradient handling
            prior_grad = -self.prior.gradient(algorithm.x)
            if self.prior_is_subset:
                self.subset_grads.append(prior_grad)
            else:
                self.subset_grads = [s + prior_grad / self.num_subsets for s in self.subset_grads]

            self.full_gradient += prior_grad
        else:
            self.full_gradient = -self.prior.gradient(algorithm.x) + self.obj_fun.gradient(algorithm.x)
        return self.full_gradient

    def gradient_estimate(self, algorithm):
        """Estimate the SVRG gradient using variance reduction."""
        self.subset = self.sampler()
        subset_grad = self.subset_gradient(algorithm.x, self.subset)
        return self.num_subsets * (subset_grad - self.anchor_subset_gradient(self.subset)) + self.full_gradient

    def gradient(self, algorithm):
        """Return full gradient periodically or estimate using SVRG."""
        self.counter += 1
        if self.counter % self.update_interval() == 0:
            self.store_x = algorithm.x.copy()  # Only necessary for SVRG
            return self.get_full_gradient(algorithm)
        else:
            return self.gradient_estimate(algorithm)

class SAGAGradient(GradientFunction):
    def __init__(self, obj_fun, prior, sampler, prior_is_subset=False):
        super().__init__(obj_fun, prior, sampler, prior_is_subset)
        self.subset_grads = None

    def gradient_estimate(self, algorithm):
        """Estimate the SAGA gradient using stored subset gradients."""
        self.subset = self.sampler()
        subset_grad = self.subset_gradient(algorithm.x, self.subset)
        ret = self.num_subsets * (subset_grad - self.subset_grads[self.subset]) + self.full_gradient
        self.full_gradient += subset_grad - self.subset_grads[self.subset]
        self.subset_grads[self.subset] = subset_grad
        return ret

    def gradient(self, algorithm):
        """Initialize full gradient if needed, then estimate gradient."""
        if self.subset_grads is None:
            self.get_full_gradient(algorithm)
        return self.gradient_estimate(algorithm)
    
### Samplers ###

class Sampler(ABC):
    @abstractmethod
    def next(self):
        pass

    def __call__(self):
        return self.next()

class UniformSampler(Sampler):
    def __init__(self, num_subsets):
        self.num_subsets = num_subsets
        self.counter = -1

    def next(self):
        self.counter += 1
        return self.counter % self.num_subsets
    
class RandomSampler(Sampler):

    def __init__(self, num_subsets, 
                 probs = None,
                 replace = True):
        self.num_subsets = num_subsets
        if probs is None:
            probs = np.ones(num_subsets) / num_subsets
        self.probs = probs
        self.replace = replace
        if not self.replace:
            self.used_samples = set()

    def next(self):
        if not self.replace:
            while True:
                sample = np.random.choice(self.num_subsets, p=self.probs)
                if sample not in self.used_samples:
                    self.used_samples.add(sample)
                    return sample
        else:
            return np.random.choice(self.num_subsets, p=self.probs)
        

### update_intervals ###

class UpdateInterval(ABC):

    @abstractmethod
    def __call__(self):
        pass

class FixedUpdateInterval(UpdateInterval):

    def __init__(self, interval):
        self.interval = interval

    def __call__(self):
        return self.interval

        


        
