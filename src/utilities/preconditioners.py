import numpy as np
from cil.framework import BlockDataContainer
from cil.optimisation.utilities import Preconditioner
from cil.optimisation.functions import ScaledFunction
from sirf.STIR import SeparableGaussianImageFilter

class ConstantPreconditioner(Preconditioner):
    """Constant preconditioner."""
    def __init__(self, value):
        self.value = value

    def apply(self, algorithm, gradient, out=None):
        if out is None:
            out = algorithm.solution.copy()
        out.fill(gradient * self.value)
        return out


class PreconditionerWithInterval(Preconditioner):
    """Preconditioner with support for update intervals and freezing behavior."""
    def __init__(self, update_interval=1, 
                 freeze_iter=np.inf):
        self.update_interval = update_interval
        self.freeze_iter = freeze_iter
        self.freeze = None
        self.precond = None

    def apply(self, algorithm, gradient, out=None):
        """
        Apply the preconditioner, managing freezing and update intervals.
        """
        if out is None:
            out = algorithm.solution.copy()
        if algorithm.iteration < self.freeze_iter:
            if algorithm.iteration % self.update_interval == 0 or self.precond is None:
                self.precond = self.compute_preconditioner(algorithm)
            out.fill(gradient * self.precond)
        else:
            if self.freeze is None:
                self.freeze = self.compute_preconditioner(algorithm)
            out.fill(gradient * self.freeze)
        return out

    def compute_preconditioner(self, algorithm, out=None):
        """Compute the preconditioner."""
        if out is None:
            out = algorithm.solution.copy()
        raise NotImplementedError


class BSREMPreconditioner(PreconditionerWithInterval):
    """Preconditioner for BSREM."""
    def __init__(self, s_inv, update_interval=1, 
                 freeze_iter=np.inf, epsilon=None,
                 max_vals=None, smooth=True,):
        super().__init__(update_interval, freeze_iter)
        self.s_inv = s_inv
        if smooth:
            self.gaussian = SeparableGaussianImageFilter()
            self.gaussian.set_fwhms((10,10,10))
        else:
            self.gaussian = None
        if epsilon is None:
            epsilon = s_inv.max() * 1e-10
        self.epsilon = epsilon
        self.max_vals = max_vals

    def compute_preconditioner(self, algorithm, out=None):
        if out is None:
            out = algorithm.solution.copy()
        x = algorithm.solution.copy()
        if self.max_vals is not None:
            if isinstance(x, BlockDataContainer):
                for i, el in enumerate(out.containers):
                    el = el.minimum(self.max_vals[i])
                    if self.gaussian is not None:
                        self.gaussian.apply(el)
                    x.containers[i].fill(el)
            else:
                x = x.minimum(self.max_vals)
                if self.gaussian is not None:
                    self.gaussian.apply(x)
                
        if out is None:
            return (x + self.epsilon) * self.s_inv 
        out.fill((x + self.epsilon) * self.s_inv)
        return out


class ImageFunctionPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner for the prior, using the inverse Hessian diagonal.
    """
    def __init__(self, function, scale, update_interval=1, 
                 freeze_iter=np.inf, epsilon = 0,
                 max_value=np.inf):
        super().__init__(update_interval, freeze_iter)
        self.function = function
        self.scale = scale
        self.epsilon = epsilon
        self.max_value = max_value
        
    def compute_preconditioner(self, algorithm, out=None):
        precond = self.scale * self.function(algorithm.solution)
        precond = precond.maximum(self.epsilon)
        precond = precond.minimum(self.max_value)
        if out is None:
            return precond
        out.fill(precond)
        return out


class HarmonicMeanPreconditioner(PreconditionerWithInterval):
    """Preconditioner that combines two preconditioners using a harmonic mean."""
    def __init__(self, preconds, 
                 update_interval=np.inf, 
                 freeze_iter=np.inf, epsilon=1e-6,
                 scales =None
                 ):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds
        self.epsilon = epsilon
        self.scales = scales

    def compute_preconditioner(self, algorithm, out=None):
        if out is None:
            out = algorithm.solution.copy()
        a = self.preconds[0].compute_preconditioner(algorithm)
        b = self.preconds[1].compute_preconditioner(algorithm)
        if self.scales is not None:
            a.sapyb(self.scales[0], a, 0) 
            b.sapyb(self.scales[1], b, 0)
        out.fill(2 * a * b / (a + b + self.epsilon))
        return out
    
class ClampedHarmonicMeanPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner that combines two preconditioners using a Clamped harmonic mean.
    This is where the maximum of the mean cannot be greater than the maximum of each preconditioner.
    """
    def __init__(self, preconds, update_interval=np.inf, freeze_iter=np.inf, epsilon=1e-6):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds
        self.epsilon = epsilon

    def compute_preconditioner(self, algorithm, out=None):
        a = self.preconds[0].compute_preconditioner(algorithm)
        b = self.preconds[1].compute_preconditioner(algorithm)
        mean = 2 * a * b / (a + b + self.epsilon)
        mean.minimum(a, out=mean)
        mean.minimum(b, out=mean)
        if out is None:
            return mean
        out.fill(mean)
        return out
class MeanPreconditioner(PreconditionerWithInterval):
    """Preconditioner that combines two preconditioners using a simple mean."""
    def __init__(self, preconds, update_interval=np.inf, freeze_iter=np.inf):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds

    def compute_preconditioner(self, algorithm, out=None):
        if out is None:
            out = algorithm.solution.copy()
        a = self.preconds[0].compute_preconditioner(algorithm)
        b = self.preconds[1].compute_preconditioner(algorithm)
        out.fill((a + b) / 2)
        return out

class ClampedMeanPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner that combines two preconditioners using a Clamped mean.
    This is where the maximum of the mean cannot be greater than the maximum of each preconditioner.
    """
    def __init__(self, preconds, update_interval=np.inf, freeze_iter=np.inf):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds

    def compute_preconditioner(self, algorithm, out=None):
        a = self.preconds[0].compute_preconditioner(algorithm)
        b = self.preconds[1].compute_preconditioner(algorithm)
        mean = (a + b) / 2
        mean.minimum(a, out=mean)
        mean.minimum(b, out=mean)
        if out is None:
            return mean
        out.fill(mean)
        return out

class IdentityPreconditioner(PreconditionerWithInterval):
    """Identity preconditioner."""
    def __init__(self, update_interval=1, freeze_iter=np.inf):
        super().__init__(update_interval, freeze_iter)

    def compute_preconditioner(self, algorithm, out=None):
        if out is None:
            out = algorithm.solution.copy()
        out.fill(1)
        return out


class SubsetPreconditioner(PreconditionerWithInterval):
    """Base class for subset preconditioners."""
    def __init__(self, num_subsets, update_interval=1, freeze_iter=np.inf):
        super().__init__(update_interval, freeze_iter)
        self.num_subsets = num_subsets

    def compute_preconditioner(self, algorithm, out=None):
        raise NotImplementedError


class SubsetEMPreconditioner(SubsetPreconditioner):
    """
    Preconditioner for EM with subsets using sensitivities.
    Can be used for OSEM with sequential sampler or for stochastic EM with random sampler.
    """
    def __init__(self, num_subsets, sensitivities, update_interval=1, freeze_iter=np.inf, epsilon=1e-6):
        super().__init__(num_subsets, update_interval, freeze_iter)
        self.counter = 0
        self.sensitivities = sensitivities
        self.epsilon = epsilon

    def compute_preconditioner(self, algorithm, out=None):
        if out is None:
            out = algorithm.solution.copy()
        if isinstance(algorithm.f, ScaledFunction):
            adj = self.sensitivities[algorithm.f.function.data_passes_indices[-1][0]]
        else:
            adj = self.sensitivities[algorithm.f.data_passes_indices[-1][0]]
        out.fill(algorithm.solution / (adj + self.epsilon))
        return out


class SubsetKernelisedEMPreconditioner(SubsetPreconditioner):
    """
    Subset preconditioner for (hybrid) kernelised EM.
    Can be used for OS(H)KEM with sequential sampler or for stochastic (H)KEM with random sampler.
    """
    def __init__(self, num_subsets, sensitivities, kernel, update_interval=1, freeze_iter=np.inf, epsilon=1e-6):
        super().__init__(num_subsets, update_interval, freeze_iter=np.inf)
        self.counter = 0
        self.sensitivities = sensitivities
        self.kernel = kernel
        self.epsilon = epsilon
        self.frozen_alpha = None
        self.freeze_kernel_iter = freeze_iter

    def apply(self, algorithm, gradient, out=None):
        """
        Apply the preconditioner, managing freezing and update intervals.
        """
        if out is None:
            out = algorithm.solution.copy()
        if algorithm.iteration % self.update_interval == 0 or self.precond is None:
            self.precond = self.compute_preconditioner(algorithm).abs()
        out.fill(gradient * self.precond)
        return out

    def compute_preconditioner(self, algorithm, out=None):
        # for the kernelised EM, we need to freeze the alpha after a certain number of iterations
        # rather than freezing the whole preconditioner
        if algorithm.iteration >= self.freeze_kernel_iter:
            self.kernel.freeze_alpha = True
        if out is None:
            out = algorithm.solution.copy()
        if isinstance(algorithm.f, ScaledFunction):
            adj = self.sensitivities[algorithm.f.function.data_passes_indices[-1][0]]
        else:
            adj = self.sensitivities[algorithm.f.data_passes_indices[-1][0]]
        out.fill(algorithm.solution / (self.kernel.adjoint(adj)+ self.epsilon))
        return out