from cil.optimisation.utilities import Preconditioner
from cil.framework import BlockDataContainer
import numpy as np

class ConstantPreconditioner(Preconditioner):
    """
    Constant preconditioner.
    """
    def __init__(self, value):
        self.value = value

    def apply(self, algorithm, gradient, out=None):
        if out is None:
            out = algorithm.solution.copy()
        out.fill(gradient * self.value)
        return out
    
    def compute_preconditioner(self, algorithm):
        return self.value

class PreconditionerWithInterval(Preconditioner):
    """
    Preconditioner with support for update intervals and freezing behavior.
    """
    def __init__(self, update_interval=1, freeze_iter=np.inf):
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
                print("Updating preconditioner")
                self.precond = self.compute_preconditioner(algorithm)
            out.fill(gradient * self.precond)
            return out
        else:
            if self.freeze is None:
                self.freeze = self.compute_preconditioner(algorithm)
            out.fill(gradient * self.freeze)
            return out
        
    def compute_preconditioner(self, algorithm):
        """
        Compute the preconditioner.
        """
        raise NotImplementedError

    
class BSREMPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner for BSREM.
    """
    def __init__(self, s_inv, update_interval=1, freeze_iter=np.inf, epsilon=None):
        super().__init__(update_interval, freeze_iter)
        self.s_inv = s_inv
        if epsilon is None:
            epsilon = s_inv.max() * 1e-6
        self.epsilon = epsilon

    def compute_preconditioner(self, algorithm):
        return algorithm.solution * self.s_inv + self.epsilon
        
class ImageFunctionPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner for the prior, using the inverse Hessian diagonal.
    """
    def __init__(self, function, scale, update_interval=1, freeze_iter=np.inf):
        super().__init__(update_interval, freeze_iter)
        self.function = function
        self.scale = scale

    def compute_preconditioner(self, algorithm):
        return self.scale*self.function(algorithm.solution)
  
# TODO: extend these to multiple preconditioners (>2)

class HarmonicMeanPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner that combines two preconditioners using a harmonic mean.
    """
    def __init__(self, preconds, update_interval=np.inf, freeze_iter=np.inf, epsilon=1e-6):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds
        self.epsilon = epsilon

    def compute_preconditioner(self, algorithm):
        a = self.preconds[0].compute_preconditioner(algorithm)
        b = self.preconds[1].compute_preconditioner(algorithm)
        return 2 * a * b / (a + b + self.epsilon)
    
class MeanPreconditioner(PreconditionerWithInterval):
    """
    Precoditioner that combines two preconditioners using a simple mean.
    """
    def __init__(self, preconds, update_interval=np.inf, freeze_iter=np.inf):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds
        
    def compute_preconditioner(self, algorithm):
        a = self.preconds[0].compute_preconditioner(algorithm)
        b = self.preconds[1].compute_preconditioner(algorithm)
        return (a + b) / 2
    
class IdentityPreconditioner(PreconditionerWithInterval):
    """
    Identity preconditioner.
    """
    def __init__(self, update_interval=1, freeze_iter=np.inf):
        super().__init__(update_interval, freeze_iter)
        
    def compute_preconditioner(self, algorithm):
        return 1