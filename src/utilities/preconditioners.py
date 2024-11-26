from cil.optimisation.utilities import Preconditioner
import numpy as np

class PreconditionerWithInterval(Preconditioner):
    """
    Preconditioner with support for update intervals and freezing behavior.
    """
    def __init__(self, update_interval=1, freeze_iter=np.inf, epsilon=1e-6):
        self.update_interval = update_interval
        self.freeze_iter = freeze_iter
        self.epsilon = epsilon
        self.freeze = None
        self.precond = None

    def apply(self, algorithm, gradient, out=None):
        """
        Apply the preconditioner, managing freezing and update intervals.
        """
        if algorithm.iteration < self.freeze_iter:
            if algorithm.iteration % self.update_interval == 0 or self.precond is None:
                self.precond = self.compute_preconditioner(algorithm)
            ret = gradient * self.precond
            if out is not None:
                out.fill(ret)
            return ret
        else:
            if self.freeze is None:
                del self.precond
                self.freeze = self.compute_preconditioner(algorithm)
            ret = gradient * self.freeze
            if out is not None:
                out.fill(ret)
            return ret
        
    def compute_preconditioner(self, algorithm):
        """
        Compute the preconditioner.
        """
        raise NotImplementedError

    
class BSREMPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner for BSREM.
    """
    def __init__(self, s_inv, update_interval=1, freeze_iter=np.inf, epsilon=1e-6):
        super().__init__(update_interval, freeze_iter, epsilon)
        self.s_inv = s_inv

    def compute_preconditioner(self, algorithm):
        return algorithm.x * self.s_inv + self.epsilon
        
class PriorInvHessianDiagPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner for the prior, using the inverse Hessian diagonal.
    """
    def __init__(self, prior, update_interval=1, freeze_iter=np.inf, epsilon=1e-6):
        super().__init__(update_interval, freeze_iter, epsilon)
        self.prior = prior

    def compute_preconditioner(self, algorithm):
        return self.prior.inv_hessian_diag(algorithm.x).abs()
    
class PriorHessianDiagPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner for the prior, using the Hessian diagonal.
    """
    def __init__(self, prior, update_interval=1, freeze_iter=np.inf, epsilon=1e-6):
        super().__init__(update_interval, freeze_iter, epsilon)
        self.prior = prior

    def compute_preconditioner(self, algorithm):
        return self.prior.hessian_diag(algorithm.x).abs()
  
# TODO: extend these to multiple preconditioners (>2)

class HarmonicMeanPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner that combines two preconditioners using a harmonic mean.
    """
    def __init__(self, preconds, update_interval=1, freeze_iter=np.inf, epsilon=1e-6):
        super().__init__(update_interval, freeze_iter, epsilon)
        self.preconds = preconds

    def compute_preconditioner(self, algorithm):
        a = self.preconds[0].compute_preconditioner(algorithm)
        b = self.preconds[1].compute_preconditioner(algorithm)
        return 2 * a * b / (a + b + self.epsilon)
    
class MeanPreconditioner(PreconditionerWithInterval):
    """
    Precoditioner that combines two preconditioners using a simple mean.
    """
    def __init__(self, preconds, update_interval=1, freeze_iter=np.inf, epsilon=1e-6):
        super().__init__(update_interval, freeze_iter, epsilon)
        self.preconds = preconds
        
    def compute_preconditioner(self, algorithm):
        a = self.preconds[0].compute_preconditioner(algorithm)
        b = self.preconds[1].compute_preconditioner(algorithm)
        return (a + b) / 2