from abc import ABC, abstractmethod

class Preconditioner(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def calculate_preconditioner(self, algorithm):
        pass
    
class BSREMPreconditioner(Preconditioner):
    
    def __init__(self, eps):
        self.eps = eps
        
    def __call__(self, algorithm):
        return self.calculate_preconditioner(algorithm.x)
    
    def calculate_preconditioner(self, algorithm):
        return algorithm.x / (algorithm.average_sensitivity) + self.eps 
    
class HarmonicMeanPreconditionerBSREMPrior(Preconditioner):
    def __init__(self, update_interval, eps):
        self.update_interval = update_interval
        self.precond = None   
        self.eps = eps
        
    def __call__(self, algorithm):
        return self.calculate_preconditioner(algorithm) if self.precond is None or algorithm.iteration % self.update_interval == 0 else self.precond
    
    def calculate_preconditioner(self, algorithm):
        self.precond = 2 * algorithm.x / (algorithm.average_sensitivity + algorithm.x * algorithm.prior.hessian_diag(algorithm.x).abs()) + self.eps
        return self.precond
    
class HessianDiagPreconditionerBSREMPrior(Preconditioner):
    
    def __call__(self, algorithm):
        return self.calculate_preconditioner(algorithm.x)
    
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
        if algorithm.subset == algorithm.num_subsets - 1:
            return 1/(algorithm.num_subsets - 1) * self.preconditioners[1].calculate_preconditioner(algorithm)
        # otherwise use BSREM preconditioner
        return self.preconditioners[0].calculate_preconditioner(algorithm)