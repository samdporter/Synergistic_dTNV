from cil.optimisation.utilities import callbacks
from cil.optimisation.functions import ScaledFunction
from cil.framework import BlockDataContainer
from sirf.STIR import ImageData
import pandas as pd

class Callback(callbacks.Callback):
    """
    CIL Callback but with `self.skip_iteration` checking `min(self.interval, algo.update_objective_interval)`.
    TODO: backport this class to CIL.
    """
    def __init__(self, interval, **kwargs):
        super().__init__(**kwargs)
        self.interval = interval

    def skip_iteration(self, algo) -> bool:
        return algo.iteration % min(self.interval,
                                    algo.update_objective_interval) != 0 and algo.iteration != algo.max_iteration
        
class SaveImageCallback(Callback):
    """
    CIL Callback that saves an image to disk.
    """ 
    def __init__(self, filename, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return
        if isinstance(algo.solution, ImageData):
            algo.solution.write(f"{self.filename}_{algo.iteration}.hv")
        elif isinstance(algo.solution, BlockDataContainer):
            for i, el in enumerate(algo.solution.containers):
                el.write(f"{self.filename}_{i}_{algo.iteration}.hv")

class SaveKernelisedImageCallback(Callback):
    """
    Save the alpha image to disk.
    """
    def __init__(self, filename, interval, kernel_op,**kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename
        self.kernel_op = kernel_op


    def __call__(self, algo):
        if algo.iteration % self.interval != 0:
            return
        # Save the alpha image
        image = self.kernel_op.direct(algo.solution)
        image.write(f"{self.filename}_{algo.iteration}.hv")
                
class SaveGradientUpdateCallback(Callback):
    """
    CIL Callback that saves the gradient update to disk.
    """
    def __init__(self, filename, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return
        if isinstance(algo.gradient_update, ImageData):
            algo.gradient_update.write(f"{self.filename}_{algo.iteration}.hv")
        elif isinstance(algo.gradient_update, BlockDataContainer):
            for i, el in enumerate(algo.gradient_update.containers):
                el.write(f"{self.filename}_{i}_{algo.iteration}.hv")

class PrintObjectiveCallback(Callback):
    
    def __call__(self, algo):
        if algo.iteration % algo.update_objective_interval == 0:
            print(f"iter: {algo.iteration} objective: {algo.objective[-1]}")
        
class SaveObjectiveCallback(Callback):
    """
    CIL Callback that saves the objective function value to disk.
    """
    def __init__(self, filename, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return
        pd.DataFrame(algo.objective).to_csv(f"{self.filename}.csv")
        
class SavePreconditionerCallback(Callback):
    """
    CIL Callback that saves the preconditioner to disk.
    """
    def __init__(self, filename, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename
        
    def __call__(self, algo):
        preconditioner = algo.solution.copy()
        algo.preconditioner.compute_preconditioner(algo, preconditioner)
        if self.skip_iteration(algo):
            return
        if isinstance(preconditioner, ImageData):
            preconditioner.write(f"{self.filename}_{algo.iteration}.hv")
        elif isinstance(preconditioner, BlockDataContainer):
            for i, el in enumerate(preconditioner.containers):
                el.write(f"{self.filename}_{i}_{algo.iteration}.hv")
                
class SubsetValueCallback(Callback):
    """
    CIL Callback that saves the stochastic gradient value to disk.
    """
    def __init__(self, filename, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename
        # create panda dataframe and save all subset fucntion values in it
        self.subset_values = pd.DataFrame()
        
    def __call__(self, algo):
        if self.skip_iteration(algo):
            return  
        try: func_list = algo.f.functions
        except: func_list = algo.f.function.functions
        for i, function in enumerate(func_list):
            # needs to add to new line for iteration algo.iteration
            self.subset_values.at[algo.iteration, f"Subset {i}"] = function(algo.solution)
        # add a sum at first column
        self.subset_values.at[algo.iteration, "Sum"] = sum([function(algo.solution) for function in func_list])
        self.subset_values.to_csv(f"{self.filename}.csv")
        
