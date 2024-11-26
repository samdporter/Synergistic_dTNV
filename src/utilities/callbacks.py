from cil.optimisation.utilities import callbacks
from cil.framework import BlockDataContainer
from sirf.STIR import ImageData

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
        if isinstance(algo.x, ImageData):
            algo.x.write(f"{self.filename}_{algo.iteration}.hv")
        elif isinstance(algo.x, BlockDataContainer):
            for i, el in enumerate(algo.x.containers):
                el.write(f"{self.filename}_{i}_{algo.iteration}.hv")
                
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
    """
    CIL Callback that prints the objective function value.
    """
    def __init__(self, interval, **kwargs):
        super().__init__(interval, **kwargs)

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return
        print(f"Iteration {algo.iteration}, Objective {algo.objective[-1]}")