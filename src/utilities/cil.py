from cil.optimisation.operators import LinearOperator
from cil.optimisation.utilities import StepSizeRule
from sirf.STIR import ImageData

import numpy as np
import os


class BlockIndicatorBox():
    
    def __init__(self, lower = 0, upper = np.inf):
        self.lower = lower
        self.upper = upper
        
    def __call__(self, x):
        # because we're using this as a projection, this should always return 0
        # se we'll be a bit cheeky and return 0.0
        return 0.0

    def proximal(self, x, tau, out=None):
        if out is None:
            out = x.copy()
        out.fill(x.maximum(self.lower))
        out.fill(out.minimum(self.upper))
        return out    
    
class LinearDecayStepSizeRule(StepSizeRule):
    """
    Linear decay of the step size with iteration.
    """
    def __init__(self, initial_step_size: float, decay: float):
        self.initial_step_size = initial_step_size
        self.decay = decay
        self.step_size = initial_step_size

    def get_step_size(self, algorithm):
        return self.initial_step_size / (1 + self.decay * algorithm.iteration)
    
class ZeroEndSlices(LinearOperator):
    """
    Zeros the end slices of the input image.
    Not really linear but we'll pretend it is.
    """
    
    def __init__(self, num_slices, image):
        
        self.num_slices = num_slices
        
        super().__init__(domain_geometry=image, range_geometry=image)
        
    def direct(self, x, out=None):
        if out is None:
            out = x.copy()
        out_arr = out.as_array()
        out_arr[-self.num_slices:,:,:] = 0
        out_arr[:self.num_slices,:,:] = 0
        out.fill(out_arr)
        return out
    
    def adjoint(self, x, out=None):
        return self.direct(x, out)
    
class NaNToZeroOperator(LinearOperator):
    """ Puts zeros in NaNs """

    def __init__(self, image):
        super().__init__(domain_geometry=image, range_geometry=image)

    def direct(self, x, out=None):
        if out is None:
            out = x.copy()
        out_arr = out.as_array()
        out_arr[np.isnan(out_arr)] = 0
        out.fill(out_arr)
        return out

    def adjoint(self, x, out=None):
        return self.direct(x, out)
    
class ArmijoStepSearchRule(StepSizeRule):
    """
    Armijo rule for step size for initial steps, followed by linear decay.
    """
    def __init__(self, initial_step_size: float, beta: float, 
                 max_iter: int, tol: float, steps: int,
                 maximiser=False):
        
        self.initial_step_size = initial_step_size
        self.min_step_size = initial_step_size
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.steps = steps
        self.counter = 0
        self.f_x = None
        self.maximiser = maximiser

    def get_step_size(self, algorithm):
        """
        Calculate and return the step size based on the Armijo rule.
        Step size is updated every `update_interval` iterations or during the initial steps.

        After Armijo iterations are exhausted, linear decay is applied.
        """
        # Check if we're within the initial steps or at an update interval
        if self.counter < self.steps: 
            if self.f_x is None:
                self.f_x = algorithm.f(algorithm.solution) + algorithm.g(algorithm.solution)
            precond_grad = algorithm.preconditioner.apply(algorithm, algorithm.gradient_update)

            # if x is zero and the gradient is zero, we should ignore the gradient
            is_zero = algorithm.solution.power(0)
            precond_grad_negs = precond_grad.minimum(0)
            precond_grad_pos = precond_grad.maximum(0)
            precond_grad = precond_grad_pos + precond_grad_negs * is_zero

            g_norm = algorithm.gradient_update.dot(precond_grad)
            print(f"Old Objective value: {self.f_x}")


            
            # Reset step size to initial value for the Armijo search
            step_size = self.initial_step_size
            
            # Armijo step size search
            for _ in range(self.max_iter):
                # Proximal step
                x_new = algorithm.g.proximal(algorithm.solution.copy() - step_size * precond_grad, step_size)
                f_x_new = algorithm.f(x_new) + algorithm.g(x_new)
                print(f"New Objective value: {f_x_new}")
                # Armijo condition check
                print(f"Condition value: {self.f_x - self.tol * step_size * g_norm}")
                if self.maximiser:
                    # negating the condition to turn it into a maximisation problem
                    # we still use a minus sign because step size is negative
                    if f_x_new >= self.f_x - self.tol * step_size * g_norm:
                        self.f_x = f_x_new
                        break
                else:
                    if f_x_new <= self.f_x - self.tol * step_size * g_norm:
                        self.f_x = f_x_new
                        break
                
                # Reduce step size
                step_size *= self.beta

                print(f"Step size: {step_size}")
            
            # Update the internal state with the new step size as the minimum of the current and previous step sizes
            self.min_step_size = min(step_size, self.min_step_size)
            
            if self.counter < self.steps:
                self.counter += 1
            
        return step_size
    
class CouchShiftOperator(LinearOperator):
    """
    A linear operator that shifts the couch position in an image by modifying the
    'first pixel offset (mm) [3]' value in the associated Interfile header (.hv).

    Parameters:
    -----------
    image : ImageData
        The input image whose couch position is to be shifted.
    shift : float
        The amount by which to shift the couch position along the z-axis (in mm).
    """

    def __init__(self, image, shift):
        """
        Initialize the CouchShiftOperator.

        Parameters:
        -----------
        image : ImageData
            The input image whose couch position is to be shifted.
        shift : float
            The amount by which to shift the couch position along the z-axis (in mm).
        """
        self.shift = shift
        # need to create range geometry by shifting the image
        range_geometry = self.direct(image)
        super().__init__(domain_geometry=image, range_geometry=range_geometry)

    def direct(self, x, out=None):
        """
        Apply the couch shift to the input image.

        Parameters:
        -----------
        x : ImageData
            The input image to be shifted.
        out : ImageData, optional
            If provided, the result will be stored in this object. Otherwise, a new
            ImageData object will be created.

        Returns:
        --------
        ImageData
            The shifted image.
        """
        # Write the input image to a temporary file
        x.write("tmp_shifted.hv")

        # Modify the 'first pixel offset (mm) [3]' in the temporary file
        self.modify_first_pixel_offset("tmp_shifted.hv", self.shift)

        # If `out` is provided, update it
        if out is not None:
            out.read_from_file("tmp_shifted.hv")
        else:
            out = ImageData("tmp_shifted.hv")

        return out

    def adjoint(self, x, out=None):
        """
        Revert the couch shift by setting the offset back to zero.

        Parameters:
        -----------
        x : ImageData
            The input image whose couch shift is to be reverted.
        out : ImageData, optional
            If provided, the result will be stored in this object. Otherwise, a new
            ImageData object will be created.

        Returns:
        --------
        ImageData
            The image with the couch shift reverted.
        """
        # Write the input image to a temporary file
        x.write("tmp_shifted.hv")

        # Revert the 'first pixel offset (mm) [3]' to zero in the temporary file
        self.modify_first_pixel_offset("tmp_shifted.hv", 0)

        # If `out` is provided, update it
        if out is not None:
            out.read_from_file("tmp_shifted.hv")
            
        else:
            out = ImageData("tmp_shifted.hv")

        # Delete the temporary file
        os.remove("tmp_shifted.hv")

        return out

    @staticmethod
    def modify_first_pixel_offset(file_path, new_offset):
        """
        Modify the 'first pixel offset (mm) [3]' value in an Interfile header (.hv).

        Parameters:
        -----------
        file_path : str
            The path to the Interfile header (.hv) to be modified.
        new_offset : float
            The new value for 'first pixel offset (mm) [3]'.
        """
        try:
            # Read the file content
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify the specific line
            for i, line in enumerate(lines):
                if line.strip().startswith("first pixel offset (mm) [3]"):
                    lines[i] = f"first pixel offset (mm) [3] := {new_offset}\n"
                    break

            # Write the updated content back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)
        except Exception as e:
            raise RuntimeError(f"Failed to modify the file {file_path}: {e}")


    