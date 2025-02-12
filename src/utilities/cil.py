from cil.optimisation.operators import LinearOperator
from cil.optimisation.utilities import StepSizeRule
from cil.framework import BlockDataContainer
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
        self.modify_pixel_offset("tmp_shifted.hv", self.shift, 3)

        # If `out` is provided, update it
        if out is not None:
            out.read_from_file("tmp_shifted.hv")
        else:
            out = ImageData("tmp_shifted.hv")

        # Delete the temporary file
        os.remove("tmp_shifted.hv")

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
    def modify_pixel_offset(file_path, new_offset, pixel_index):
        """
        Modify the 'first pixel offset (mm) [pixel_index]' value in an Interfile header (.hv).

        Parameters:
        -----------
        file_path : str
            The path to the Interfile header (.hv) to be modified.
        new_offset : float
            The new value for 'first pixel offset (mm) [pixel_index]'.
        """
        delete_file = False
        if isinstance(file_path, ImageData):
            print("This is supposed to be a file path but got an ImageData object. Writing to a temporary file.")
            delete_file = True
            file_path.write("tmp_shift.hv")
            file_path = "tmp_shift.hv"
        try:
            # Read the file content
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify the specific line
            for i, line in enumerate(lines):
                if line.strip().startswith(f"first pixel offset (mm) [{pixel_index}] :="):
                    lines[i] = f"first pixel offset (mm) [{pixel_index}] := {new_offset}\n"
                    break

            # Write the updated content back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)
        except Exception as e:
            raise RuntimeError(f"Failed to modify the file {file_path}: {e}")
        
        image = ImageData(file_path)

        if delete_file:
            os.remove(file_path)
        
        return image
    
def get_couch_shift_from_sinogram(sinogram):

    # If sinogram is AcquisitionData, write it to a temporary file
    if hasattr(sinogram, 'write'):
        tmp_file = "tmp_sinogram.s"
        sinogram.write(tmp_file)
        sinogram_file = tmp_file
    elif isinstance(sinogram, str):
        sinogram_file = sinogram
    else:
        raise ValueError("Input sinogram must be an AcquisitionData object or a file path as a string.")

    start_horizontal_bed_position = None

    # Read the file and extract the desired value
    try:
        with open(sinogram_file, 'r') as file:
            for line in file:
                if line.startswith("start horizontal bed position (mm) :="):
                    # Extract the value after ":="
                    start_horizontal_bed_position = float(line.split(":=")[1].strip())
                    break
    finally:
        # Remove the temporary file if created
        if hasattr(sinogram, 'write') and os.path.exists(tmp_file):
            os.remove(tmp_file)

    if start_horizontal_bed_position is None:
        raise ValueError("Could not find 'start horizontal bed position (mm)' in the sinogram file.")

    return start_horizontal_bed_position

class ImageCombineOperator(LinearOperator):
    def __init__(self, images: BlockDataContainer, sensitivities: BlockDataContainer = None, offset_z = 0):
        self.images = images

        reference = ImageData()
        dim_xy = images[0].dimensions()[1]
        dim_z = ImageCombineOperator.get_combined_length_voxels(images)
        offset_xy = images[0].get_geometrical_info().get_offset()[0]
        reference.initialise((dim_xy, dim_xy, dim_z), images[0].voxel_sizes(), (offset_xy, offset_xy, offset_z))
        self.reference = reference

        self.sensitivities = sensitivities
        
        # Ensure all images have the same voxel size as the reference
        assert all(img.voxel_sizes() == reference.voxel_sizes() for img in images)
        
        # Ensure the combined image length matches the reference dimensions
        assert self.get_combined_length_voxels(images) == reference.dimensions()[0]
        
        super().__init__(domain_geometry=images, range_geometry=reference)
    
    @staticmethod
    def get_length(image):
        return image.dimensions()[0] * image.voxel_sizes()[0]
    
    @staticmethod
    def get_offset(image):
        return image.get_geometrical_info().get_offset()[2]
    
    @staticmethod
    def get_combined_length(images):
        offsets = [ImageCombineOperator.get_offset(img) for img in images]
        lengths = [ImageCombineOperator.get_length(img) for img in images]
        
        return max(offset + length for offset, length in zip(offsets, lengths)) - min(offsets)
    
    @staticmethod
    def get_combined_length_voxels(images):
        voxel_size = images[0].voxel_sizes()[0]
        assert all(img.voxel_sizes() == images[0].voxel_sizes() for img in images)
        
        length = ImageCombineOperator.get_combined_length(images)
        
        assert ((length / voxel_size) % 1) - 1 < 1e-3
        return int(round(length / voxel_size))
    
    @staticmethod
    def combine_images(reference, images, sensitivity_images=None):
        assert all(img.voxel_sizes() == images[0].voxel_sizes() for img in images)
        assert ImageCombineOperator.get_combined_length_voxels(images) == reference.dimensions()[0]
        
        zoomed_images = [img.zoom_image_as_template(reference) for img in images]
        combined_image = reference.get_uniform_copy(0)
        
        if sensitivity_images is None:
            overlap = reference.get_uniform_copy(0)
            zoomed_ones = [img.get_uniform_copy(1).zoom_image_as_template(reference) for img in images]
            
            for img in zoomed_ones:
                overlap += img
            overlap -= 1
            
            for img in zoomed_images:
                combined_image += img
            
            combined_image -= combined_image * overlap / len(images)
        else:
            zoomed_sensitivities = [sens.zoom_image_as_template(reference) for sens in sensitivity_images]
            total_sensitivity = sum(zoomed_sensitivities, reference.get_uniform_copy(0))
            total_sensitivity = np.maximum(total_sensitivity, 1e-10)
            
            for img, sens in zip(zoomed_images, zoomed_sensitivities):
                combined_image += (sens / total_sensitivity) * img
        
        return combined_image
    
    @staticmethod
    def retrieve_original_images(combined_image, original_references, sensitivity_images=None):
        original_images = []
        
        if sensitivity_images is None:
            overlap = combined_image.get_uniform_copy(0)
            zoomed_ones = [img.get_uniform_copy(1).zoom_image_as_template(combined_image) for img in original_references]
            
            for img in zoomed_ones:
                overlap += img
            overlap -= 1
            
            combined_image -= combined_image * overlap / len(original_references)
            
            for ref in original_references:
                original_images.append(combined_image.zoom_image_as_template(ref))
        else:
            zoomed_sensitivities = [sens.zoom_image_as_template(ref) for sens, ref in zip(sensitivity_images, original_references)]
            total_sensitivity = sum(zoomed_sensitivities, combined_image.get_uniform_copy(0))
            total_sensitivity = np.maximum(total_sensitivity, 1e-10)
            
            for ref, sens in zip(original_references, zoomed_sensitivities):
                zoomed_image = (combined_image * sens) / total_sensitivity
                original_images.append(zoomed_image.zoom_image_as_template(ref))
        
        return original_images
    
    def direct(self, x, out=None):
        if out is None:
            out = self.range_geometry.get_uniform_copy(0)
        
        out.fill(self.combine_images(self.reference, self.images, self.sensitivities))
        return out
    
    def adjoint(self, x, out=None):
        if out is None:
            out = BlockDataContainer(*[img.get_uniform_copy(0) for img in self.images.containers])
        
        original_images = self.retrieve_original_images(x, self.images, self.sensitivities)
        
        for img, container in zip(original_images, out.containers):
            container.fill(img)
        
        return out