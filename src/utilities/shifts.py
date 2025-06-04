from cil.optimisation.operators import LinearOperator
from cil.framework import BlockDataContainer
from sirf.STIR import ImageData

import numpy as np
import os


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
        range_geometry = self.initialise_shift(image, out=None)
        super().__init__(
            domain_geometry=image, 
            range_geometry=range_geometry
        )
        
        self.unshifted_image = image.copy()
        self.shifted_image = range_geometry.copy()

    def initialise_shift(self, x, out=None):
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
    
    def direct(self, x, out=None):
        
        x_arr = x.as_array()
        if out is not None:
            out.fill(x_arr)
            return out
        else:
            self.shifted_image.fill(x_arr)
            return self.shifted_image           
        

    def adjoint(self, x, out=None):

        x_arr = x.as_array()
        if out is not None:
            out.fill(x_arr)
            return out
        else:
            self.unshifted_image.fill(x_arr)
            return self.unshifted_image

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
    
def get_couch_shift_from_header(header_filepath):

    start_horizontal_bed_position = None

    # Read the file and extract the desired value
    with open(header_filepath, 'r') as file:
        for line in file:
            if line.startswith("start horizontal bed position (mm) :="):
                # Extract the value after ":="
                start_horizontal_bed_position = float(line.split(":=")[1].strip())

    if start_horizontal_bed_position is None:
        raise ValueError("Could not find 'start horizontal bed position (mm)' in the sinogram file.")

    return start_horizontal_bed_position

import re

def get_couch_shift_from_acqusition_data(sinogram) -> float:

    header = sinogram.get_info()

    pattern = r"start\s+horizontal\s+bed\s+position\s+\(mm\)\s*:=\s*([-+]?\d*\.?\d+)"
    match = re.search(pattern, header)
    if match is None:
        raise ValueError("Horizontal bed position not found.")
    return float(match.group(1))

def get_couch_shift_from_sinogram(sinogram) -> float:

    if isinstance(sinogram, str):
        return get_couch_shift_from_header(sinogram)
    else:
        return get_couch_shift_from_acqusition_data(sinogram)

class ImageCombineOperator(LinearOperator):
    def __init__(
        self, images: BlockDataContainer,
    ):
        self.images = images

        self.reference = ImageData()
        dim_xy = images.containers[0].dimensions()[1]
        dim_z = ImageCombineOperator.get_combined_length_voxels(images)
        offset_xy = images.containers[0].get_geometrical_info().get_offset()[0]
        offset_z = -images.containers[-1].get_geometrical_info().get_offset()[2]
        print(f"setting offset_z to {-offset_z}. If something goes wrong try swapping the image order")
        # for some reason, initialising as offset_xy=0 works here
        self.reference.initialise((dim_z, dim_xy, dim_xy), images.containers[0].voxel_sizes(), (offset_z, 0,0))
        self.reference = self.reference
        
        # Ensure all images have the same voxel size as the reference
        assert all(img.voxel_sizes() == self.reference.voxel_sizes() for img in images.containers), "All images must have the same voxel size as the reference"
        
        # Ensure the combined image length matches the reference dimensions
        assert self.get_combined_length_voxels(images) == self.reference.dimensions()[0], f"Combined image length and reference dimensions do not match. Something is wrong \n Combined image length: {self.get_combined_length_voxels(images)} \n Reference dimensions: {self.reference.dimensions()[0]}"
        
        super().__init__(domain_geometry=images, range_geometry=self.reference)
    
    
    @staticmethod
    def get_combined_length(images):
        offsets = [img.get_geometrical_info().get_offset()[2] for img in images.containers]
        lengths = [img.dimensions()[0] * img.voxel_sizes()[0] for img in images.containers]
        
        return max(offset + length for offset, length in zip(offsets, lengths)) - min(offsets)
    
    @staticmethod
    def get_combined_length_voxels(images):
        voxel_size = images.containers[0].voxel_sizes()[0]
        assert all(img.voxel_sizes() == images.containers[0].voxel_sizes() for img in images.containers)
        
        length = ImageCombineOperator.get_combined_length(images)
        
        assert ((length / voxel_size) % 1) - 1 < 1e-3
        return int(round(length / voxel_size))
    

    @staticmethod
    def combine_images(
        reference: ImageData,
        images: BlockDataContainer,
        sens_images: BlockDataContainer = None,
        weight_overlap: bool = False,
    ):
        """
        Combines images onto `reference`. If weight_overlap=True, then:
        - overlap mask M = (coverage_count ≥ 2)
        - num = ∑_i [S_i · f_i],  den = ∑_i [S_i]
        - out = M*(num/den) + (1−M)*∑_i[f_i]
        Else does plain ∑_i[f_i].
        """
        # zoom all images
        zoomed_imgs  = [img.zoom_image_as_template(reference)
                        for img in images.containers]

        if not weight_overlap:
            out = reference.get_uniform_copy(0)
            for z in zoomed_imgs:
                out += z
            return out

        # 1) build coverage masks (1 inside each img's FOV, 0 outside)
        zoomed_masks = [
            img.get_uniform_copy(1).zoom_image_as_template(reference)
            for img in images.containers
        ]
        cov_arrs = [m.as_array() for m in zoomed_masks]
        coverage = sum(cov_arrs)                  # integer count
        overlap = (coverage >= 2)                 # boolean mask

        # 2) zoom sensitivities and pull raw arrays
        zoomed_sens = [
            s.zoom_image_as_template(reference)
            for s in sens_images.containers
        ]
        img_arrs  = [z.as_array() for z in zoomed_imgs]
        sens_arrs = [s.as_array() for s in zoomed_sens]

        # 3) numerator, denominator, simple sum
        num    = sum(f*s for f, s in zip(img_arrs, sens_arrs))  # ∑ S_i·f_i
        den    = sum(sens_arrs)                                 # ∑ S_i
        simple = sum(img_arrs)                                  # ∑ f_i

        # 4) merge
        combined = np.where(overlap, num/den, simple)

        out = reference.get_uniform_copy(0)
        out.fill(combined)
        return out


    @staticmethod
    def retrieve_original_images(combined_image, original_references):
        """
        Retrieves the original images from the combined image.
        
        Parameters:
            combined_image: The image obtained from combine_images.
            original_references: List of original image references.
            weight_overlap: If True, adjust for overlapping regions. Default is False (ignore weighting).
        
        Returns:
            original_images: List of images zoomed to the original references.
        """
        original_images = []

        for ref in original_references:
            original_images.append(combined_image.zoom_image_as_template(ref))

        return original_images

    def direct(self, images: BlockDataContainer, out=None):
        if out is None:
            out = self.range_geometry().allocate(0)
        
        out.fill(ImageCombineOperator.combine_images(self.reference, images))
        return out
    
    def adjoint(self, image, out=None):
        if out is None:
            out = BlockDataContainer(*[img.get_uniform_copy(0) for img in self.images.containers])
        
        original_images = ImageCombineOperator.retrieve_original_images(image, self.images)
        
        for img, container in zip(original_images, out.containers):
            container.fill(img)
        
        return out