import os
import numpy as np
import logging
from typing import List, Dict
from sirf.STIR import AcquisitionData, ImageData

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def create_spect_uniform_image(sinogram, xy=None, origin=None):
    """
    Create a uniform image for SPECT data based on the sinogram dimensions.
    Adjusts the z-direction voxel size and image dimensions to create a template
    image.

    Args:
        sinogram (AcquisitionData): The SPECT sinogram.
        origin (tuple, optional): The origin of the image. Defaults to (0, 0, 0)
            if not provided.

    Returns:
        ImageData: A uniform SPECT image initialized with the computed dimensions
            and voxel sizes.
    """
    # Create a uniform image from the sinogram and adjust z-voxel size.
    print(type(xy))
    image = sinogram.create_uniform_image(value=1, xy=int(xy))
    voxel_size = list(image.voxel_sizes())
    voxel_size[0] *= 2  # Adjust z-direction voxel size.

    # Compute new dimensions based on the uniform image.
    dims = list(image.dimensions())
    dims[0] = dims[0] // 2 + dims[0] % 2  # Halve the first dimension (with rounding)
    dims[1] -= dims[1] % 2                # Ensure even number for second dimension
    dims[2] = dims[1]                     # Set third dimension equal to second dimension

    if origin is None:
        origin = (0, 0, 0)

    # Initialize a new image with computed dimensions, voxel sizes, and origin.
    new_image = ImageData()
    new_image.initialise(tuple(dims), tuple(voxel_size), tuple(origin))
    return new_image


def get_pet_data(path: str, suffix: str = "") -> dict:
    """
    Load PET data from the given path.
    
    This function always loads a template image and then attempts to load the
    initial image. If the initial image is not found, it creates a uniform copy
    of the template image (filled with ones).

    Args:
        path (str): Path to the data directory.
        suffix (str): Optional suffix appended to filenames.

    Returns:
        dict: A dictionary with keys: "acquisition_data", "additive",
        "normalisation", "attenuation", "template_image", "initial_image", and
        optionally "spect".
    """
    pet_data = {}
    pet_data["acquisition_data"] = AcquisitionData(
        os.path.join(path, f"prompts{suffix}.hs")
    )
    pet_data["additive"] = AcquisitionData(
        os.path.join(path, f"additive_term{suffix}.hs")
    )
    pet_data["normalisation"] = AcquisitionData(
        os.path.join(path, f"mult_factors{suffix}.hs")
    )
    pet_data["attenuation"] = ImageData(os.path.join(path, f"umap_zoomed.hv"))

    # Always load the template image.
    template_img_path = os.path.join(path, f"template_image.hv")
    try:
        pet_data["template_image"] = ImageData(template_img_path)
    except Exception as e_template:
        logging.error("Failed to load PET template image (%s)", str(e_template))
        raise RuntimeError("Unable to load PET template image.") from e_template

    # Try to load the initial image.
    initial_img_path = os.path.join(path, f"initial_image.hv")
    try:
        pet_data["initial_image"] = ImageData(initial_img_path).maximum(0)
    except Exception as e_initial:
        logging.warning("No PET initial image found (%s). Using uniform copy of template image.",
                        str(e_initial))
        pet_data["initial_image"] = pet_data["template_image"].get_uniform_copy(1)

    try:
        pet_data["spect"] = ImageData(os.path.join(path, "spect.hv"))
    except Exception as e_spect:
        logging.info("No SPECT guidance image found for PET: %s", str(e_spect))

    return pet_data


def get_pet_data_multiple_bed_pos(path: str, suffixes: List[str], tof=False) -> Dict[str, object]:
    """
    Load PET data for multiple bed positions from the given path.
    
    For each suffix in the list, a dictionary is created with keys:
    "acquisition_data", "additive", "normalisation", "attenuation",
    and "spect". The "initial_image" is created per bed position using the same
    fallback strategy (try initial image, then template image, then a uniform image).
    
    The outer dictionary contains:
        "bed_positions": A dictionary mapping each suffix to its data dictionary.
        "initial_image": A combined initial image (obtained by summing the individual
                         initial images) over all bed positions.
    
    Args:
        path (str): Path to the data directory.
        suffixes (List[str]): List of suffixes for different bed positions.
    
    Returns:
        dict: A dictionary with keys "bed_positions" and "initial_image".
    """
    
    tof_str = "tof" if tof else "non_tof"    
    
    pet_data = {}
    
    ### Shared data across all bed positions ###
    pet_data["attenuation"] = ImageData(os.path.join(path, f"umap_zoomed.hv"))

    # Always load the template image.
    template_img_path = os.path.join(path, f"template_image.hv")
    try:
        pet_data["template_image"] = ImageData(template_img_path)
    except Exception as e_template:
        logging.error("Failed to load PET template image (%s)", str(e_template))
        raise RuntimeError("Unable to load PET template image.") from e_template

    # Try to load the initial image.
    initial_img_path = os.path.join(path, tof_str, f"initial_image.hv")
    try:
        pet_data["initial_image"] = ImageData(initial_img_path).maximum(0)
    except Exception as e_initial:
        logging.warning("No PET initial image found (%s). Using uniform copy of template image.",
                        str(e_initial))
        pet_data["initial_image"] = pet_data["template_image"].get_uniform_copy(1)

    try:
        pet_data["spect"] = ImageData(os.path.join(path, "spect.hv"))
    except Exception as e_spect:
        logging.info("No SPECT guidance image found for PET: %s", str(e_spect))
        
        
    ### Data for each bed position ###
    pet_data["bed_positions"] = {}
    for suffix in suffixes:
        bed_data = {}
        bed_data["acquisition_data"] = AcquisitionData(
            os.path.join(path, tof_str, f"prompts{suffix}.hs")
        )
        bed_data["additive"] = AcquisitionData(
            os.path.join(path, tof_str, f"additive_term{suffix}.hs")
        )
        bed_data["normalisation"] = AcquisitionData(
            os.path.join(path, tof_str, f"mult_factors{suffix}.hs")
        )
        bed_data["template_image"] = ImageData(
            os.path.join(path, f"template_image{suffix}.hv")
        )
        bed_data["initial_image"] = ImageData(
            os.path.join(path, tof_str, f"initial_image{suffix}.hv")
        ).maximum(0)
        bed_data["attenuation"] = ImageData(
            os.path.join(path, f"umap{suffix}.hv")
        )

        pet_data["bed_positions"][suffix] = bed_data
        
    return pet_data
        
def get_spect_data(path: str) -> dict:
    """
    Load SPECT data from the given path.
    
    This function always loads a template image and then attempts to load the
    initial image. If the initial image is not found, it creates a uniform copy
    of the template image (filled with ones). Also, the attenuation image is flipped
    on the x-axis due to a known STIR bug.

    Args:
        path (str): Path to the data directory.

    Returns:
        dict: A dictionary with keys: "acquisition_data", "additive", "attenuation",
        "template_image", and "initial_image".
    """
    spect_data = {}
    spect_data["acquisition_data"] = AcquisitionData(os.path.join(path, "peak.hs"))

    try:
        spect_data["additive"] = AcquisitionData(os.path.join(path, "scatter.hs"))
    except Exception as e_scatter:
        logging.warning("No scatter data found (%s). Using zeros.", str(e_scatter))
        spect_data["additive"] = AcquisitionData(spect_data["acquisition_data"])
        spect_data["additive"].fill(0)

    spect_data["attenuation"] = ImageData(os.path.join(path, "umap_zoomed.hv"))
    # Flip the attenuation image on the x-axis due to bug in STIR.
    attn_arr = spect_data["attenuation"].as_array()
    attn_arr = np.flip(attn_arr, axis=-1)
    spect_data["attenuation"].fill(attn_arr)

    # Always load the template image.
    template_img_path = os.path.join(path, "template_image.hv")
    try:
        spect_data["template_image"] = ImageData(template_img_path)
    except Exception as e_template:
        logging.error("Failed to load SPECT template image (%s)", str(e_template))
        raise RuntimeError("Unable to load SPECT template image.") from e_template

    # Try to load the initial image.
    initial_img_path = os.path.join(path, "initial_image.hv")
    try:
        spect_data["initial_image"] = ImageData(initial_img_path).maximum(0)
    except Exception as e_initial:
        logging.warning("No SPECT initial image found (%s). Using uniform copy of template image.",
                        str(e_initial))
        spect_data["initial_image"] = spect_data["template_image"].get_uniform_copy(1)

    return spect_data