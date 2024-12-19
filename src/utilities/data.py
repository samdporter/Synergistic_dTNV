from sirf.STIR import (AcquisitionData, ImageData)
import os
import numpy as np

def get_pet_data(path):

    pet_data = {}
    pet_data["acquisition_data"] = AcquisitionData(os.path.join(path,  "PET/prompts.hs"))
    pet_data["additive"] = AcquisitionData(os.path.join(path,  "PET/additive.hs"))
    pet_data["normalisation"] = AcquisitionData(os.path.join(path,  "PET/mult_factors.hs"))
    pet_data["initial_image"] = ImageData(os.path.join(path,  "PET/initial_image.hv")).maximum(0)

    return pet_data

def get_spect_data(path):

    spect_data = {}
    spect_data["acquisition_data"] = AcquisitionData(os.path.join(path,  "SPECT/peak.hs"))
    spect_data["additive"] = AcquisitionData(os.path.join(path,  "SPECT/scatter.hs"))
    spect_data["attenuation"] = ImageData(os.path.join(path,  "SPECT/umap.hv"))
    # Need to flip the attenuation image on the x-axis due to bug in STIR
    attn_arr = spect_data["attenuation"].as_array()
    attn_arr = np.flip(attn_arr, axis=-1)
    spect_data["attenuation"].fill(attn_arr)
    spect_data["initial_image"] = ImageData(os.path.join(path,  "SPECT/initial_image.hv")).maximum(0)

    return spect_data