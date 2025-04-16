import numpy as np
from sirf.STIR import (
    SPECTUBMatrix, AcquisitionModelUsingMatrix,
    AcquisitionModelUsingParallelproj,
    AcquisitionModelUsingRayTracingMatrix,
    SeparableGaussianImageFilter,
    TruncateToCylinderProcessor,
)

def get_pet_am(
        gpu=True, gauss_fwhm=None, 
    ):
    if gpu:
        pet_am = AcquisitionModelUsingParallelproj()
    else:
        pet_am = AcquisitionModelUsingRayTracingMatrix()
        pet_am.set_num_tangential_LORs(10)
    
    if gauss_fwhm:
        pet_psf = SeparableGaussianImageFilter()
        pet_psf.set_fwhms(gauss_fwhm)
        pet_am.set_image_data_processor(pet_psf)
    
    return pet_am


def get_spect_am(
        spect_data, res = None, 
        keep_all_views_in_cache=True, 
        gauss_fwhm=None,
        attenuation=True
    ):
    spect_am_mat = SPECTUBMatrix()
    spect_am_mat.set_keep_all_views_in_cache(
        keep_all_views_in_cache
    )
    if attenuation:
        try:
            spect_am_mat.set_attenuation_image(
                spect_data["attenuation"]
            )
        except:
            print("No attenuation data")
    if res:
        spect_am_mat.set_resolution_model(*res)
    spect_am = AcquisitionModelUsingMatrix(spect_am_mat)
    if gauss_fwhm:
        spect_psf = SeparableGaussianImageFilter()
        spect_psf.set_fwhms(gauss_fwhm) 
        spect_am.set_image_data_processor(spect_psf)
    return spect_am