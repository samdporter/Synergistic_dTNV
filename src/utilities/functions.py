from sirf.STIR import (AcquisitionData, ImageData,
                       SPECTUBMatrix, AcquisitionModelUsingMatrix,
                          AcquisitionModelUsingParallelproj,
                            AcquisitionModelUsingRayTracingMatrix,
                                SeparableGaussianImageFilter,
                                    TruncateToCylinderProcessor)
import numpy as np

def get_pet_am(gpu=True, gauss_fwhm=None, truncate_cylinder=True):
    if gpu:
        pet_am = AcquisitionModelUsingParallelproj()
    else:
        pet_am = AcquisitionModelUsingRayTracingMatrix()
        pet_am.set_num_tangential_LORs(10)
    if gauss_fwhm:
        pet_psf = SeparableGaussianImageFilter()
        pet_psf.set_fwhms(gauss_fwhm) 
        pet_am.set_image_data_processor(pet_psf)
    if truncate_cylinder:
        cyl = TruncateToCylinderProcessor()
        cyl.set_strictly_less_than_radius(True)
        pet_am.set_image_data_processor(cyl)
    return pet_am

def get_spect_am(spect_data, res = None, keep_all_views_in_cache=True, gauss_fwhm=None):
    spect_am_mat = SPECTUBMatrix()
    try:
        spect_am_mat.set_attenuation_image(spect_data["attenuation"])
    except:
        print("No attenuation data")
    spect_am_mat.set_keep_all_views_in_cache(keep_all_views_in_cache)
    if res:
        spect_am_mat.set_resolution_model(*res)
    spect_am = AcquisitionModelUsingMatrix(spect_am_mat)
    if gauss_fwhm:
        spect_psf = SeparableGaussianImageFilter()
        spect_psf.set_fwhms(gauss_fwhm) 
        spect_am.set_image_data_processor(spect_psf)
    return spect_am