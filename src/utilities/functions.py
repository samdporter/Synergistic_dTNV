import numpy as np
from types import MethodType
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

def compute_kappa_squared_image_from_partitioned_objective(obj_funs, initial_image, normalise=True):
    """
    Computes a "kappa" image for a prior as sqrt(H.1).
    This will attempt to give uniform "perturbation response".
    See Yu-jung Tsai et al. TMI 2020 https://doi.org/10.1109/TMI.2019.2913889

    WARNING: Assumes the objective function has been set-up already.
    """
    out = initial_image.get_uniform_copy(0)
    for obj_fun in obj_funs:
        # need to get the function from the ScaledFunction OperatorCompositionFunction
        out += obj_fun.function.multiply_with_Hessian(initial_image, initial_image.allocate(1))
    out = out.abs()
    # shouldn't really need to do this, but just in case
    out = out.maximum(0)
    # debug printing
    print(f"max: {out.max()}")
    mean = out.sum()/out.size
    print(f"mean: {mean}")
    if normalise:
        # we want to normalise by thye median
        median = out.as_array().flatten()
        median.sort()
        median = median[int(median.size/2)]
        print(f"median: {median}")
        out /= median
    return out

def attach_prior_hessian(prior, epsilon = 0) -> None:
    """Attach an inv_hessian_diag method to the prior function."""

    def inv_hessian_diag(self, x, out=None, epsilon=epsilon):
        ret = self.function.operator.adjoint(
            self.function.function.inv_hessian_diag(
                self.function.operator.direct(x),
            )
        )
        if out is not None:
            out.fill(ret)
        return ret

    prior.inv_hessian_diag = MethodType(inv_hessian_diag, prior)