import sys
import os
import numpy as np
import argparse
from types import MethodType
import pandas as pd
from numbers import Number
import shutil

# SIRF imports
from sirf.STIR import (ImageData, AcquisitionData, 
                       AcquisitionModelUsingMatrix, 
                       AcquisitionModelUsingParallelproj, 
                       AcquisitionSensitivityModel, 
                       SPECTUBMatrix,MessageRedirector,
                       TruncateToCylinderProcessor, SeparableGaussianImageFilter,
                       AcquisitionModelUsingRayTracingMatrix,
                       make_Poisson_loglikelihood,
                       )
from sirf.Reg import AffineTransformation
#AcquisitionData.set_storage_scheme('file')

# CIL imports
from cil.framework import BlockDataContainer, DataContainer
from cil.optimisation.operators import BlockOperator, ZeroOperator

parser = argparse.ArgumentParser(description='BSREM')

parser.add_argument('--alpha', type=float, default=128, help='alpha')
parser.add_argument('--beta', type=float, default=0.05, help='beta')
parser.add_argument('--delta', type=float, default=1e-6, help='delta')
# num_subsets can be an integer or a string of two integers separated by a comma
parser.add_argument('--num_subsets', type=str, default="12", help='number of subsets')
parser.add_argument('--use_kappa', action='store_true', help='use kappa')
parser.add_argument('--initial_step_size', type=float, default=2, help='initial step size')

parser.add_argument('--iterations', type=int, default=60, help='max iterations')
parser.add_argument('--update_interval', type=int, default=12, help='update interval')
parser.add_argument('--relaxation_eta', type=float, default=0.1, help='relaxation eta')

parser.add_argument('--data_path', type=str, default="/home/storage/copied_data/data/phantom_data/for_cluster", help='data path')
parser.add_argument('--output_path', type=str, default="/home/sam/working/BSREM_PSMR_MIC_2024/results/test", help='output path')
parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/src', help='source path')
parser.add_argument('--working_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/tmp', help='working path')
parser.add_argument('--save_images', type=bool, default=True, help='save images')

# set numpy seed - None if not set
parser.add_argument('--seed', type=int, default=None, help='numpy seed')

parser.add_argument('--stochastic', action='store_true', help='Enables stochastic processing')
parser.add_argument('--svrg', action='store_true', help='Enables SVRG')
parser.add_argument('--saga', action='store_true', help='Enables SAGA')
parser.add_argument('--with_replacement', action='store_true', help='Enables replacement')
parser.add_argument('--single_modality_update', action='store_true', help='Enables single modality update')
parser.add_argument('--prior_is_subset', action='store_true', help='Sets prior as subset')
parser.add_argument('--gpu', action='store_false', default=True, help='Disables GPU')
parser.add_argument('--keep_all_views_in_cache', action='store_false', default=True, help='Do not keep all views in cache')


args = parser.parse_args()

# Imports from my stuff and SIRF contribs
sys.path.insert(0, args.source_path)
from BSREM.BSREM import BSREMmm_of
from BSREM.preconditioners import (ConditionalPreconditioner, FixedPreconditioner, 
                                   HarmonicMeanPreconditionerBSREMPrior, HessianDiagPreconditionerBSREMPrior)
from structural_priors.tmp_classes import (OperatorCompositionFunction,
                                                   ZoomOperator, CompositionOperator,
                                                    NiftyResampleOperator,
                                                    FairL21Norm,
                                                        )

from structural_priors.Operator import Operator, NumpyDataContainer, NumpyBlockDataContainer
from structural_priors.Function import Function, SIRFBlockFunction
Operator.is_linear = lambda self: True
from structural_priors.Gradients import DirectionalGradient
from structural_priors.VTV import create_vectorial_total_variation
from structural_priors.Gradients import Jacobian

# Monkey patching
BlockOperator.forward = lambda self, x: self.direct(x)
BlockOperator.backward = lambda self, x: self.adjoint(x)

BlockDataContainer.get_uniform_copy = lambda self, n: BlockDataContainer(*[x.clone().fill(n) for x in self.containers])
BlockDataContainer.max = lambda self: max(d.max() for d in self.containers)

ZeroOperator.backward = lambda self, x: self.adjoint(x)
ZeroOperator.forward = lambda self, x: self.direct(x)

def get_filters():
    cyl, gauss = TruncateToCylinderProcessor(), SeparableGaussianImageFilter()
    cyl.set_strictly_less_than_radius(True)
    gauss.set_fwhms((7,7,7))
    return cyl, gauss

def get_pet_data(path):

    pet_data = {}
    pet_data["acquisition_data"] = AcquisitionData(os.path.join(path,  "PET/projdata_bed0.hs"))
    pet_data["additive"] = AcquisitionData(os.path.join(path,  "PET/additive3d_bed0_nonan.hs"))
    pet_data["normalisation"] = AcquisitionData(os.path.join(path,  "PET/inv_normacfprojdata_bed0.hs"))
    pet_data["initial_image"] = ImageData(os.path.join(path,  "PET/pet_osem_20.hv")).maximum(0)

    return pet_data

def get_spect_data(path):

    spect_data = {}
    spect_data["acquisition_data"] = AcquisitionData(os.path.join(path,  "SPECT/peak_1_projdata__f1g1d0b0.hs"))
    spect_data["additive"] = AcquisitionData(os.path.join(path,  "SPECT/simind_scatter_ellipses_megp_cpd.hs"))
    spect_data["attenuation"] = ImageData(os.path.join(path,  "SPECT/umap_zoomed.hv"))
    # Need to flip the attenuation image on the x-axis due to bug in STIR
    attn_arr = spect_data["attenuation"].as_array()
    attn_arr = np.flip(attn_arr, axis=-1)
    spect_data["attenuation"].fill(attn_arr)
    spect_data["initial_image"] = ImageData(os.path.join(path,  "SPECT/spect_osem_20.hv")).maximum(0)

    return spect_data

def get_zoom_transform(data_path, filename, zoom_operator, template_image):
    transform =  AffineTransformation(os.path.join(data_path, "Registration", filename))
    resampler = NiftyResampleOperator(template_image, template_image, transform)
    return CompositionOperator([zoom_operator, resampler])

def get_pet_am(pet_data, gpu):
    if gpu:
        pet_am = AcquisitionModelUsingParallelproj()
    else:
        pet_am = AcquisitionModelUsingRayTracingMatrix()
        pet_am.set_num_tangential_LORs(10)
    asm = AcquisitionSensitivityModel(pet_data["normalisation"])
    pet_am.set_acquisition_sensitivity(asm)
    pet_am.set_additive_term(pet_data["additive"])
    # using adjoint(forard(image)) & STIR find_fwhm_in_image
    # 1cm FWHM from NEMA 2001 (Mediso AnyScan specificaitons)
    # operations applied one after the other to find total FWHM
    pet_psf = SeparableGaussianImageFilter()
    pet_psf.set_fwhms((4.5,7.5,7.5)) 
    pet_am.set_image_data_processor(pet_psf)
    #pet_am.set_up(pet_data["acquisition_data"], pet_data["initial_image"])
    return pet_am

def get_spect_am(spect_data, keep_all_views_in_cache=False):
    spect_am_mat = SPECTUBMatrix()
    spect_am_mat.set_attenuation_image(spect_data["attenuation"])
    spect_am_mat.set_keep_all_views_in_cache(keep_all_views_in_cache)
    # using Tc99m (140 keV) AnyScan measured resolution modelling
    # close enough to 150 keV Y90
    # spect_am_mat.set_resolution_model(1.81534, 0.02148, False) 
    spect_am_mat.set_resolution_model(0.9323, 0.03, False) 
    # using gaussians estimated from Y90 data (see notebook)
    # /home/sam/working/simulated_data/data/data.ipynb
    spect_am = AcquisitionModelUsingMatrix(spect_am_mat)
    spect_psf = SeparableGaussianImageFilter()
    #spect_psf.set_fwhms((22, 19, 19))
    #spect_am.set_image_data_processor(spect_psf)
    spect_am.set_additive_term(spect_data["additive"]) #TODO: change back
    #spect_am.set_up(spect_data["acquisition_data"], spect_data["initial_image"])
    return spect_am

def get_objective_function(data, acq_model, initial_image, num_subsets):
    
    obj_fun = make_Poisson_loglikelihood(data)
    obj_fun.set_acquisition_model(acq_model)
    obj_fun.set_num_subsets(num_subsets)
    obj_fun.set_up(initial_image)
    
    return obj_fun

def get_vectorial_tv(bo, ct, alpha, beta, initial_estimates, delta, gpu=False, kappa=False):
    if kappa:
        kappas = [np.reciprocal(k.as_array()) for weight, k in kappa.containers]
    else:
        kappas = None
    weights = [alpha, beta]

    vtv = create_vectorial_total_variation(smoothing_function='fair', eps=delta, gpu=gpu)
    jac = NumpyBlockDataContainer(bo.direct(initial_estimates), Jacobian(anatomical=ct.as_array(), voxel_sizes=ct.voxel_sizes(), 
                                                                        gpu=gpu, weights=weights, kappas=None))
    jac_co = CompositionOperator([bo, jac])
    jac_co.range_geometry = lambda: initial_estimates
    return OperatorCompositionFunction(vtv, jac_co)

def compute_kappa_squared_image(obj_fun, initial_image):
    '''
    Computes a "kappa" image for a prior as sqrt(H.1). This will attempt to give uniform "perturbation response".
    See Yu-jung Tsai et al. TMI 2020 https://doi.org/10.1109/TMI.2019.2913889

    WARNING: Assumes the objective function has been set-up already
    '''
    # This needs SIRF 3.7. If you don't have that yet, you should probably upgrade anyway!
    Hessian_row_sum = obj_fun.multiply_with_Hessian(initial_image,  initial_image.allocate(1))
    return (-1*Hessian_row_sum)
    
# Change to working directory - this is where the tmp_ files will be saved
os.chdir(args.working_path)

def main(args):

    # if single_modality_update is False, num_subsets must be integer
    if not args.single_modality_update:
        try:
            args.num_subsets = int(args.num_subsets)
        except:
            raise ValueError("num_subsets must be an integer if single_modality_update is False")
    if isinstance(args.num_subsets, Number):
        pet_num_subsets = int(args.num_subsets)
        spect_num_subsets = int(args.num_subsets)
    elif isinstance(args.num_subsets, str):
        num_subsets = args.num_subsets
        subset_list = num_subsets.split(",")
        # if list is lenght one, set both to the same value
        if len(subset_list) == 1:
            pet_num_subsets = int(subset_list[0])
            spect_num_subsets = int(subset_list[0])
        else:
            pet_num_subsets = int(subset_list[0])
            spect_num_subsets = int(subset_list[1])

    cyl, gauss, = get_filters()
    ct = ImageData(os.path.join(args.data_path, "CT/ct_zoomed_smallFOV.hv"))
    # normalise the CT image
    ct+=(-ct).max()
    ct/=ct.max()    

    pet_data  = get_pet_data(args.data_path)
    cyl.apply(pet_data["initial_image"])
    #gauss.apply(pet_data["initial_image"])
    pet_data["initial_image"].write("initial_image_0.hv")

    pet2ct = NiftyResampleOperator(ct, pet_data["initial_image"], AffineTransformation(os.path.join(args.data_path, "Registration", "pet_to_ct_smallFOV.txt")))
    pet_am = get_pet_am(pet_data,  gpu=True)
    pet_am.direct = lambda x: pet_am.forward(x)
    pet_am.adjoint = lambda x: pet_am.backward(x)
    
    pet_obj_fun = get_objective_function(pet_data["acquisition_data"], pet_am, pet_data["initial_image"], pet_num_subsets)

    spect_data  = get_spect_data(args.data_path)
    cyl.apply(spect_data["initial_image"])
    #gauss.apply(spect_data["initial_image"])
    spect_data["initial_image"].write("initial_image_1.hv")

    spect2ct = NiftyResampleOperator(ct, spect_data["initial_image"], AffineTransformation(os.path.join(args.data_path, "Registration", "spect_to_ct_smallFOV.txt")))
    spect_am = get_spect_am(spect_data, args.keep_all_views_in_cache)
    spect_am.direct = lambda x: spect_am.forward(x)
    spect_am.adjoint = lambda x: spect_am.backward(x)
    
    spect_obj_fun = get_objective_function(spect_data["acquisition_data"], spect_am, spect_data["initial_image"], spect_num_subsets)
    
    zero_pet2ct = ZeroOperator(pet_data["initial_image"], ct)
    zero_spect2ct = ZeroOperator(spect_data["initial_image"], ct)

    bo = BlockOperator(pet2ct, zero_spect2ct,
                        zero_pet2ct, spect2ct, 
                        shape = (2,2))

    initial_estimates = BlockDataContainer(pet_data["initial_image"], spect_data["initial_image"])

    if args.use_kappa:
        kappa = bo.direct(BlockDataContainer(compute_kappa_squared_image(pet_obj_fun, pet_data["initial_image"]),compute_kappa_squared_image(spect_obj_fun, spect_data["initial_image"])))
        for i, el in enumerate(kappa.containers):
            gauss.apply(el)
            el.write(f"kappa_{i}.hv")
    else:
        kappa = False

    prior = get_vectorial_tv(bo, ct, args.alpha, args.beta, initial_estimates, delta=args.delta, gpu=args.gpu, kappa=kappa)

    pet2spect_zero = ZeroOperator(pet_data["initial_image"], spect_data["acquisition_data"])
    spect2pet_zero = ZeroOperator(spect_data["initial_image"], pet_data["acquisition_data"])

    acquisition_model = BlockOperator(pet_am, spect2pet_zero,
                                        pet2spect_zero, spect_am,
                                    shape=(2,2)) 
    data = BlockDataContainer(pet_data["acquisition_data"], spect_data["acquisition_data"])

    initial = initial_estimates
        
    acquisition_model.is_linear = MethodType(lambda self: True, acquisition_model)
        
    bsrem=BSREMmm_of(SIRFBlockFunction([pet_obj_fun, spect_obj_fun]), prior, 
                        initial=initial, initial_step_size=args.initial_step_size, relaxation_eta=args.relaxation_eta, 
                        update_objective_interval=args.update_interval, save_path=args.working_path,
                        stochastic=args.stochastic, svrg=args.svrg, saga=args.saga, with_replacement=args.with_replacement,
                        single_modality_update=args.single_modality_update, save_images = args.save_images,
                        prior_is_subset=args.prior_is_subset, update_max=100*initial.max())

    bsrem.max_iteration=args.iterations
    bsrem.run(args.iterations, verbose=2)

    return bsrem

if __name__ == "__main__":
    
    # Redirect messages if needed
    _ = MessageRedirector()

    # Create a dataframe for all arguments and save as CSV
    df_args = pd.DataFrame([vars(args)])
    df_args.to_csv(os.path.join(args.output_path, "args.csv"))

    # Print all arguments
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Run main function and retrieve result
    bsrem = main(args)

    # Ensure output path exists
    os.makedirs(args.output_path, exist_ok=True)

    # Save reconstructed images based on type
    if isinstance(bsrem.x, ImageData):
        bsrem.x.write(os.path.join(args.output_path, f"bsrem_a_{args.alpha}_b_{args.beta}.hv"))

    elif isinstance(bsrem.x, BlockDataContainer):
        for i, el in enumerate(bsrem.x.containers):
            el.write(os.path.join(args.output_path, f"bsrem_modality_{i}_a_{args.alpha}_b_{args.beta}.hv"))

    # Save loss data
    df_objective = pd.DataFrame([l[0] for l in bsrem.loss])
    df_objective.to_csv(os.path.join(args.output_path, f"bsrem_objective_a_{args.alpha}_b_{args.beta}.csv"))

    df_data = pd.DataFrame([l[1] for l in bsrem.loss])
    df_prior = pd.DataFrame([l[2] for l in bsrem.loss])

    df_data.to_csv(os.path.join(args.output_path, f"bsrem_data_a_{args.alpha}_b_{args.beta}.csv"))
    df_prior.to_csv(os.path.join(args.output_path, f"bsrem_prior_a_{args.alpha}_b_{args.beta}.csv"))

    # Combine loss data into a single CSV
    df_full = pd.concat([df_objective, df_data, df_prior], axis=1)
    df_full.columns = ["Objective", "Data", "Prior"]
    df_full.to_csv(os.path.join(args.output_path, f"bsrem_full_a_{args.alpha}_b_{args.beta}.csv"))

    # Remove temporary files
    for file in os.listdir(args.working_path):
        if file.startswith("tmp_") and (file.endswith(".s") or file.endswith(".hs")):
            os.remove(os.path.join(args.working_path, file))

    # Move leftover files (if any) to the output path
    for file in os.listdir(args.working_path):
        if file.endswith((".hv", ".v", ".ahv")):
            print(f"Moving to {os.path.join(args.output_path, file)}")
            shutil.move(os.path.join(args.working_path, file), os.path.join(args.output_path, file))

    print("Done")


