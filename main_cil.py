import sys
import os
import numpy as np
import argparse
from types import MethodType
import pandas as pd
from numbers import Number
import shutil

import cProfile
import pstats

# SIRF imports
from sirf.STIR import (ImageData, AcquisitionData,MessageRedirector,
                       TruncateToCylinderProcessor, SeparableGaussianImageFilter,
                       )

from sirf.Reg import AffineTransformation
from sirf.contrib.partitioner import partitioner
AcquisitionData.set_storage_scheme('memory')

# CIL imports
from cil.framework import BlockDataContainer
from cil.optimisation.operators import BlockOperator, ZeroOperator, IdentityOperator
from cil.optimisation.functions import SVRGFunction, SAGAFunction, OperatorCompositionFunction
from cil.optimisation.algorithms import ISTA
from cil.optimisation.utilities import Sampler, StepSizeRule

parser = argparse.ArgumentParser(description='BSREM')

parser.add_argument('--alpha', type=float, default=128, help='alpha')
parser.add_argument('--beta', type=float, default=0.05, help='beta')
parser.add_argument('--delta', type=float, default=1e-6, help='delta')
parser.add_argument('--num_subsets', type=str, default="9,6", help='number of subsets')
parser.add_argument('--prior_probability', type=float, default=0.5, help='prior probability')
parser.add_argument('--use_kappa', action='store_true', help='use kappa')
parser.add_argument('--initial_step_size', type=float, default=0.5, help='initial step size')

parser.add_argument('--iterations', type=int, default=250, help='max iterations')
parser.add_argument('--update_interval', type=int, default=None, help='update interval')
parser.add_argument('--relaxation_eta', type=float, default=0.1, help='relaxation eta')

#/home/storage/copied_data/data/phantom_data/for_cluster, /home/sam/working/OSEM/simple_data
parser.add_argument('--data_path', type=str, default="/home/sam/working/OSEM/simple_data", help='data path')
parser.add_argument('--output_path', type=str, default="/home/sam/working/BSREM_PSMR_MIC_2024/results/test", help='output path')
parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/src', help='source path')
parser.add_argument('--working_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/tmp', help='working path')
parser.add_argument('--save_images', type=bool, default=True, help='save images')
parser.add_argument('--speedy', action='store_true', help='Speedy mode - no corrections for SPECT')

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
from structural_priors.VTV import WeightedVectorialTotalVariation
from utilities.data import get_pet_data, get_spect_data
from utilities.functions import get_pet_am, get_spect_am
from utilities.preconditioners import *
from utilities.callbacks import *
from utilities.nifty import NiftyResampleOperator

# Monkey patching
BlockOperator.forward = lambda self, x: self.direct(x)
BlockOperator.backward = lambda self, x: self.adjoint(x)

BlockDataContainer.get_uniform_copy = lambda self, n: BlockDataContainer(*[x.clone().fill(n) for x in self.containers])
BlockDataContainer.max = lambda self: max(d.max() for d in self.containers)

ZeroOperator.backward = lambda self, x: self.adjoint(x)
ZeroOperator.forward = lambda self, x: self.direct(x)

compare_preconds = True
check_gradients = False

def get_filters():
    cyl, gauss = TruncateToCylinderProcessor(), SeparableGaussianImageFilter()
    cyl.set_strictly_less_than_radius(True)
    gauss.set_fwhms((7,7,7))
    return cyl, gauss

def compute_kappa_squared_image(obj_fun, initial_image):
    Hessian_row_sum = obj_fun.multiply_with_Hessian(initial_image,  initial_image.allocate(1))
    return (-1*Hessian_row_sum)

def get_s_inv(obj_funs, initial_estimates, compare_preconds=False):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimates.get_uniform_copy(0)
    for i, el in enumerate(s_inv.containers):
        for obj_fun in obj_funs[i]:
            tmp = obj_fun.function.get_subset_sensitivity(0)
            tmp.maximum(0, out = tmp)
            tmp_arr = tmp.as_array()
            tmp.fill(np.reciprocal(tmp_arr, where=tmp_arr!=0))
            el += tmp
    if compare_preconds:
        for i, el in enumerate(s_inv.containers):
            el.write(f"inv_subset_sensitivity_{i}.hv")
    return s_inv

def compute_hessian_diagonals(data_list, obj_funs_list, output_filenames):

    for data, obj_funs, output_file in zip(data_list, obj_funs_list, output_filenames):
        # Initialize uniform copies
        ones_image = data["initial_image"].get_uniform_copy(1)
        hessian_diag = ones_image.get_uniform_copy(0)

        # Accumulate Hessian contributions
        for obj_fun in obj_funs:
            hessian_diag += obj_fun.function.multiply_with_Hessian(
                data["initial_image"], ones_image
            )

        # Take absolute values and write the result
        hessian_diag = hessian_diag.abs()
        hessian_diag.write(output_file)

# Change to working directory - this is where the tmp_ files will be saved
os.chdir(args.working_path)

class BlockIndicatorBox():
    
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        
    def __call__(self, x):
        # because we're using this as a projection, this should always return 0
        return 0.0

    def proximal(self, x, tau, out=None):
        res = x.copy()
        for el in res.containers:
            el.maximum(self.lower, out=el)
            el.minimum(self.upper, out=el)
        if out is not None:
            out.fill(res)
        return res
    
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
       
def main(args):
    """
    Main function to perform image reconstruction using BSREM and PSMR algorithms.
    """
    
    # convert num_subsets to list of ints
    num_subsets = [int(i) for i in args.num_subsets.split(",")]

    cyl, gauss, = get_filters()
    ct = ImageData(os.path.join(args.data_path, "CT/ct_zoomed_smallFOV.hv"))
    # normalise the CT image
    ct+=(-ct).max()
    ct/=ct.max()    

    pet_data  = get_pet_data(args.data_path)
    spect_data  = get_spect_data(args.data_path)
    
    # apply filters to initial images
    #cyl.apply(spect_data["initial_image"])
    gauss.apply(spect_data["initial_image"])
    #cyl.apply(pet_data["initial_image"])
    gauss.apply(pet_data["initial_image"])

    initial_estimates = BlockDataContainer(pet_data["initial_image"], spect_data["initial_image"])
    
    # set up resampling operators
    pet_data["initial_image"].write("initial_image_0.hv")
    pet2ct = NiftyResampleOperator(ct, pet_data["initial_image"], AffineTransformation(os.path.join(args.data_path, "Registration", "pet_to_ct_smallFOV.txt")))
    zero_pet2ct = ZeroOperator(pet_data["initial_image"], ct)
    
    spect_data["initial_image"].write("initial_image_1.hv")
    spect2ct = NiftyResampleOperator(ct, spect_data["initial_image"], AffineTransformation(os.path.join(args.data_path, "Registration", "spect_to_ct_smallFOV.txt")))
    zero_spect2ct = ZeroOperator(spect_data["initial_image"], ct)

    # set up prior
    bo = BlockOperator(pet2ct, zero_spect2ct,
                        zero_pet2ct, spect2ct, 
                        shape = (2,2))
    vtv = WeightedVectorialTotalVariation(bo.direct(initial_estimates), [args.alpha, args.beta], args.delta, anatomical=ct)
    prior = OperatorCompositionFunction(vtv, bo)

    # set up data fidelity functions
    
    pet_ds, _, pet_obj_funs = partitioner.data_partition(pet_data['acquisition_data'], pet_data['additive'], pet_data['normalisation'], 
                                                        num_batches=num_subsets[0], mode = "staggered",
                                                        create_acq_model=lambda: get_pet_am(args.gpu, gauss_fwhm=(4.5,7.5,7.5), truncate_cylinder=True))
    for obj_fun in pet_obj_funs:
        obj_fun.set_up(initial_estimates[0])
        
    spect_ds, _, spect_obj_funs = partitioner.data_partition(spect_data['acquisition_data'], spect_data['additive'], spect_data['acquisition_data'].get_uniform_copy(1), 
                                                        num_batches=num_subsets[1], mode = "staggered",
                                                        create_acq_model=lambda: get_spect_am(spect_data, (0.9323, 0.03, False), keep_all_views_in_cache=args.keep_all_views_in_cache))
    for obj_fun in spect_obj_funs:
        obj_fun.set_up(initial_estimates[1])
    
    assert sum([d.shape[2] for d in pet_ds]) == pet_data['acquisition_data'].shape[2], "PET data subsets do not sum to total number of views"
    assert sum([d.shape[2] for d in spect_ds]) == spect_data['acquisition_data'].shape[2], "SPECT data subsets do not sum to total number of views"

    spect2pet_zero = ZeroOperator(spect_data["initial_image"], pet_data["initial_image"])
    pet2pet_id = IdentityOperator(pet_data["initial_image"])
    pet2spect_zero = ZeroOperator(pet_data["initial_image"], spect_data["initial_image"])
    spect2spect_id = IdentityOperator(spect_data["initial_image"])
    
    pet_obj_funs = [OperatorCompositionFunction(obj_fun, BlockOperator(pet2pet_id, spect2pet_zero, shape = (1,2))) for obj_fun in pet_obj_funs]
    spect_obj_funs = [OperatorCompositionFunction(obj_fun, BlockOperator(pet2spect_zero, spect2spect_id, shape = (1,2))) for obj_fun in spect_obj_funs]
    
    all_funs = pet_obj_funs + spect_obj_funs

    if compare_preconds:
        compute_hessian_diagonals([pet_data, spect_data], [pet_obj_funs, spect_obj_funs], ["pet_hessian_diag.hv", "spect_hessian_diag.hv"])

    # now the prior needs to be scaled according to the probability of the prior
    # A probability of 0.5 means the prior is used every other iteration
    # therefore the prior needs to be scaled by 1/(2*len(pet_obj_funs + spect_obj_funs))
    prior = -args.prior_probability/(len(pet_obj_funs + spect_obj_funs)) * prior

    # This is a scaled function but if we multiply / divide by a scalar, it will affect the preconditioner
    # Effectively scaling twice so we won't scale the hessian diag 
    # Possibly there is a better way to do this. This is pretty horrible
    def hessian_diag(self, x, out=None):
        ret = self.function.operator.adjoint(self.function.function.hessian_diag(self.function.operator.direct(x)))
        if out is not None:
            out.fill(ret)
        return ret
    
    def inv_hessian_diag(self, x, out=None):
        ret = self.function.operator.adjoint(self.function.function.inv_hessian_diag(self.function.operator.direct(x)))
        if out is not None:
            out.fill(ret)
        return ret
    
    prior.hessian_diag = MethodType(hessian_diag, prior)
    prior.inv_hessian_diag = MethodType(inv_hessian_diag, prior)

    all_funs.append(prior)
    
    if compare_preconds:
        # test if working by finding diagonal of Hessian
        inv_hess_diag = prior.inv_hessian_diag(initial_estimates)
        for i, el in enumerate(inv_hess_diag.containers):
            el.abs().write(f"prior_inv_hessian_diag_{i}.hv")
        print("Inverse Hessian diagonal computed")
    
    cyl.apply(initial_estimates[0])
    
    if check_gradients:
        # write gradients to disk
        gradient_origin = ["pet"]*len(pet_obj_funs) + ["spect"]*len(spect_obj_funs) + ["prior"]
        for i, obj_fun in enumerate(all_funs):
            grad = obj_fun.gradient(initial_estimates)
            for j, el in enumerate(grad.containers):
                el.write(f"gradient_{gradient_origin[i]}_{i}_{j}.hv")

    s_inv = get_s_inv([pet_obj_funs, spect_obj_funs], initial_estimates, compare_preconds=compare_preconds)
    
    # set up preconditioners
    bsrem_precond = BSREMPreconditioner(s_inv, freeze_iter=len(all_funs)*5)
    prior_precond = PriorInvHessianDiagPreconditioner(prior, update_interval=len(all_funs), freeze_iter=len(all_funs)*5)
    precond = HarmonicMeanPreconditioner([bsrem_precond, prior_precond], update_interval=len(all_funs), freeze_iter=len(all_funs)*10)
    #precond = MeanPreconditioner([bsrem_precond, prior_precond], update_interval=len(all_funs), freeze_iter=len(all_funs)*10)
    precond = bsrem_precond
    
    if compare_preconds:
        # evaluate bsrem preconditioner on initial images
        initial_bsrem = initial_estimates * bsrem_precond.s_inv
        for i, el in enumerate(initial_bsrem.containers):
            el.write(f"bsrem_precond_{i}.hv")
            
    # probabilities need to reflect the different number of subsets and prior probability 
    # so 1 full update is done every iteration
    pet_probs = [1/len(pet_obj_funs)]*len(pet_obj_funs)
    spect_probs = [1/len(spect_obj_funs)]*len(spect_obj_funs)
    probs = pet_probs + spect_probs
    probs = [p/sum(probs) for p in probs]

    # normalize the probabilities to account for prior probability
    scale = 1 - args.prior_probability / sum(probs)
    probs = [p * scale for p in probs]
    probs += [args.prior_probability]

    # assert that the probabilities sum to 1
    assert abs(sum(probs) - 1) < 1e-10, f"Probabilities do not sum to 1, got {sum(probs)}"

    # set up gradient variance reduction function
    # snaphshot update interval needs to take into account that the prior is updated every prob/2 iterations
    snapshot_update_interval = len(all_funs)*2 + round(0.5/args.prior_probability * len(all_funs))
    f = -SVRGFunction(all_funs, sampler = Sampler.random_with_replacement(len(all_funs),prob=probs), 
                      snapshot_update_interval=snapshot_update_interval, store_gradients=True)
    
    # And finally set up the algorithm
    algo = ISTA(initial = initial_estimates, f=f, g = BlockIndicatorBox(lower=0, upper = np.inf), preconditioner=precond,
                step_size=LinearDecayStepSizeRule(args.initial_step_size, args.relaxation_eta),
                update_objective_interval = snapshot_update_interval//2,)
    
    # and now we run it
    algo.run(len(all_funs)*10,  verbose=1,
             callbacks=[SaveImageCallback("bsrem", snapshot_update_interval//2), 
                        SaveGradientUpdateCallback("gradient_update", snapshot_update_interval//2),
                        PrintObjectiveCallback(snapshot_update_interval//2)],
             )
    
    return algo
    
if __name__ == "__main__":
    
    # Create a profiler instance
    profiler = cProfile.Profile()
    
    # Redirect messages if needed
    msg = MessageRedirector()

    # Create a dataframe for all arguments and save as CSV
    df_args = pd.DataFrame([vars(args)])
    df_args.to_csv(os.path.join(args.output_path, "args.csv"))

    # Print all arguments
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Run main function and retrieve result
    profiler.enable()
    
    bsrem = main(args)
    
    profiler.disable()
    
    # Output results to a file
    with open("profile_results.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()  # Remove extraneous path information
        stats.sort_stats("tottime")  # Sort by individual function time
        stats.print_stats()

    # Ensure output path exists
    os.makedirs(args.output_path, exist_ok=True)

    # Save reconstructed images based on type
    if isinstance(bsrem.x, ImageData):
        bsrem.x.write(os.path.join(args.output_path, f"bsrem_a_{args.alpha}_b_{args.beta}.hv"))

    elif isinstance(bsrem.x, BlockDataContainer):
        for i, el in enumerate(bsrem.x.containers):
            el.write(os.path.join(args.output_path, f"bsrem_modality_{i}_a_{args.alpha}_b_{args.beta}.hv"))

    # Save loss data
    df_objective = pd.DataFrame([l for l in bsrem.loss])
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