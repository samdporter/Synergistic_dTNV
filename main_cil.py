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
from cil.optimisation.operators import (BlockOperator, ZeroOperator, 
                                        IdentityOperator, LinearOperator,
                                        CompositionOperator)
from cil.optimisation.functions import (SVRGFunction, SAGAFunction, BlockFunction,
                                        OperatorCompositionFunction, ZeroFunction, 
                                        SGFunction, KullbackLeibler, SumFunction)
from cil.optimisation.algorithms import ISTA
from cil.optimisation.utilities import Sampler, StepSizeRule

parser = argparse.ArgumentParser(description='BSREM')

parser.add_argument('--alpha', type=float, default=128, help='alpha')
parser.add_argument('--beta', type=float, default=0.5, help='beta')
parser.add_argument('--delta', type=float, default=None, help='delta')
parser.add_argument('--num_subsets', type=str, default="18,12", help='number of subsets')
parser.add_argument('--num_prior_calls', type=float, default=6, help='prior probability')
parser.add_argument('--use_kappa', action='store_true', help='use kappa')
parser.add_argument('--initial_step_size', type=float, default=1, help='initial step size')

parser.add_argument('--iterations', type=int, default=250, help='max iterations')
parser.add_argument('--update_interval', type=int, default=None, help='update interval')
parser.add_argument('--relaxation_eta', type=float, default=0.02, help='relaxation eta')

#/home/storage/copied_data/data/phantom_data/for_cluster, /home/sam/working/OSEM/simple_data
parser.add_argument('--data_path', type=str, default="/home/storage/copied_data/data/phantom_data/for_cluster", help='data path')
parser.add_argument('--output_path', type=str, default="/home/sam/working/BSREM_PSMR_MIC_2024/results/test", help='output path')
parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/src', help='source path')
parser.add_argument('--working_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/tmp', help='working path')
parser.add_argument('--save_images', type=bool, default=True, help='save images')
parser.add_argument('--speedy', action='store_true', help='Speedy mode - no corrections for SPECT')

# set numpy seed - None if not set
parser.add_argument('--seed', type=int, default=None, help='numpy seed')

parser.add_argument('--algorithm', type=str, default="svrg", help='algorithm - gd, mlem, ordered, stochastic, saga or svrg')
parser.add_argument('--single_modality_update', action='store_true', help='Enables single modality update')
parser.add_argument('--prior_is_subset', action='store_true', help='Sets prior as subset')
parser.add_argument('--no_gpu', action='store_true', help='Disables GPU')
parser.add_argument('--keep_all_views_in_cache', action='store_false', default=True, help='Do not keep all views in cache')
parser.add_argument('--use_cil_kl', action='store_true', help='Use CIL Kullback-Leibler')

args = parser.parse_args()

# Imports from my stuff and SIRF contribs
sys.path.insert(0, args.source_path)
from structural_priors.VTV import WeightedVectorialTotalVariation
from utilities.data import get_pet_data, get_spect_data
from utilities.functions import get_pet_am, get_spect_am
from utilities.preconditioners import *
from utilities.callbacks import *
from utilities.nifty import NiftyResampleOperator
from utilities.sirf import (get_block_objective, set_up_partitioned_objectives, 
                            set_up_kl_objectives, get_s_inv_from_obj,
                            compute_inv_hessian_diagonals, get_s_inv_from_am,
                            get_subset_data, get_filters)
from utilities.cil import (BlockIndicatorBox, LinearDecayStepSizeRule, 
                           ZeroEndSlices, ArmijoStepSearchRule, NaNToZeroOperator)

BlockDataContainer.get_uniform_copy = lambda self, n: BlockDataContainer(*[x.clone().fill(n) for x in self.containers])
BlockDataContainer.max = lambda self: max(d.max() for d in self.containers)


# Change to working directory - this is where the tmp_ files will be saved
os.chdir(args.working_path)

def update(self):
    r"""Performs a single iteration of ISTA

    .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))

    Updated so that the gradient update and preconditioned update can be separated

    """
    print("doing new update")
    # gradient step
    self.f.gradient(self.x_old, out=self.gradient_update)

    ### step size choice before the preconditioner
    try:
        step_size = self.step_size_rule.get_step_size(self)
    except NameError:
        raise NameError(msg='`step_size` must be `None`, a real float or a child class of :meth:`cil.optimisation.utilities.StepSizeRule`')

    # preconditioner step - separating preconditioner from gradient update
    if self.preconditioner is not None:
        self.x_old.sapyb(1., self.preconditioner.apply(self, self.gradient_update), -step_size, out=self.x_old)
    else:
        self.x_old.sapyb(1., self.gradient_update, -step_size, out=self.x_old)

    # proximal step
    self.g.proximal(self.x_old, step_size, out=self.x)

ISTA.update = update
    
def main(args):
    """
    Main function to perform image reconstruction using BSREM and PSMR algorithms.
    """
    
    if args.algorithm == "mlem" or args.algorithm == "gd":
        num_subsets = [1 for _ in args.num_subsets.split(",")]
    else:
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
    cyl.apply(spect_data["initial_image"])
    gauss.apply(spect_data["initial_image"])
    cyl.apply(pet_data["initial_image"])
    gauss.apply(pet_data["initial_image"])

    if args.delta is None:
        args.delta = min(pet_data["initial_image"].max()/1e5, spect_data["initial_image"].max()/1e5)

    initial_estimates = BlockDataContainer(pet_data["initial_image"], spect_data["initial_image"])
    
    # set up resampling operators
    pet_data["initial_image"].write("initial_image_0.hv")
    pet2ct = CompositionOperator(NiftyResampleOperator(ct, pet_data["initial_image"], AffineTransformation(os.path.join(args.data_path, "Registration", "pet_to_ct_smallFOV.txt"))), NaNToZeroOperator(pet_data["initial_image"]))
    zero_pet2ct = ZeroOperator(pet_data["initial_image"], ct)
    
    spect_data["initial_image"].write("initial_image_1.hv")
    spect2ct = CompositionOperator(NiftyResampleOperator(ct, spect_data["initial_image"], AffineTransformation(os.path.join(args.data_path, "Registration", "spect_to_ct_smallFOV.txt"))), NaNToZeroOperator(spect_data["initial_image"]))
    zero_spect2ct = ZeroOperator(spect_data["initial_image"], ct)

    # set up prior
    bo = BlockOperator(pet2ct, zero_spect2ct,
                        zero_pet2ct, spect2ct, 
                        shape = (2,2))
    
    vtv = WeightedVectorialTotalVariation(bo.direct(initial_estimates), [args.alpha, args.beta], args.delta, 
                                          anatomical=ct, gpu=not args.no_gpu, stable=False)
    prior = OperatorCompositionFunction(vtv, bo)

    # set up data fidelity functions
    
    pet_datas, pet_ams, pet_obj_funs = partitioner.data_partition(pet_data['acquisition_data'], pet_data['additive'], 
                                                        pet_data['normalisation'], num_batches=num_subsets[0], mode = "staggered",
                                                        create_acq_model=lambda: get_pet_am(not args.no_gpu, gauss_fwhm=(4.5,7.5,7.5),truncate_cylinder=True))
    
    # Unfortately need to redo this to avoid SPECT bug
    # defining here so we know we are using the same partitioning
    spect_datas ,spect_ams, spect_obj_funs = partitioner.data_partition(spect_data['acquisition_data'], spect_data['additive'],
                                        spect_data['acquisition_data'].get_uniform_copy(1), num_batches=num_subsets[1], mode = "staggered",
                                        create_acq_model=lambda: get_spect_am(spect_data, None, gauss_fwhm=(6.1,6.1,6.1),
                                                                              keep_all_views_in_cache=args.keep_all_views_in_cache))
    
    
    if not args.use_cil_kl:
        del pet_datas, spect_datas, pet_ams, spect_ams
        pet_obj_funs, spect_obj_funs = set_up_partitioned_objectives(pet_data, spect_data, pet_obj_funs, spect_obj_funs)
        s_inv = get_s_inv_from_obj([pet_obj_funs, spect_obj_funs], initial_estimates)
        args.initial_step_size = -args.initial_step_size
        prior_multiplier = -1
    else:       
        pet_norms = get_subset_data(pet_data['normalisation'], num_subsets[0], stagger = "staggered")
        pet_obj_funs, spect_obj_funs = set_up_kl_objectives(pet_data, spect_data, pet_datas, pet_norms, spect_datas, pet_ams, spect_ams)
        s_inv = get_s_inv_from_am([pet_ams, spect_ams], initial_estimates)
        prior_multiplier = 1
    
    if args.algorithm == "mlem":
        all_funs = pet_obj_funs + spect_obj_funs
        update_interval = 1
    
    else:
        # This is where stuff goes wrong
        pet_obj_funs = [get_block_objective(pet_data['initial_image'], spect_data['initial_image'], obj_fun, order=0) for obj_fun in pet_obj_funs]
        spect_obj_funs = [get_block_objective(spect_data['initial_image'], pet_data['initial_image'], obj_fun, order=1) for obj_fun in spect_obj_funs]
        
        all_funs = pet_obj_funs + spect_obj_funs
        
        if args.num_prior_calls is None:
            args.num_prior_calls = len(all_funs)
        
        # now the prior needs to be scaled according to the probability of the prior
        # A probability of 0.5 means the prior is used every other iteration
        # therefore the prior needs to be scaled by 1/(2*len(pet_obj_funs + spect_obj_funs))
        if args.num_prior_calls > 0:
            prior = prior_multiplier/args.num_prior_calls * prior
        else:
            prior = ZeroFunction()
        
        update_interval = len(all_funs) + args.num_prior_calls   

        # This is a scaled function but if we multiply / divide by a scalar, it will affect the preconditioner
        # Effectively scaling twice so we won't scale the hessian diag 
        # Possibly there is a better way to do this. This is pretty horrible
        
        def inv_hessian_diag(self, x, out=None):
            ret = self.function.operator.adjoint(self.function.function.inv_hessian_diag(self.function.operator.direct(x)))
            if out is not None:
                out.fill(ret)
            return ret
        
        prior.inv_hessian_diag = MethodType(inv_hessian_diag, prior)

        all_funs.append(prior)
    
    # set up preconditioners
    # Define the lambda function
    #inv_hessian_data_function = lambda solution: compute_inv_hessian_diagonals(solution, [pet_obj_funs, spect_obj_funs])
    #data_precond = ImageFunctionPreconditioner(inv_hessian_data_function, 1, update_interval=update_interval, freeze_iter=np.inf)
    #precond = HarmonicMeanPreconditioner([data_precond, prior_precond], update_interval=1, freeze_iter=np.inf)
    #precond = HarmonicMeanPreconditioner([bsrem_precond, prior_precond], update_interval=1, freeze_iter=len(all_funs)*10)
    #precond = bsrem_precond
    #precond = IdentityPreconditioner()
    #const_precond = ConstantPreconditioner(compute_inv_hessian_diagonals(initial_estimates, [pet_obj_funs, spect_obj_funs]))
    #precond = HarmonicMeanPreconditioner([const_precond, prior_precond], update_interval, freeze_iter=np.inf)
    #precond = const_precond
    #precond = MeanPreconditioner([const_precond, prior_precond], update_interval, freeze_iter=np.inf)

    bsrem_precond = BSREMPreconditioner(s_inv, 1, len(all_funs)*5, epsilon = args.delta)
    prior_precond =ImageFunctionPreconditioner(prior.inv_hessian_diag, 1, update_interval, freeze_iter=np.inf)
    precond = HarmonicMeanPreconditioner([bsrem_precond, prior_precond], update_interval=update_interval, freeze_iter=len(all_funs)*10)

    if args.num_prior_calls == 0 or args.algorithm == "mlem":
        if not isinstance(precond, IdentityPreconditioner):
            precond = bsrem_precond
        print("Using BSREM preconditioner because no prior is used")
    if args.algorithm == "ordered":
        f = SGFunction(all_funs, sampler=Sampler.sequential(len(all_funs)))
    elif args.algorithm == "mlem" or args.algorithm == "gd":
        f = SumFunction(*all_funs[-1], *all_funs[:-1]*args.num_prior_calls)
    else:
        # probabilities need to reflect the different number of subsets and prior probability 
        # so 1 full update is done every iteration
        pet_probs = [1/update_interval]*num_subsets[0]
        spect_probs = [1/update_interval]*num_subsets[1]
        probs = pet_probs + spect_probs + [args.num_prior_calls/update_interval]

        # assert that the probabilities sum to 1
        assert abs(sum(probs) - 1) < 1e-10, f"Probabilities do not sum to 1, got {sum(probs)}"

        # set up gradient variance reduction function
        if args.algorithm == "stochastic":
            f = SGFunction(all_funs, sampler = Sampler.random_with_replacement(len(all_funs),prob=probs))
        elif args.algorithm == "svrg":
            # snaphshot update interval needs to take into account that the prior is updated every prob/2 iterations
            f = SVRGFunction(all_funs, sampler = Sampler.random_with_replacement(len(all_funs),prob=probs), 
                            snapshot_update_interval=2*update_interval, store_gradients=True)
        elif args.algorithm == "saga":
            f = SAGAFunction(all_funs, sampler = Sampler.random_with_replacement(len(all_funs),prob=probs))
        else:
            raise ValueError(f"Unknown algorithm {args.algorithm}")
        
    
    if args.use_cil_kl:
        maximiser = False
    else:
        maximiser = True
    
    callback_update_interval = 1
    callbacks = [SaveImageCallback("image", callback_update_interval),
                    SaveGradientUpdateCallback("gradient_update", callback_update_interval),
                    PrintObjectiveCallback(update_interval),
                    SaveObjectiveCallback("objectives", update_interval),
                    SavePreconditionerCallback("preconditioner", callback_update_interval),
                    SubsetValueCallback("subsets", update_interval),]

    armijo_search = ArmijoStepSearchRule(args.initial_step_size, 0.5, 100, 0.01, 100, maximiser=maximiser)
        
    init_algo = ISTA(initial = initial_estimates, f=SumFunction(*all_funs), g = BlockIndicatorBox(lower=0, upper = np.inf), preconditioner=precond,
                    step_size=armijo_search, update_objective_interval = 1)
    
    init_algo.run(5, verbose=1, callbacks=callbacks)

    # And finally set up the algorithm
    algo = ISTA(initial = init_algo.solution, f=f, g = BlockIndicatorBox(lower=0, upper = np.inf), preconditioner=precond,
                step_size=armijo_search.min_step_size,
                update_objective_interval = update_interval,)
    
    print("running algorithm")
    
    # and now we run it
    algo.run(update_interval*50,  verbose=1, callbacks=callbacks,)
    
    return algo

def save_results(bsrem):
    
    # Output results to a file
    with open("profile_results.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()  # Remove extraneous path information
        stats.sort_stats("tottime")  # Sort by individual function time
        stats.print_stats()

    # Ensure output path exists
    os.makedirs(args.output_path, exist_ok=True)

    # Save loss data
    df_objective = pd.DataFrame([l for l in bsrem.loss])
    df_objective.to_csv(os.path.join(args.output_path, f"bsrem_objective_a_{args.alpha}_b_{args.beta}.csv"))

    # Remove temporary files
    for file in os.listdir(args.working_path):
        if file.startswith("tmp_") and (file.endswith(".s") or file.endswith(".hs")):
            os.remove(os.path.join(args.working_path, file))

    # Move leftover files (if any) to the output path
    for file in os.listdir(args.working_path):
        if file.endswith((".hv", ".v", ".ahv")):
            print(f"Moving to {os.path.join(args.output_path, file)}")
            shutil.move(os.path.join(args.working_path, file), os.path.join(args.output_path, file))
    
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
    
    save_results(bsrem)

    print("Done")