#!/usr/bin/env python3
#main_cil_1bpos_skem.py
import sys
import os
import numpy as np
import argparse
from types import MethodType
import pandas as pd
import shutil
import logging
from typing import Tuple, List, Any
import cProfile
import pstats

# SIRF imports
from sirf.STIR import (
    ImageData,
    AcquisitionData,
    MessageRedirector,
)
from sirf.Reg import NiftiImageData3DDisplacement
from sirf.contrib.partitioner import partitioner

AcquisitionData.set_storage_scheme("memory")

# CIL imports
from cil.framework import BlockDataContainer

# Monkey patching for BlockDataContainer
class EnhancedBlockDataContainer(BlockDataContainer):
    def get_uniform_copy(self, n) -> "EnhancedBlockDataContainer":
        """Return a copy with each container filled with n."""
        return EnhancedBlockDataContainer(*[x.clone().fill(n) for x in self.containers])
    
    def max(self) -> float:
        """Return the maximum value across all containers."""
        return max(d.max() for d in self.containers)

from cil.optimisation.operators import (
    BlockOperator,
    ZeroOperator,
    IdentityOperator,
    CompositionOperator,
    LinearOperator,
)
from cil.optimisation.functions import (
    SVRGFunction,
    SAGAFunction,
    SGFunction,
    SumFunction,
)
from cil.optimisation.functions import (
    OperatorCompositionFunction,
    ZeroFunction,
)
from cil.optimisation.algorithms import ISTA
from cil.optimisation.utilities import Sampler, StepSizeRule

try:
    from .main_functions import *
except:
    from main_functions import *


def parse_spect_res(x):
    vals = x.split(',')
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("spect_res must be 3 values: float,float,bool")
    return float(vals[0]), float(vals[1]), vals[2].lower() == 'true'

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="BSREM")
    parser.add_argument("--alpha", type=float, default=300, help="alpha")
    parser.add_argument("--beta", type=float, default=1, help="beta")
    parser.add_argument("--delta", type=float, default=None, help="delta")
    parser.add_argument(
        "--num_subsets", type=str, default="9,12", help="number of subsets"
    )
    parser.add_argument("--no_prior", action="store_true", help="no prior")
    parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--use_kappa", action="store_true", help="use kappa")
    parser.add_argument(
        "--initial_step_size", type=float, default=.1, help="initial step size"
    )
    parser.add_argument(
        "--update_interval", type=int, default=None, help="update interval"
    )
    parser.add_argument(
        "--relaxation_eta", type=float, default=0.02, help="relaxation eta"
    )
    parser.add_argument(
        "--tail_singular_values", 
        type=int,
        default=1,
        help="Number of largest singular values to keep"
    )
    parser.add_argument(
        "--normalise_gradients",
        action="store_true",
        help="Normalise gradients",
    )
    parser.add_argument(
        "--pet_gauss_fwhm",
        type=float,
        nargs=3,
        default=(5.0, 5.0, 5.0),
        help="Gaussian FWHM for smoothing."
    )
    parser.add_argument(
        "--spect_gauss_fwhm",
        type=float,
        nargs=3,
        default=(15.7, 15.7, 15.7),
        help="Gaussian FWHM for smoothing."
    )
    parser.add_argument(
    "--spect_res",
    type=parse_spect_res,
    default=(1.78, 0.03, False),
    help="Tuple of (float, float, bool) for SPECT resolution and use flag (e.g. 1.78,0.03,True)"
    )
    parser.add_argument(
        "--pet_data_path",
        type=str,
        default="/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/PET/phantom",
        help="pet data path",
    )
    parser.add_argument(
        "--multiple_bed_position",
        action="store_true",
        help="multiple bed position",
    )
    parser.add_argument(
        "--spect_data_path",
        type=str,
        default="/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/SPECT/phantom_140",
        help="spect data path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/sam/working/BSREM_PSMR_MIC_2024/results/test",
        help="output path",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/home/sam/working/BSREM_PSMR_MIC_2024/src",
        help="source path",
    )
    parser.add_argument(
        "--working_path",
        type=str,
        default="/home/sam/working/BSREM_PSMR_MIC_2024/tmp",
        help="working path",
    )
    parser.add_argument("--save_images", type=bool, default=True, help="save images")
    parser.add_argument("--seed", type=int, default=None, help="numpy seed")
    parser.add_argument("--no_gpu", action="store_true", help="Disables GPU")
    parser.add_argument(
        "--keep_all_views_in_cache",
        action="store_false",
        default=True,
        help="Do not keep all views in cache",
    )
    parser.add_argument("--num_neighbours", type=int, default=5)
    parser.add_argument("--num_non_zero_features", type=int, default=1)
    parser.add_argument("--sigma_e", type=float, default=1.0)
    parser.add_argument("--sigma_a", type=float, default=3.0)
    parser.add_argument("--sigma_d", type=float, default=5.0)
    parser.add_argument("--only_2D", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    return parser.parse_args()


args = parse_arguments()

# Add the path to the sys.path
sys.path.append(args.source_path)
from structural_priors.VTV import WeightedVectorialTotalVariation
from utilities.data import get_pet_data, get_spect_data
from utilities.functions import get_pet_am, get_spect_am
from utilities.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    HarmonicMeanPreconditioner,
    MeanPreconditioner,
)

from utilities.callbacks import (
    SaveImageCallback,
    SaveGradientUpdateCallback,
    PrintObjectiveCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
)
from utilities.nifty import NiftyResampleOperator
from utilities.sirf import (
    get_block_objective,
    get_sensitivity_from_subset_objs,
    get_filters,
)
from utilities.cil import (
    BlockIndicatorBox,
    LinearDecayStepSizeRule,
    NaNToZeroOperator,
)
from KEM.my_kem import get_kernel_operator


def update_step(self) -> None:
    r"""Perform a single ISTA update iteration.

    .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))
    """
    logging.info("Performing ISTA update step")
    self.f.gradient(self.x_old, out=self.gradient_update)
    try:
        step_size = self.step_size_rule.get_step_size(self)
    except NameError:
        raise NameError(
            "`step_size` must be None, a real float or a child class of "
            "cil.optimisation.utilities.StepSizeRule"
        )
    if self.preconditioner is not None:
        self.x_old.sapyb(
            1.0,
            self.preconditioner.apply(self, self.gradient_update),
            -step_size,
            out=self.x_old,
        )
    else:
        self.x_old.sapyb(1.0, self.gradient_update, -step_size, out=self.x_old)
    self.g.proximal(self.x_old, step_size, out=self.x)


# Attach the new update method to ISTA.
ISTA.update = update_step


def configure_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    

def save_args(args, output_filename):
    # Save command-line arguments.
    os.makedirs(args.output_path, exist_ok=True)
    df_args = pd.DataFrame([vars(args)])
    df_args.to_csv(os.path.join(args.output_path, output_filename), index=False)
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info(f"Arguments saved to {os.path.join(args.output_path, output_filename)}")
        
        
def prepare_data(args):
    """
    Prepare the CT image, PET and SPECT data, and initial estimates.

    Returns:
        ct: Normalized CT image.
        pet_data: Dictionary containing PET data.
        spect_data: Dictionary containing SPECT data.
        initial_estimates: BlockDataContainer combining PET and SPECT initial images.
        cyl, gauss: Filter objects.
    """
    
    # get guidance image
    ct = ImageData(os.path.join(args.pet_data_path, "umap_zoomed.hv"))
    # Normalize CT image
    ct += (-ct).max()
    ct /= ct.max()

    pet_data = get_pet_data(args.pet_data_path)
    spect_data = get_spect_data(args.spect_data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters()

    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])
    
    pet_data["initial_image"].write("initial_image_0.hv")
    spect_data["initial_image"].write("initial_image_1.hv")

    # Set delta (smoothing parameter) if not provided
    if args.delta is None:
        # set delta as 10000 times smaller than maximum of the initial images
        # multiplied by the smallest weighting (alpha/beta)
        args.delta = max(
            pet_data["initial_image"].max() / 1e4,
            spect_data["initial_image"].max() / 1e4,
        ) * min(args.alpha, 1)
            

    initial_estimates = EnhancedBlockDataContainer(
        args.alpha*pet_data["initial_image"], spect_data["initial_image"]
    )
    
    for i, image in enumerate(initial_estimates.containers):
        image.write(os.path.join(args.output_path, f"initial_image_{i}.hv"))

    # check for nans in all data
    for data in [ct, pet_data["initial_image"], spect_data["initial_image"]]:
        if np.isnan(data.as_array()).any():
            logging.warning("An image contains NaNs")
            break
    for data in [
        pet_data["acquisition_data"], spect_data["acquisition_data"],
        pet_data["normalisation"],
        pet_data["additive"], spect_data["additive"]
    ]:
        if np.isnan(data.as_array()).any():
            logging.warning("A ProjData contains NaNs")
            break

    return ct, pet_data, spect_data, initial_estimates


def get_resampling_operators(
    args, pet_data, spect_data,
):
    """
    Set up resampling operators for SPECT images to PET images.

    Returns:
        spect2ct, zero_spect2ct operators.
    """
    spect2pet = CompositionOperator(
        NiftyResampleOperator(
            pet_data["initial_image"],
            spect_data["initial_image"],
            NiftiImageData3DDisplacement(
                os.path.join(args.spect_data_path, "spect2pet.nii")
            ),
        ),
        NaNToZeroOperator(pet_data["initial_image"]),
    )
    return spect2pet


def get_prior(
    args, ct, pet_data, spect_data,
    initial_estimates, spect2pet,
    kappas=None,
):
    """
    Set up the prior function for image reconstruction.

    Returns:
        prior: The constructed prior function.
        bo: The block operator used within the prior.
    """
    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),  # pet2pet
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),  # zero_spect2pet
        ZeroOperator(pet_data["initial_image"]),  # zero_pet2pet
        spect2pet,  # spect2pet
        shape=(2, 2),
    )

    if kappas is not None:
        for i, kappa in enumerate(kappas.containers):
            logging.info(f"Writing kappa {i} with max {kappa.max()}")
            kappa.write(os.path.join(args.output_path, f"kappa_{i}.hv"))
        kappas = bo.direct(kappas)
    else:
        kappas = EnhancedBlockDataContainer(
            pet_data["initial_image"].get_uniform_copy(1),
            spect_data["initial_image"].get_uniform_copy(1),
        )
        kappas = bo.direct(kappas)

    vtv = WeightedVectorialTotalVariation(
        bo.direct(initial_estimates),
        kappas,
        args.delta,
        anatomical=pet_data["attenuation"],
        gpu=not args.no_gpu,
        stable=False,
        tail_singular_values=args.tail_singular_values, 
        normalise_gradients=args.normalise_gradients,
    )
    prior = OperatorCompositionFunction(vtv, bo)
    return prior

def get_default_hyperparams(args):
    """
    Return a dictionary of default hyperparameters for the kernel operator.
    """
    hyperparams = {
        "num_neighbours": args.num_neighbours,
        "sigma_anat": args.sigma_a,
        "sigma_dist": args.sigma_d,
        "sigma_emission": args.sigma_e,
        'kernel_type': 'neighbourhood',
        'normalize_features': True,
        'normalize_kernel': True,
        'use_mask': False,
        'mask_k': 48,
        'recalc_mask': False,
        'distance_weighting': True,
        'hybrid': args.hybrid,

    }
    return hyperparams

def get_data_fidelity(
    args, pet_data, spect_data,
    get_pet_am, get_spect_am,
    num_subsets
):
    """
    Set up data fidelity (objective) functions.

    Returns:
        all_funs: List of all objective functions.
        update_interval: Update interval used by the algorithm.
        s_inv: Sensitivity image ^ -1.
        pet_ams, spect_ams: Acquisition model components.
    """
    # Partition PET data.
    
    _, _, pet_obj_funs = partitioner.data_partition(
        pet_data["acquisition_data"],
        pet_data["additive"],
        pet_data["normalisation"],
        num_batches=num_subsets[0],
        mode="staggered",
        create_acq_model=get_pet_am,
    )
    _, _, spect_obj_funs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"],
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am,
    )

    for obj_fun in pet_obj_funs:
        obj_fun.set_up(pet_data['initial_image'])
    for obj_fun in spect_obj_funs:
        obj_fun.set_up(spect_data['initial_image'])

    # Get sensitivity image ^ -1 now before we complicate things
    sens=[
        get_sensitivity_from_subset_objs(obj_funs, initial_image)
            for obj_funs, initial_image in zip(
                [pet_obj_funs, spect_obj_funs],
                [pet_data["initial_image"],spect_data["initial_image"]]
            )
    ]
    
    hyperparams = get_default_hyperparams(args)

    K_pet = get_kernel_operator(
        pet_data["initial_image"],
        backend='numba',
    )
    K_pet.parameters = hyperparams
    K_pet.set_anatomical_image(pet_data["attenuation"])

    pet_obj_funs = [
        OperatorCompositionFunction(obj_fun, K_pet)
        for obj_fun in pet_obj_funs
    ]

    K_spect = get_kernel_operator(
        spect_data["initial_image"],
        backend='numba',
    )
    K_spect.parameters = hyperparams
    K_spect.set_anatomical_image(spect_data["attenuation"])
    
    spect_obj_funs = [
        OperatorCompositionFunction(obj_fun, K_spect)
        for obj_fun in spect_obj_funs
    ]

    s_inv = [K.adjoint(s) for K, s in zip(
        [K_pet, K_spect], sens
        )]
    
    for s in s_inv:
        s_arr = s.as_array()
        s.fill(np.reciprocal(s_arr, where=s_arr > 0))
    s_inv = EnhancedBlockDataContainer(*s_inv)
        
    pet_obj_funs = [
        get_block_objective(
            pet_data["initial_image"],
            spect_data["initial_image"],
            obj_fun,
            scale = 1/args.alpha,
            order=0,
        )
        for obj_fun in pet_obj_funs
    ]
    spect_obj_funs = [
        get_block_objective(
            spect_data["initial_image"],
            pet_data["initial_image"],
            obj_fun,
            scale = 1,
            order=1,
        )
        for obj_fun in spect_obj_funs
    ]
    
    _, gauss = get_filters()
    if args.use_kappa:
        kappa = get_kappa_squareds(
            [pet_obj_funs, spect_obj_funs], 
            [pet_data["initial_image"], spect_data["initial_image"]],
            normalise=True
        )
        for kappa_image in kappa.containers:
            gauss.apply(kappa_image)
        all_funs = pet_obj_funs + spect_obj_funs
    else:
        kappa = None

    all_funs = pet_obj_funs + spect_obj_funs

    return all_funs, s_inv, kappa, [K_pet, K_spect]


def get_kappa_squareds(obj_funs_list, image_list, normalise=False):
    """
    Compute the kappa squared images for each objective function.

    Returns:
        kappa_squareds: List of kappa squared images.
    """
    kappa_squareds = []
    for obj_funs, image in zip(obj_funs_list, image_list):
        kappa_squareds.append(
            compute_kappa_squared_image_from_partitioned_objective(obj_funs, image, normalise)
        )
    return EnhancedBlockDataContainer(*kappa_squareds)


def get_preconditioners(
    args: argparse.Namespace,
    s_inv: Any,
    all_funs: List[Any],
    update_interval: int,
    prior: Any, initial_estimates: EnhancedBlockDataContainer,
) -> Any:
    """
    Set up the preconditioners.

    Returns:
        The combined preconditioner.
    """
    
    max_vals = [el.max() for el in initial_estimates.containers]
    minmax_val = min(max_vals)
    
    bsrem_precond = BSREMPreconditioner(
        s_inv, 1, np.inf, 
        epsilon=0,
        max_vals=max_vals,
    )
    if prior is None:
        return bsrem_precond
    
    prior_precond = ImageFunctionPreconditioner(
        prior.inv_hessian_diag, 1., 
        update_interval, 
        freeze_iter=np.inf,
    )
    
    precond = HarmonicMeanPreconditioner(
        [bsrem_precond, prior_precond],
        update_interval=update_interval,
        freeze_iter=len(all_funs) * 10,
    )
    
    return precond

def get_probabilities(args, num_subsets, update_interval):
    pet_probs = [1 / update_interval] * num_subsets[0]
    spect_probs = [1 / update_interval] * num_subsets[1]
    probs = pet_probs + spect_probs
    assert abs(sum(probs) - 1) < 1e-10, f"Probabilities do not sum to 1, got {sum(probs)}"
    return probs

class SaveKernelisedImageCallback(SaveImageCallback):
    def __init__(self, filename, interval, Ks, **kwargs):
        super().__init__(filename, interval, **kwargs)
        self.Ks = Ks
        
    def __call__(self, algo):
        if algo.iteration % self.interval != 0:
            return
        # Save the alpha image
        for i, alpha in enumerate(algo.solution.containers):
            image = self.Ks[i].direct(alpha)
            image.write(f"{self.filename}_{i}_{algo.iteration}.hv")

def get_callbacks(args, update_interval: int, Ks) -> List[Any]:
    """
    Set up callbacks for the algorithm.

    Returns:
        A list of callback objects.
    """
    return [
        SaveImageCallback(os.path.join(args.output_path, "alpha"), update_interval),
        SaveKernelisedImageCallback(
            os.path.join(args.output_path, "image"), update_interval, Ks),
        SaveGradientUpdateCallback(os.path.join(args.output_path, "gradient"), update_interval),
        SavePreconditionerCallback(os.path.join(args.output_path, "preconditioner"),update_interval),
        PrintObjectiveCallback(update_interval),
        SaveObjectiveCallback(os.path.join(args.output_path, "objective"), update_interval),
    ]


def get_algorithm(
    init_solution: EnhancedBlockDataContainer,
    f_obj: Any,
    precond: Any,
    step_size: float,
    update_interval: int,
    subiterations: int,
    callbacks: List[Any],
) -> ISTA:
    """
    Set up and run the ISTA algorithm.

    Returns:
        The ISTA instance.
    """
    algo = ISTA(
        initial=init_solution,
        f=f_obj,
        g=BlockIndicatorBox(lower=0, upper=np.inf),
        preconditioner=precond,
        step_size=step_size,
        update_objective_interval=update_interval,
    )
    logging.info("Running algorithm")
    algo.run(subiterations, verbose=1, callbacks=callbacks)
    return algo


def save_results(
    bsrem: ISTA, args: argparse.Namespace, Ks
) -> None:
    """Save profiling information and results to disk."""

    os.makedirs(args.output_path, exist_ok=True)
    df_objective = pd.DataFrame([l for l in bsrem.loss])
    df_objective.to_csv(
        os.path.join(
            args.output_path, f"bsrem_objective_a_{args.alpha}_b_{args.beta}.csv"
        ),
        index=False,
    )
    
    # Save the final image
    for i, alpha in enumerate(bsrem.solution.containers):
        alpha.write(os.path.join(args.output_path, f"bsrem_alpha_{i}.hv"))
        image = Ks[i].direct(alpha)
        image.write(os.path.join(args.output_path, f"bsrem_image_{i}.hv"))
        

    for file in os.listdir(args.working_path):
        if file.startswith("tmp_") and (file.endswith(".s") or file.endswith(".hs")):
            os.remove(os.path.join(args.working_path, file))
    for file in os.listdir(args.working_path):
        if file.endswith((".hv", ".v", ".ahv")):
            logging.info(f"Moving file {file} to {args.output_path}")
            shutil.move(
                os.path.join(args.working_path, file),
                os.path.join(args.output_path, file),
            )


def main() -> None:
    """Main function to execute the image reconstruction algorithm."""
    configure_logging()
    os.chdir(args.working_path)

    # Redirect messages if needed.
    msg = MessageRedirector()

    # Data preparation.
    ct, pet_data, spect_data, initial_estimates = prepare_data(args)

    save_args(args, "args.csv")
    
    # Set up resampling operators.
    spect2pet = get_resampling_operators(args, pet_data, spect_data)
    
    get_pet_am_with_res = lambda: get_pet_am(
        not args.no_gpu,
        gauss_fwhm=args.pet_gauss_fwhm,
    )
    
    get_spect_am_with_res = lambda: get_spect_am(
        spect_data,
        res=args.spect_res,
        keep_all_views_in_cache=args.keep_all_views_in_cache,
        gauss_fwhm=args.spect_gauss_fwhm,
        attenuation=True,
    )

    # Set up data fidelity functions.
    num_subsets = [int(i) for i in args.num_subsets.split(",")]
    all_funs, s_inv, kappa, Ks = get_data_fidelity(
        args, 
        pet_data, spect_data, 
        get_pet_am_with_res,
        get_spect_am_with_res,
        num_subsets
    )
    
    if args.no_prior:
        prior = None
    else:
        # Set up the prior.
        prior = get_prior(
                args, ct, pet_data, 
                spect_data, initial_estimates, 
                spect2pet, kappa
                )
        # Scale and attach Hessian to the prior if needed.
        prior = -args.beta / len(all_funs) * prior
        attach_prior_hessian(prior)

        for i, fun in enumerate(all_funs):
            all_funs[i] = SumFunction(fun, prior)
        
    update_interval = len(all_funs)
    
    # Set up preconditioners.
    precond = get_preconditioners(
        args, s_inv, all_funs, 
        update_interval, prior,
        initial_estimates
    )

    probs = get_probabilities(args, num_subsets, update_interval)
    
    f_obj = -SVRGFunction(
            all_funs, sampler=Sampler.random_with_replacement(len(all_funs), prob=probs,),
            snapshot_update_interval=update_interval*2, store_gradients=True
        )
    #f_obj = -SAGAFunction(
    #        all_funs, sampler=Sampler.random_with_replacement(len(all_funs), prob=probs,),
    #    )
    #f_obj.function.warm_start_approximate_gradients(initial_estimates)
    #f_obj = -SGFunction(
    #        all_funs, sampler=Sampler.random_with_replacement(len(all_funs), prob=probs,),
    #    )

    callbacks = get_callbacks(args, update_interval, Ks)

    algo = get_algorithm(
        initial_estimates,
        f_obj,
        precond,
        LinearDecayStepSizeRule(
            args.initial_step_size,
            args.relaxation_eta,
        ),
        update_interval,
        args.num_epochs * update_interval,
        callbacks,
    )

    save_results(algo, args, Ks)
    logging.info("Done")


if __name__ == "__main__":

    # set up profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()

    profiler.disable()
    # Output results to a file
    output_file = os.path.join(args.output_path, "profiling_results.txt")
    with open(output_file, "w") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.strip_dirs().sort_stats("cumulative").print_stats()
    logging.info(f"Profiling results saved to {output_file}")
