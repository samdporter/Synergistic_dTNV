#!/usr/bin/env python3
#%%
import sys
import os
import numpy as np
import argparse
from types import MethodType
import pandas as pd
import shutil
import logging
from typing import Tuple, List, Any

# SIRF imports
from sirf.STIR import (
    ImageData,
    AcquisitionData,
    MessageRedirector,
    KOSMAPOSLReconstructor,
    make_Poisson_loglikelihood,
    AcquisitionModelUsingParallelproj
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
)
from cil.optimisation.functions import (
    SVRGFunction,
    SAGAFunction,
    SGFunction,
    SumFunction,
    BlockFunction
)
from cil.optimisation.functions import (
    OperatorCompositionFunction,
    ZeroFunction,
)
from cil.optimisation.algorithms import ISTA
from cil.optimisation.utilities import Sampler, StepSizeRule

try:
    from ..main_functions import *
except:
    from main_functions import *

#%%
def parse_spect_res(x):
    vals = x.split(',')
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("spect_res must be 3 values: float,float,bool")
    return float(vals[0]), float(vals[1]), vals[2].lower() == 'true'

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="BSREM")
    parser.add_argument("--alpha", type=float, default=1000000, help="alpha")
    parser.add_argument("--beta", type=float, default=1, help="beta")
    parser.add_argument("--delta", type=float, default=None, help="delta")
    parser.add_argument(
        "--num_subsets", type=str, default=9, help="number of subsets"
    )
    parser.add_argument("--no_prior", action="store_true", help="no prior")
    parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--use_kappa", action="store_true", help="use kappa")
    parser.add_argument(
        "--initial_step_size", type=float, default=1, help="initial step size"
    )
    parser.add_argument("--iterations", type=int, default=250, help="max iterations")
    parser.add_argument(
        "--update_interval", type=int, default=None, help="update interval"
    )
    parser.add_argument(
        "--relaxation_eta", type=float, default=0.02, help="relaxation eta"
    )
    parser.add_argument(
        "--tail_singular_values", 
        type=int,
        default=None,
        help="Number of largest singular values to keep"
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
        default=(6.2, 6.2, 6.2),
        help="Gaussian FWHM for smoothing."
    )
    parser.add_argument(
    "--spect_res",
    type=parse_spect_res,
    default=(2.1, 0.02, False),
    default=None,
    help="Tuple of (float, float, bool) for SPECT resolution and use flag (e.g. 0.0923,0.03,True)"
    )
    parser.add_argument(
        "--pet_data_path",
        type=str,
        default="/home/storage/prepared_data/oxford_patient_data/sirt3/PET",
        help="pet data path",
    )
    parser.add_argument(
        "--spect_data_path",
        type=str,
        default="/home/storage/prepared_data/oxford_patient_data/sirt3/SPECT",
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
    parser.add_argument(
        "--use_tof",
        action="store_true",
        default=False,
        help="use TOF",
    )
    parser.add_argument("--save_images", type=bool, default=True, help="save images")
    parser.add_argument("--seed", type=int, default=None, help="numpy seed")
    parser.add_argument("--no_gpu", action="store_true", help="Disables GPU")
    parser.add_argument(
        "--stop_keep_all_views_in_cache",
        action="store_false",
        default=True,
        help="Do not keep all views in cache",
    )
    return parser.parse_known_args()[0]


args = parse_arguments()

#%%
# Add the path to the sys.path
sys.path.append(args.source_path)
from structural_priors.VTV import WeightedVectorialTotalVariation
from utilities.data import get_pet_data_multiple_bed_pos, get_spect_data
from utilities.functions import get_pet_am, get_spect_am
from utilities.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    HarmonicMeanPreconditioner,
    MeanPreconditioner,
    ClampedMeanPreconditioner,
    ClampedHarmonicMeanPreconditioner,
)

from utilities.callbacks import (
    SaveImageCallback,
    SaveGradientUpdateCallback,
    PrintObjectiveCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
)
from KEM.kem import KernelOperator
from utilities.nifty import NiftyResampleOperator
from utilities.sirf import (
    get_block_objective,
    get_s_inv_from_subset_objs,
    get_sensitivity_from_subset_objs,
    get_filters,
)
from utilities.cil import (
    BlockIndicatorBox,
    LinearDecayStepSizeRule,
    NaNToZeroOperator,
    AdjointOperator,
)
from utilities.shifts import (
    CouchShiftOperator,
    ImageCombineOperator,
    get_couch_shift_from_sinogram
)

#%%
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

def save_images(output_alpha, output_x, out_dir, out_suffix=""):
    """
    Save the reconstructed images to disk.
    """
    output_alpha.write(os.path.join(out_dir, "reconstruction_alpha" + out_suffix))
    output_x.write(os.path.join(out_dir, "reconstruction_x" + out_suffix))
        
        
def prepare_data(args):
    """
    Prepare theumapimage, PET and SPECT data, and initial estimates.

    Returns:
        ct: Normalizedumapimage.
        pet_data: Dictionary containing PET data.
        spect_data: Dictionary containing SPECT data.
        initial_estimates: BlockDataContainer combining PET and SPECT initial images.
        cyl, gauss: Filter objects.
    """
    
    # get guidance image
    umap= ImageData(os.path.join(args.pet_data_path, "umap_zoomed.hv"))
    # Normalizeumapimage
    umap+= (-umap).max()
    umap/=umap.max()

    
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof = args.use_tof,
        suffixes=["_f1b1", "_f2b1"]
    )
    
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
        args.delta = max(
            pet_data["initial_image"].max() / 1e2,
            spect_data["initial_image"].max() / 1e2,
        ) * min(args.alpha, args.beta)

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )
    
    for i, image in enumerate(initial_estimates.containers):
        image.write(os.path.join(args.output_path, f"initial_image_{i}.hv"))

    # check for nans in all data
    for data in [umap, pet_data["initial_image"], spect_data["initial_image"]]:
        if np.isnan(data.as_array()).any():
            logging.warning("An image contains NaNs")
            break

    return umap, pet_data, spect_data, initial_estimates

def get_shift_operator(args, pet_data):
    """
    Set up the shift operator for the image reconstruction.

    Returns:
        shift_operator: The constructed shift operator.
    """
    
    suffixes=["_f1b1", "_f2b1"]
    
    pet_shifts = [
        get_couch_shift_from_sinogram(
            pet_data['bed_positions'][suffix]["acquisition_data"])
        for suffix in suffixes
    ]
    
    shift_ops = [
        CouchShiftOperator(pet_data['bed_positions'][suffix]["template_image"], pet_shift)
        for suffix, pet_shift in zip(suffixes, pet_shifts)
    ]
    
    shifted_images = [
        op.direct(pet_data['bed_positions'][suffix]["template_image"])
        for suffix, op in zip(suffixes, shift_ops)
    ]
    
    combine_op = ImageCombineOperator(
        EnhancedBlockDataContainer(
            *shifted_images
        )
    )
    
    unshift_ops = [
        AdjointOperator(op)
        for op in shift_ops
    ]
    
    uncombine_op = AdjointOperator(
        combine_op
    )
    
    choose_op_0 = BlockOperator(
        IdentityOperator(
            shifted_images[0]
        ),
        ZeroOperator(
            shifted_images[1],
            shifted_images[0]
        ),
        shape=(1, 2)
    )
    
    choose_op_1 = BlockOperator(
        ZeroOperator(
            shifted_images[0],
            shifted_images[1]
        ), 
        IdentityOperator(
            shifted_images[1]
        ),
        shape=(1, 2)
    )
    
    choose_ops = [
        choose_op_0,
        choose_op_1
    ]
        
    return uncombine_op, unshift_ops, choose_ops

def get_data_fidelity(
    args, pet_data,
    get_pet_am,
    num_subsets, uncombine_op,
    unshift_ops, choose_ops
):
    """
    Set up data fidelity (objective) functions.
    
    Returns:
        List of objective functions.
    """
    # Partition data for each bed position.
    pet_dfs = [
        partitioner.data_partition(
            pet_data['bed_positions'][suffix]["acquisition_data"],
            pet_data['bed_positions'][suffix]["additive"],
            pet_data['bed_positions'][suffix]["normalisation"],
            num_batches=args.num_subsets,
            mode="staggered",
            create_acq_model=get_pet_am,
        )[2]
        for suffix in pet_data["bed_positions"]
    ]

    # Set up the objective functions for each bed position.
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(pet_data['bed_positions'][suffix]["template_image"])
            
        
    # Before we add all the complicated operators, we'll get the kappa images and sensitivity images
    pet_sens = [
        get_sensitivity_from_subset_objs(
            df, pet_data['bed_positions'][suffix]["template_image"]
            ) 
        for df, suffix in zip(pet_dfs, pet_data["bed_positions"])
    ]
    
    # need to unshift and then combine pet images
    pet_sens_combined = uncombine_op.adjoint(
        EnhancedBlockDataContainer(*[
            unshift_op.adjoint(s)
            for unshift_op, s in zip(unshift_ops, pet_sens)
        ]) 
    )
    s_inv = pet_sens_combined.clone()
    sens_array = pet_sens_combined.as_array()
    s_inv.fill(
        np.reciprocal(
            sens_array,
            where=sens_array != 0,
        )
    )
    cyl, _ = get_filters()
    cyl.apply(s_inv)
    
    # save the s_inv images
    for i, image in enumerate(s_inv.containers):
        image.write(os.path.join(args.output_path, f"s_inv_{i}.hv"))
        logging.info(f"Writing s_inv_{i} with max {image.max()}")
            
    # For each bed position, update each subset operator.
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            # Replace the operator with the composed operator that applies:
            # uncombine -> choose -> unshift, then the original op.
            pet_dfs[i][j] = OperatorCompositionFunction(
                pet_dfs[i][j],
                CompositionOperator(
                    unshift_ops[i],
                    choose_ops[i],
                    uncombine_op,
                ),
            )

    # At this point, pet_dfs is a list (over beds) of lists (over subsets).
    # Flatten the list so you obtain one function per subset per bed.
    pet_combined_dfs = [df for bed in pet_dfs for df in bed]

    all_funs = pet_combined_dfs

    return all_funs, s_inv, None

def get_default_hyperparams(args):
    """
    Return a dictionary of default hyperparameters for the kernel operator.
    """
    hyperparams = {
        "num_neighbours": args.num_neighbours,
        "num_non_zero_features": args.num_non_zero_features,
        "sigma_m": args.sigma_m,
        "sigma_p": args.sigma_p,
        "sigma_dm": args.sigma_dm,
        "sigma_dp": args.sigma_dp,
        "only_2D": args.only_2D,
        "hybrid": args.hybrid,
    }
    return hyperparams

def get_kernel_operator(args, guide_image, template_image, template_sinogram, hyperparams):
        
    K = KernelOperator(
        template_image,
        template_sinogram,
        guide_image,
        num_neighbours=hyperparams["num_neighbours"],
        num_non_zero_features=hyperparams["num_non_zero_features"],
        sigma_m=hyperparams["sigma_m"],
        sigma_p=hyperparams["sigma_p"],
        sigma_dm=hyperparams["sigma_dm"],
        sigma_dp=hyperparams["sigma_dp"],
        only_2D=hyperparams["only_2D"],
        hybrid=hyperparams["hybrid"]
    )

def get_probabilities(args, num_subsets, update_interval):
    pet_probs = [1 / update_interval] * num_subsets[0] * 2
    spect_probs = [1 / update_interval] * num_subsets[1]
    probs = pet_probs + spect_probs
    assert abs(sum(probs) - 1) < 1e-10, f"Probabilities do not sum to 1, got {sum(probs)}"
    return probs

def get_callbacks(args, update_interval: int) -> List[Any]:
    """
    Set up callbacks for the algorithm.

    Returns:
        A list of callback objects.
    """
    return [
        SaveImageCallback(os.path.join(args.output_path, "alpha"), update_interval),
        SaveGradientUpdateCallback(os.path.join(args.output_path, "gradient"), update_interval),
        SavePreconditionerCallback(os.path.join(args.output_path, "preconditioner"), update_interval),
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
    bsrem: ISTA, args: argparse.Namespace
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
        
    save_args(args, "args.csv")

    # Data preparation.
    umap, pet_data, initial_estimates = prepare_data(args)
    
    get_pet_am_with_res = lambda: get_pet_am(
        not args.no_gpu,
        gauss_fwhm=args.pet_gauss_fwhm,
    )
    
    uncombine_op, unshift_ops, choose_ops = get_shift_operator(args, pet_data)


    # Set up data fidelity functions.
    num_subsets = [int(i) for i in args.num_subsets.split(",")]
    all_funs, s_inv, kappa = get_data_fidelity(
        args, 
        pet_data, 
        get_pet_am_with_res,
        uncombine_op,
        unshift_ops,     
        choose_ops
    )
        
    update_interval = len(all_funs)

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

    callbacks = get_callbacks(args, update_interval)

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

    save_results(algo, args)
    logging.info("Done")

#%%
if __name__ == "__main__":
    main()
