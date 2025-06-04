#!/usr/bin/env python3
"""
Hybrid Kernelized Expectation Maximization Reconstruction Demo

This master script can run two different reconstruction algorithms:
  1. KOSMAPOSL (Kernelized Ordered Subsets Maximum A-Posteriori One Step Late)
  2. ISTA-based reconstruction

Hyperparameters for the kernel operator are stored in a dictionary.
"""

import os
import sys
import argparse
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    IndicatorBox, SGFunction,
    OperatorCompositionFunction
)
from cil.optimisation.utilities import Sampler
from cil.optimisation.operators import (
    LinearOperator, CompositionOperator
)

from sirf.contrib.partitioner import partitioner
from sirf.STIR import (
    AcquisitionData, 
    MessageRedirector,
    TruncateToCylinderProcessor,
    AcquisitionSensitivityModel, 
    KOSMAPOSLReconstructor,
    make_Poisson_loglikelihood,
)
from sirf.Reg import AffineTransformation

# Set storage scheme and redirect messages
AcquisitionData.set_storage_scheme("memory")
msg = MessageRedirector()

# Ensure local modules are found
sys.path.insert(0, "/home/sam/working/BSREM_PSMR_MIC_2024/src")
from utilities.preconditioners import SubsetKernelisedEMPreconditioner
from utilities.callbacks import SaveImageCallback, PrintObjectiveCallback, Callback
from utilities.data import get_pet_data, get_spect_data
from utilities.functions import get_pet_am, get_spect_am
from KEM.kem import KernelOperator

class Truncate(LinearOperator):

    """CIL Wrapper for SIRF TruncateToCylinderProcessor.
    """

    def __init__(self, domain_geometry, **kwargs):
        super().__init__(
            domain_geometry=domain_geometry, 
            range_geometry=domain_geometry
        )
        
        self.truncate = TruncateToCylinderProcessor()
        self.truncate.set_strictly_less_than_radius(True)

    def __call__(self, x, out=None):
        return self.direct(x, out)

    def direct(self, x, out=None):
        if out is None:
            out = x.copy()
        self.truncate.apply(out)
        return out

    def adjoint(self, x, out=None):
        return self.direct(x, out)

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

def save_images(output_alpha, output_x, out_dir, out_suffix=""):
    """
    Save the reconstructed images to disk.
    """
    output_alpha.write(os.path.join(out_dir, "reconstruction_alpha" + out_suffix))
    output_x.write(os.path.join(out_dir, "reconstruction_x" + out_suffix))


def run_kosmaposl(args, data, hyperparams, get_am):
    """
    Run the KOSMAPOSL reconstruction algorithm.
    """
    
    am = get_am()
    
    if args.modality == "SPECT":
        data["normalisation"] = data["acquisition_data"].get_uniform_copy(1)
        
    am.set_acquisition_sensitivity(
        AcquisitionSensitivityModel(data["normalisation"])
    )
    am.set_additive_term(data["additive"])

    image = data["initial_image"]
    obj_fun = make_Poisson_loglikelihood(data["acquisition_data"])
    obj_fun.set_acquisition_model(am)
    
    out_dir = os.path.join(args.out_path, args.out_suffix)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if args.modality == "PET":
        if args.guidance == "emission":
            guidance = data["spect"]
        else:
            guidance = data["attenuation"]
    else:
        if args.guidance == "emission":
            print("Emission guidance not available, using attenuation")
            guidance = data["attenuation"]
        else:
            guidance = data["attenuation"]

    recon = KOSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(args.num_subsets)
    recon.set_num_subiterations(args.num_subsets * args.num_epochs)
    recon.set_anatomical_prior(guidance)
    recon.set_num_neighbours(hyperparams["num_neighbours"])
    recon.set_num_non_zero_features(hyperparams["num_non_zero_features"])
    recon.set_sigma_m(hyperparams["sigma_m"])
    recon.set_sigma_p(hyperparams["sigma_p"])
    recon.set_sigma_dm(hyperparams["sigma_dm"])
    recon.set_sigma_dp(hyperparams["sigma_dp"])
    recon.set_only_2D(hyperparams["only_2D"])
    recon.set_hybrid(hyperparams["hybrid"])
    recon.enable_output()
    recon.set_save_interval(args.num_subsets)
    recon.set_output_filename_prefix(os.path.join(out_dir, "kosmaposl"))

    print("Setting up KOSMAPOSL reconstruction, please wait...")
    current_alpha = image.get_uniform_copy(1)
    recon.set_up(current_alpha) 
    recon.set_current_estimate(current_alpha)
    recon.reconstruct(current_alpha)

    output_alpha = recon.get_current_estimate()
    output_x = recon.compute_kernelised_image(output_alpha, output_alpha)
    
    save_images(output_alpha, output_x, out_dir, "")

def run_ista(args, data, hyperparams, get_am):
    """
    Run the ISTA-based reconstruction algorithm.
    """
    
    if args.modality == "SPECT":
        data["normalisation"] = data["acquisition_data"].get_uniform_copy(1)
    
    _, _, objs = partitioner.data_partition(
        data["acquisition_data"],
        data["additive"],
        data["normalisation"],
        args.num_subsets,
        mode=args.sampling,
        create_acq_model=get_am
    )
    
    for obj in objs:    
        obj.set_up(data["initial_image"])

    if args.modality == "PET":
        if args.guidance == "emission":
            guidance = data["spect"]
        else:
            guidance = data["attenuation"]
    else:
        if args.guidance == "emission":
            print("Emission guidance not available, using attenuation")
            guidance = data["attenuation"]
        else:
            guidance = data["attenuation"]

    K = KernelOperator(
        data["initial_image"],
        data["acquisition_data"],
        guidance,
        num_neighbours=hyperparams["num_neighbours"],
        num_non_zero_features=hyperparams["num_non_zero_features"],
        sigma_m=hyperparams["sigma_m"],
        sigma_p=hyperparams["sigma_p"],
        sigma_dm=hyperparams["sigma_dm"],
        sigma_dp=hyperparams["sigma_dp"],
        only_2D=hyperparams["only_2D"],
        hybrid=hyperparams["hybrid"]
    )

    f_list = [
        OperatorCompositionFunction(
            obj,
            CompositionOperator(K, Truncate(data["initial_image"]))
        )
        for obj in objs
    ]
    sampler = Sampler.sequential(args.num_subsets)
    f = -SGFunction(f_list, sampler)
    g = IndicatorBox(lower=0)

    sensitivities = [
        obj.get_subset_sensitivity(0)
        for obj in objs
    ]
    for s in sensitivities:
        s = s.maximum(0)

    precond = SubsetKernelisedEMPreconditioner(
        args.num_subsets,
        sensitivities,
        K,
        freeze_iter=args.freeze_iter,
        epsilon=data["initial_image"].max() * 1e-12 # could this be a problem?
    )

    init_alpha = data["initial_image"].get_uniform_copy(1)
    truncate = TruncateToCylinderProcessor()
    truncate.set_strictly_less_than_radius(True)
    truncate.apply(init_alpha)
    num_subiterations = args.num_subsets * args.num_epochs

    algo = ISTA(
        init_alpha,
        f,
        g,
        step_size=1,
        preconditioner=precond,
        max_iteration=num_subiterations,
        update_objective_interval=args.num_subsets
    )

    out_dir = os.path.join(args.out_path, args.out_suffix)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    class SaveKernelisedImageCallback(Callback):
        """
        Save the alpha image to disk.
        """
        def __init__(self, filename, interval, kernel_op,**kwargs):
            super().__init__(interval, **kwargs)
            self.filename = filename
            self.kernel_op = kernel_op


        def __call__(self, algo):
            if algo.iteration % self.interval != 0:
                return
            # Save the alpha image
            image = self.kernel_op.direct(algo.solution)
            image.write(f"{self.filename}_{algo.iteration}.hv")
            
            
    algo.run(
        num_subiterations,
        verbose=True,
        callbacks=[
            SaveImageCallback(
                os.path.join(out_dir, "alpha"),
                interval = args.num_subsets,
            ),
            SaveKernelisedImageCallback(
                os.path.join(out_dir, "x"),
                interval = args.num_subsets,
                kernel_op = K,
            ),
            PrintObjectiveCallback(
                interval = args.num_subsets,
            )
        ]
    )

    output_alpha = algo.solution
    output_x = K.recon.compute_kernelised_image(
        output_alpha, 
        output_alpha
    )
    
    save_images(output_alpha, output_x, out_dir, "")
    
def parse_spect_res(x):
    vals = x.split(',')
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("spect_res must be 3 values: float,float,bool")
    return float(vals[0]), float(vals[1]), vals[2].lower() == 'true'

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Hybrid Kernelized Expectation Maximization Reconstruction"
    )
    parser.add_argument(
        "--modality",
        choices=["PET", "SPECT"],
        default="PET",
        help="Modality to reconstruct."
    )
    parser.add_argument(
        "--guidance",
        choices=["emission", "attenuation"],
        default="attenuation",
        help="Guidance image to use."
    )
    parser.add_argument(
        "--method",
        choices=["kosmaposl", "ista"],
        default="ista",
        help="Reconstruction method to use."
    )
    parser.add_argument(
        "--num_subsets",
        type=int,
        default=9,
        help="Number of subsets."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of num_epochss."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/storage/copied_data/data/phantom_data/for_cluster/PET",
        help="Path to PET data."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/home/sam/working/BSREM_PSMR_MIC_2024/HKEM",
        help="Output base path."
    )
    parser.add_argument(
        "--out_suffix",
        type=str,
        default="output",
        help="Output suffix for files."
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="sequential",
        help="Sampling mode for ISTA."
    )
    parser.add_argument(
        "--gauss_fwhm",
        type=float,
        nargs=3,
        default=(6.2, 6.2, 6.2),
        help="Gaussian FWHM for smoothing."
    )
    parser.add_argument(
    "--spect_res",
    type=parse_spect_res,
    default=(1.22, 0.031, False),
    help="Tuple of (float, float, bool) for SPECT resolution and use flag (e.g. 0.0923,0.03,True)"
    )
    parser.add_argument(
        "--freeze_iter",
        type=int,
        default=np.inf,
        help="Number of iterations to freeze."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/home/sam/working/BSREM_PSMR_MIC_2024/src",
        help="source path",
    )
    parser.add_argument("--num_neighbours", type=int, default=5)
    parser.add_argument("--num_non_zero_features", type=int, default=1)
    parser.add_argument("--sigma_m", type=float, default=1.0)
    parser.add_argument("--sigma_p", type=float, default=0.1)
    parser.add_argument("--sigma_dm", type=float, default=5.0)
    parser.add_argument("--sigma_dp", type=float, default=5.0)
    parser.add_argument("--only_2D", action="store_true")
    parser.add_argument("--no_hybrid", dest="hybrid", action="store_false")
    parser.set_defaults(hybrid=True)
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    hyperparams = get_default_hyperparams(args)
        
    if args.modality == "PET":
        data = get_pet_data(args.data_path)
        get_am_smooth = lambda: get_pet_am(
            gpu=True,
            gauss_fwhm=args.gauss_fwhm
        )
    else:
        data = get_spect_data(args.data_path)
        get_am_smooth = lambda: get_spect_am(
            data,
            args.spect_res,
            True,
            args.gauss_fwhm,
        )

    if args.method == "kosmaposl":
        run_kosmaposl(args, data, hyperparams, get_am_smooth)
    elif args.method == "ista":
        run_ista(args, data, hyperparams, get_am_smooth)
    else:
        raise ValueError("Unknown method specified.")


if __name__ == "__main__":
    main()
