#%%
import sys
import os
import numpy as np
import argparse

import cProfile
import pstats

import time
start_time = time.time()

# SIRF imports
from sirf.STIR import (ImageData, AcquisitionData, 
                       MessageRedirector,
                       )

from sirf.Reg import AffineTransformation
from sirf.contrib.partitioner import partitioner
AcquisitionData.set_storage_scheme('memory')

# CIL imports
from cil.framework import BlockDataContainer
from cil.optimisation.operators import BlockOperator, ZeroOperator, GradientOperator
from cil.optimisation.functions import OperatorCompositionFunction, SmoothMixedL21Norm

#%%
parser = argparse.ArgumentParser(description='BSREM')

parser.add_argument('--alpha', type=float, default=128, help='alpha')
parser.add_argument('--beta', type=float, default=0.05, help='beta')
parser.add_argument('--delta', type=float, default=1e-6, help='delta')

#/home/storage/copied_data/data/phantom_data/for_cluster, /home/sam/working/OSEM/simple_data
parser.add_argument('--data_path', type=str, default="/home/storage/copied_data/data/phantom_data/for_cluster", help='data path')
parser.add_argument('--output_path', type=str, default="/home/sam/working/BSREM_PSMR_MIC_2024/results/test", help='output path')
parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/src', help='source path')
parser.add_argument('--working_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/tmp', help='working path')

parser.add_argument('--seed', type=int, default=None, help='numpy seed')
parser.add_argument('--keep_all_views_in_cache', action='store_false', default=True, help='Do not keep all views in cache')

args, unknown = parser.parse_known_args()

#%% Imports from my stuff and SIRF contribs
sys.path.insert(0, args.source_path)
from structural_priors.VTV import WeightedVectorialTotalVariation
from utilities.data import get_pet_data, get_spect_data
from utilities.nifty import NiftyResampleOperator

#%% Monkey patching
BlockDataContainer.get_uniform_copy = lambda self, n: BlockDataContainer(*[x.clone().fill(n) for x in self.containers])
BlockDataContainer.max = lambda self: max(d.max() for d in self.containers)

#%%
def test_function_speed(function, x, repeats=10, log_file="speed_tests.txt", name = None):
    """
    Test the speed of a function.

    Parameters:
    - function: The function to test.
    - x: Input BlockDataContainer.
    - repeats: Number of times to run the function.
    """
    import time

    call_times = []
    grad_times = []

    for i in range(repeats):
        start_time = time.time()
        function(x)
        call_times.append(time.time() - start_time)

        start_time = time.time()
        function.gradient(x)
        grad_times.append(time.time() - start_time)

    print(f"Average call time: {np.mean(call_times):.2f} seconds for function {name}")
    print(f"Average gradient time: {np.mean(grad_times):.2f} seconds for function {name}")

    with open(log_file, "a") as f:
        f.write(f"Average call time for function {function.function}: {np.mean(call_times):.2f} seconds\n")
        f.write(f"Average gradient time for function {function.function}: {np.mean(grad_times):.2f} seconds\n")

def test_values(functions, x, tol=1e-6):
    """
    Test that the values of the functions are the same.

    Parameters:
    - functions: List of functions to test.
    - x: Input BlockDataContainer.
    - tol: Tolerance for the test.
    """
    values = [f(x) for f in functions]

    for i in range(1, len(values)):
        assert np.allclose(values[0], values[i], atol=tol), f"Values of functions {functions[0].function} and {functions[i].function} are not the same."

    print("Values of functions are the same.")
    
def test_gradients(functions, x, tol=1e-6, eps=1e-6):
    """
    Test that the gradients of the functions are the same and close to the finite difference gradients.

    Parameters:
    - functions: List of functions to test.
    - x: Input BlockDataContainer.
    - tol: Tolerance for the test.
    - eps: Epsilon step size for finite difference gradient computation.
    """
    # Compute gradients of all functions at x
    gradients = [f.gradient(x) for f in functions]

    # Check that gradients of all functions are the same
    for i in range(1, len(gradients)):
        if not gradients[0].is_compatible(gradients[i]):
            raise AssertionError(f"Gradients of functions {functions[0].function} and {functions[i].function} have incompatible shapes.")
        for g1, g2 in zip(gradients[0].containers, gradients[i].containers):
            if not np.allclose(g1.as_array(), g2.as_array(), atol=tol):
                print(f"Gradients of functions 0 and {i} are not the same.")
                # print max difference and location
                max_diff = np.max(np.abs(g1.as_array() - g2.as_array()))
                print(f"Max difference: {max_diff}")
                print(f"Location of max difference: {np.unravel_index(np.argmax(np.abs(g1.as_array() - g2.as_array())), g1.shape)}")

#%% Change to working directory - this is where the tmp_ files will be saved
os.chdir(args.working_path)

#%%       
def main(args):
    """
    Main function to perform image reconstruction using BSREM and PSMR algorithms.
    """

    print("Importing data...")
    ct = ImageData(os.path.join(args.data_path, "CT/ct_zoomed_smallFOV.hv"))
    # normalise the CT image
    ct+=(-ct).max()
    ct/=ct.max()    

    pet_data  = get_pet_data(args.data_path)
    spect_data  = get_spect_data(args.data_path)

    initial_estimates = BlockDataContainer(pet_data["initial_image"], spect_data["initial_image"])
    
    print("Setting up operators...")
    pet2ct = NiftyResampleOperator(ct, pet_data["initial_image"], AffineTransformation(os.path.join(args.data_path, "Registration", "pet_to_ct_smallFOV.txt")))
    zero_pet2ct = ZeroOperator(pet_data["initial_image"], ct)
    
    spect2ct = NiftyResampleOperator(ct, spect_data["initial_image"], AffineTransformation(os.path.join(args.data_path, "Registration", "spect_to_ct_smallFOV.txt")))
    zero_spect2ct = ZeroOperator(spect_data["initial_image"], ct)

    # set up prior
    bo = BlockOperator(pet2ct, zero_spect2ct,
                        zero_pet2ct, spect2ct, 
                        shape = (2,2))
    
    print("Setting up functions...")
    gpu_vtv = WeightedVectorialTotalVariation(bo.direct(initial_estimates), [args.alpha, args.beta], args.delta, anatomical=ct, gpu=True)
    gpu_vtv_unstable = WeightedVectorialTotalVariation(bo.direct(initial_estimates), [args.alpha, args.beta], args.delta, anatomical=ct, gpu=True, stable=False)
    cpu_vtv = WeightedVectorialTotalVariation(bo.direct(initial_estimates), [args.alpha, args.beta], args.delta, anatomical=ct, gpu=False)
    gpu_prior = OperatorCompositionFunction(gpu_vtv, bo)
    gpu_prior_unstable = OperatorCompositionFunction(gpu_vtv_unstable, bo)
    cpu_prior = OperatorCompositionFunction(cpu_vtv, bo)

    print("Testing function values...")
    test_function_speed(cpu_prior, initial_estimates, name="cpu_prior")
    test_function_speed(gpu_prior, initial_estimates, name="gpu_prior")
    test_function_speed(gpu_prior_unstable, initial_estimates, name="gpu_prior_unstable")

    test_values([gpu_prior, gpu_prior_unstable, cpu_prior], initial_estimates)

    print("Testing function gradients...")
    test_gradients([gpu_prior, gpu_prior_unstable, cpu_prior], initial_estimates)

#%%
if __name__ == "__main__":

    msg = MessageRedirector()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    main(args)

    profiler.disable()
    
    # Output results to a file
    with open("profile_results_vtv.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()  # Remove extraneous path information
        stats.sort_stats("cumulative")  # Sort by cumulative time
        stats.print_stats()
