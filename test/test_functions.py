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
                       TruncateToCylinderProcessor, SeparableGaussianImageFilter,
                       MessageRedirector,
                       )

from sirf.Reg import AffineTransformation
from sirf.contrib.partitioner import partitioner
AcquisitionData.set_storage_scheme('memory')

# CIL imports
from cil.framework import BlockDataContainer
from cil.optimisation.operators import BlockOperator, ZeroOperator, IdentityOperator
from cil.optimisation.functions import OperatorCompositionFunction

parser = argparse.ArgumentParser(description='BSREM')

parser.add_argument('--alpha', type=float, default=128, help='alpha')
parser.add_argument('--beta', type=float, default=0.05, help='beta')
parser.add_argument('--delta', type=float, default=1e-6, help='delta')
parser.add_argument('--num_subsets', type=str, default="1,1", help='number of subsets')
parser.add_argument('--prior_probability', type=float, default=0.5, help='prior probability')

#/home/storage/copied_data/data/phantom_data/for_cluster, /home/sam/working/OSEM/simple_data
parser.add_argument('--data_path', type=str, default="/home/storage/copied_data/data/phantom_data/for_cluster", help='data path')
parser.add_argument('--output_path', type=str, default="/home/sam/working/BSREM_PSMR_MIC_2024/results/test", help='output path')
parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/src', help='source path')
parser.add_argument('--working_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/tmp', help='working path')

parser.add_argument('--seed', type=int, default=None, help='numpy seed')
parser.add_argument('--gpu', action='store_false', default=True, help='Disables GPU')
parser.add_argument('--keep_all_views_in_cache', action='store_false', default=True, help='Do not keep all views in cache')

args = parser.parse_args()

# Imports from my stuff and SIRF contribs
sys.path.insert(0, args.source_path)
from structural_priors.VTV import WeightedVectorialTotalVariation
from utilities.data import get_pet_data, get_spect_data
from utilities.functions import get_pet_am, get_spect_am
from utilities.nifty import NiftyResampleOperator

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

def test_random_directional_derivative(x, function, epsilon=1, tol=1e-4, repeats=10, log_file="gradient_tests.txt"):
    """
    Test the gradient of a function using random directional derivatives.

    Logs any failed tests to a file instead of raising an assertion error.

    Parameters:
    - x: Input BlockDataContainer.
    - function: The function whose gradient is being tested.
    - epsilon: Step size for finite differences.
    - tol: Tolerance for comparing gradients.
    - repeats: Number of random directional tests to perform.
    - log_file: File to log failed tests.
    """
    import time

    failed_tests = []
    passed_tests = []

    print(f"Testing random directional derivative {repeats} times")

    for i in range(repeats):

        start_time = time.time()

        # Generate random direction
        direction = x.get_uniform_copy(0)
        for el in direction.containers:
            el.fill(np.random.randn(*el.shape))
        direction /= direction.norm()
        
        # Compute gradient and directional derivative
        grad = function.gradient(x)
        grad_direction = sum(g.as_array().ravel() @ d.as_array().ravel() for g, d in zip(grad.containers, direction.containers))
        
        # Compute finite differences
        x_plus = x + epsilon * direction
        x_minus = x - epsilon * direction
        grad_direction_fd = (function(x_plus) - function(x_minus)) / (2 * epsilon)
        
        # Check results
        if np.abs(grad_direction - grad_direction_fd) >= tol:
            failed_tests.append(
                f"Failed Test {i+1}: grad_direction={grad_direction}, "
                f"grad_direction_fd={grad_direction_fd}, difference={np.abs(grad_direction - grad_direction_fd)}"
            )
        else:
            passed_tests.append(
                f"Passed Test {i+1}: grad_direction={grad_direction}, "
                f"grad_direction_fd={grad_direction_fd}, difference={np.abs(grad_direction - grad_direction_fd)}"
            )

        print(f"Test {i+1}/{repeats} took {time.time() - start_time:.2f} seconds")

    # Log results
    with open(log_file, "a") as f:
        if failed_tests:
            f.write("Failed Tests:\n")
            f.write("\n".join(failed_tests) + "\n")
        f.write("Passed Tests:\n")
        f.write("\n".join(passed_tests) + "\n")

def test_function_speed(function, x, repeats=10, log_file="speed_tests.txt"):
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

    print(f"Average call time: {np.mean(call_times):.2f} seconds")
    print(f"Average gradient time: {np.mean(grad_times):.2f} seconds")

    with open(log_file, "a") as f:
        f.write(f"Average call time for function {function.function}: {np.mean(call_times):.2f} seconds\n")
        f.write(f"Average gradient time for function {function.function}: {np.mean(grad_times):.2f} seconds\n")

# Change to working directory - this is where the tmp_ files will be saved
os.chdir(args.working_path)
       
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
    pet2ct = NiftyResampleOperator(ct, pet_data["initial_image"], AffineTransformation(os.path.join(args.data_path, "Registration", "pet_to_ct_smallFOV.txt")))
    zero_pet2ct = ZeroOperator(pet_data["initial_image"], ct)
    
    spect2ct = NiftyResampleOperator(ct, spect_data["initial_image"], AffineTransformation(os.path.join(args.data_path, "Registration", "spect_to_ct_smallFOV.txt")))
    zero_spect2ct = ZeroOperator(spect_data["initial_image"], ct)

    # set up prior
    bo = BlockOperator(pet2ct, zero_spect2ct,
                        zero_pet2ct, spect2ct, 
                        shape = (2,2))
    
    vtv = WeightedVectorialTotalVariation(bo.direct(initial_estimates), [args.alpha, args.beta], args.delta, anatomical=ct, gpu=args.gpu)
    prior = OperatorCompositionFunction(vtv, bo)

    # set up data fidelity functions
    
    _, _, pet_obj_funs = partitioner.data_partition(pet_data['acquisition_data'], pet_data['additive'], pet_data['normalisation'], 
                                                        num_batches=num_subsets[0], initial_image=pet_data["initial_image"], mode = "staggered",
                                                        create_acq_model=lambda: get_pet_am(args.gpu, truncate_cylinder=True))
        
    _, _, spect_obj_funs = partitioner.data_partition(spect_data['acquisition_data'], spect_data['additive'], spect_data['acquisition_data'].get_uniform_copy(1), 
                                                        num_batches=num_subsets[1], initial_image=spect_data["initial_image"], mode = "staggered",
                                                        create_acq_model=lambda: get_spect_am(spect_data, keep_all_views_in_cache=args.keep_all_views_in_cache))

    spect2pet_zero = ZeroOperator(spect_data["initial_image"], pet_data["initial_image"])
    pet2pet_id = IdentityOperator(pet_data["initial_image"])
    pet2spect_zero = ZeroOperator(pet_data["initial_image"], spect_data["initial_image"])
    spect2spect_id = IdentityOperator(spect_data["initial_image"])
    
    pet_obj_funs = [OperatorCompositionFunction(obj_fun, BlockOperator(pet2pet_id, spect2pet_zero, shape = (1,2))) for obj_fun in pet_obj_funs]
    spect_obj_funs = [OperatorCompositionFunction(obj_fun, BlockOperator(pet2spect_zero, spect2spect_id, shape = (1,2))) for obj_fun in spect_obj_funs]
    
    all_funs = [prior] + pet_obj_funs + spect_obj_funs

    scaled_prior = -args.prior_probability/(len(pet_obj_funs + spect_obj_funs)) * prior

    all_funs.append(scaled_prior)

    for i, fun in enumerate(all_funs):
        print(f"Testing random directional derivative for function {i}, {fun.function}")
        test_random_directional_derivative(initial_estimates, fun, epsilon=0.1, tol = 1e-4, repeats=10)

    for i, fun in enumerate(all_funs):
        print(f"Testing speed of function {i}, {fun.function}")
        test_function_speed(fun, initial_estimates, repeats=10)
    
if __name__ == "__main__":

    msg = MessageRedirector()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    main(args)

    profiler.disable()
    
    # Output results to a file
    with open("profile_results.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()  # Remove extraneous path information
        stats.sort_stats("tottime")  # Sort by individual function time
        stats.print_stats()
