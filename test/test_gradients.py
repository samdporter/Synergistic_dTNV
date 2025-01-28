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


AcquisitionData.set_storage_scheme('memory')

import matplotlib.pyplot as plt

# CIL imports
from cil.framework import BlockDataContainer
from cil.optimisation.operators import BlockOperator, ZeroOperator, GradientOperator

parser = argparse.ArgumentParser(description='BSREM')

parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/src', help='source path')

args = parser.parse_args()

# Imports from my stuff and SIRF contribs
sys.path.insert(0, args.source_path)
from structural_priors.VTV import WeightedVectorialTotalVariation
from utilities.data import get_pet_data, get_spect_data
from utilities.functions import get_pet_am, get_spect_am
from utilities.nifty import NiftyResampleOperator
from structural_priors.Gradients import Gradient, DirectionalGradient, Jacobian

# Monkey patching
BlockOperator.forward = lambda self, x: self.direct(x)
BlockOperator.backward = lambda self, x: self.adjoint(x)

BlockDataContainer.get_uniform_copy = lambda self, n: BlockDataContainer(*[x.clone().fill(n) for x in self.containers])
BlockDataContainer.max = lambda self: max(d.max() for d in self.containers)

ZeroOperator.backward = lambda self, x: self.adjoint(x)
ZeroOperator.forward = lambda self, x: self.direct(x)

def assert_gradients_close(x, gradients, tol=1e-6):

    for gradient in gradients:
        try:
            assert np.allclose(gradients[0].direct(x), gradient.direct(x), atol=tol)
        except: print(f"Error in direct method: sum of differences is {np.sum(gradients[0].direct(x) - gradient.direct(x))}")

def assert_gradients_close_cil(x, image, grad, grad_cil, tol=1e-6):

    try:
        assert np.allclose(grad.adjoint(grad.direct(x)), grad_cil.adjoint(grad_cil.direct(image)).as_array(), atol=tol)
    except:
        # print the locations of the differences that are greater than the tolerance
        diff = grad.adjoint(grad.direct(x)) - grad_cil.adjoint(grad_cil.direct(image)).as_array()
        print(np.where(np.abs(diff) > tol))
        # print max difference
        print(np.max(np.abs(diff)))

def test_adjointness(x, gradient, tol=1e-6):

    g_x = gradient.direct(x)

    y = np.random.random(g_x.shape)

    g_y = gradient.adjoint(y)

    inner_direct = np.sum(g_x * y)
    inner_adjoint = np.sum(x * g_y)

    # Check if the inner products are close
    return np.allclose(inner_direct, inner_adjoint, atol=tol)

def test_fwd_bwd_time(x, gradient, repeats=10):

    start_time = time.time()
    for _ in range(repeats):
        gradient.direct(x)
    direct_time = (time.time() - start_time) / repeats

    start_time = time.time()
    for _ in range(repeats):
        gradient.adjoint(gradient.direct(x))
    adjoint_time = (time.time() - start_time) / repeats

    return direct_time, adjoint_time

def main(args):

    voxel_sizes = [[np.random.random() for _ in range(3)] for _ in range(5)]

    arrays = [np.random.random((30,30,30)).astype(np.float32) for _ in range(5)]
    array_ts = [np.random.random((30,30,30,2)).astype(np.float32) for _ in range(5)]
    images = [ImageData() for _ in range(5)]
    
    for i, image in enumerate(images):
        image.initialise((30,30,30), vsize=tuple(voxel_sizes[i]))
        image.fill(arrays[i])

    for bnd_cond in ["Neumann", "Periodic"]:

        grad_cpu = [Gradient(voxel_sizes[i], gpu=False, bnd_cond = bnd_cond) for i in range(5)]
        grad_gpu = [Gradient(voxel_sizes[i], gpu=True, numpy_out=True, bnd_cond = bnd_cond) for i in range(5)]
        grad_cil = [GradientOperator(images[i], bnd_cond = bnd_cond) for i in range(5)]

        anatomical = np.random.random((30,30,30))
        
        dgrad_cpu = [DirectionalGradient(anatomical, voxel_sizes[i], gpu=False, bnd_cond = bnd_cond) for i in range(5)]
        dgrad_gpu = [DirectionalGradient(anatomical, voxel_sizes[i], gpu=True, numpy_out=True, bnd_cond = bnd_cond) for i in range(5)]

        jacobian_cpu = [Jacobian(voxel_sizes[i], gpu=False, bnd_cond = bnd_cond) for i in range(5)]
        jacobian_gpu = [Jacobian(voxel_sizes[i], gpu=True, numpy_out=True, bnd_cond = bnd_cond) for i in range(5)]

        djacobian_cpu = [Jacobian(voxel_sizes[i], gpu=False, bnd_cond = bnd_cond, anatomical=anatomical) for i in range(5)]
        djacobian_gpu = [Jacobian(voxel_sizes[i], gpu=True, numpy_out=True, bnd_cond = bnd_cond, anatomical=anatomical) for i in range(5)]

        for i in range(5):

            print(f"Testing gradients are close")
            assert_gradients_close(arrays[i], [grad_cpu[i], grad_gpu[i]])
            assert_gradients_close(arrays[i], [dgrad_cpu[i], dgrad_gpu[i]])
            assert_gradients_close(arrays[i], [jacobian_cpu[i], jacobian_gpu[i]])

            print(f"Testing CIL gradients are close")
            assert_gradients_close_cil(arrays[i], images[i], grad_gpu[i], grad_cil[i])
            assert_gradients_close_cil(arrays[i], images[i], grad_cpu[i], grad_cil[i])

            print(f"Testing adjointness of gradients")
            assert test_adjointness(arrays[i], grad_cpu[i])
            assert test_adjointness(arrays[i], grad_gpu[i])

            print(f"Testing adjointness of directional gradients")
            assert test_adjointness(arrays[i], dgrad_cpu[i])
            assert test_adjointness(arrays[i], dgrad_gpu[i])

            print(f"Testing adjointness of jacobians")
            assert test_adjointness(array_ts[i], jacobian_cpu[i])
            assert test_adjointness(array_ts[i], jacobian_gpu[i])

            print(f"Testing adjointness of directional jacobians")
            assert test_adjointness(array_ts[i], djacobian_cpu[i])
            assert test_adjointness(array_ts[i], djacobian_gpu[i])

            print(f"Testing forward and backward times")
            print(f"CPU gradient: {test_fwd_bwd_time(arrays[i], grad_cpu[i])}")
            print(f"GPU gradient: {test_fwd_bwd_time(arrays[i], grad_gpu[i])}")
            print(f"CIL gradient: {test_fwd_bwd_time(images[i], grad_cil[i])}")

            print(f"CPU directional gradient: {test_fwd_bwd_time(arrays[i], dgrad_cpu[i])}")
            print(f"GPU directional gradient: {test_fwd_bwd_time(arrays[i], dgrad_gpu[i])}")

            print(f"CPU jacobian: {test_fwd_bwd_time(array_ts[i], jacobian_cpu[i])}")
            print(f"GPU jacobian: {test_fwd_bwd_time(array_ts[i], jacobian_gpu[i])}")
    
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
