#%% imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# SIRF imports
from sirf.STIR import (ImageData, AcquisitionData,MessageRedirector,
                       TruncateToCylinderProcessor, SeparableGaussianImageFilter,)

from sirf.contrib.partitioner import partitioner
AcquisitionData.set_storage_scheme('memory')

# CIL imports
from cil.optimisation.operators import IdentityOperator
from cil.optimisation.functions import (KullbackLeibler, SumFunction,
                                        OperatorCompositionFunction,)


#%% argparse
parser = argparse.ArgumentParser(description='BSREM')

parser.add_argument('--num_subsets', type=str, default="1,1", help='number of subsets')

#/home/storage/copied_data/data/phantom_data/for_cluster, /home/sam/working/OSEM/simple_data
parser.add_argument('--data_path', type=str, default="/home/storage/copied_data/data/phantom_data/for_cluster", help='data path')
parser.add_argument('--output_path', type=str, default="/home/sam/working/BSREM_PSMR_MIC_2024/results/test", help='output path')
parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/src', help='source path')
parser.add_argument('--working_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/tmp', help='working path')

# set numpy seed - None if not set
parser.add_argument('--seed', type=int, default=None, help='numpy seed')
parser.add_argument('--no_gpu', action='store_true', help='Disables GPU')
parser.add_argument('--keep_all_views_in_cache', action='store_false', default=True, help='Do not keep all views in cache')

args, unknown = parser.parse_known_args()

#%% Imports from my stuff
sys.path.insert(0, args.source_path)
from utilities.data import get_pet_data
from utilities.functions import get_pet_am

#%% Monkey patching
def geometry(self):
    return self.allocate(0)

#ImageData.allocate = allocate
ImageData.geometry = property(geometry) 

#%% Change to working directory - this is where the tmp_ files will be saved
os.chdir(args.working_path)

def get_data(args):
    
    """ Get the data for the tests """
    
    # convert num_subsets to list of ints
    num_subsets = [int(i) for i in args.num_subsets.split(",")]

    pet_data  = get_pet_data(args.data_path)

    initial_estimate = pet_data["initial_image"].clone()

    return pet_data, initial_estimate, num_subsets

def set_up_part_objectives(pet_data, pet_pll_obj_funs):

    """ Sets up the part objective funcs """
    
    for obj_fun in pet_pll_obj_funs:
        obj_fun.set_up(pet_data['initial_image'])
    
    return pet_pll_obj_funs

def set_up_kl_objectives(pet_data, pet_datas, pet_ams):

    """ Sets up the KL objective funcs """
    
    for d, am in zip(pet_datas, pet_ams):
        am.set_up(d, pet_data['initial_image'])

    pet_ads = [am.get_additive_term()*pet_data['normalisation'] for am in pet_ams]

    pet_ams = [am.get_linear_acquisition_model() for am in pet_ams]

    pet_pll_obj_funs = [OperatorCompositionFunction(KullbackLeibler(data, eta=add), am) for data, add, am in zip(pet_datas, pet_ads, pet_ams)]
   
    return pet_pll_obj_funs

#%% Main
def main(args):

    print("Getting data...")

    pet_data, initial_estimate, num_subsets = get_data(args)

    print("Partitioning data...")

    pet_datas, pet_ams, pet_pll_obj_funs = partitioner.data_partition(pet_data['acquisition_data'], pet_data['additive'], 
                                                        pet_data['normalisation'], num_batches=num_subsets[0], mode = "staggered",
                                                        create_acq_model=lambda: get_pet_am(not args.no_gpu, gauss_fwhm=(4.5,7.5,7.5),truncate_cylinder=True))
    
    print("Setting up objectives...")

    pet_pll_obj_funs = set_up_part_objectives(pet_data, pet_pll_obj_funs)
    pet_kl_obj_funs = set_up_kl_objectives(pet_data, pet_datas, pet_ams)

    pll_obj = SumFunction(*pet_pll_obj_funs)
    kl_obj = -SumFunction(*pet_kl_obj_funs) # negative sign because we want to maximise
    
    def PLL_call(x, y, am):
        lam = am.forward(x) + 1e-6
        return (y * lam.log()-lam).sum()
    
    def PLL_grad_call(x, y, am):
        lam = am.forward(x) + 1e-6
        one = y.get_uniform_copy(1)
        return am.backward(y / lam - one)

    print("Calculating objective function and gradients...")

    pll_obj_val = pll_obj(initial_estimate)
    pll_obj_grad = pll_obj.gradient(initial_estimate)

    kl_obj_val = kl_obj(initial_estimate)
    kl_obj_grad = kl_obj.gradient(initial_estimate)
    
    pll_with_sirf_am_val = sum([PLL_call(initial_estimate, data, am) for data, am in zip(pet_datas, pet_ams)])
    for i, (data, am) in enumerate(zip(pet_datas, pet_ams)):
        if i == 0:
            pll_with_sirf_am_grad = PLL_grad_call(initial_estimate, data, am)
        else:
            pll_with_sirf_am_grad += PLL_grad_call(initial_estimate, data, am)

    f_values = {"SUM of part funcs": pll_obj_val, "SUM of KL funcs": kl_obj_val, "SUM of part funcs with SIRF AM": pll_with_sirf_am_val}
    gradients = {"SUM of part funcs": pll_obj_grad, "SUM of KL funcs": kl_obj_grad, "SUM of part funcs with SIRF AM": pll_with_sirf_am_grad}

    return f_values, gradients

#%% Test
def test(f_values, gradients):

    # Check if all function values are close
    if all(np.allclose(f_values["SUM of part funcs"], val, rtol=1e-5) for val in f_values.values()):
        print("Function values match")
    else:
        mismatch_message = ", ".join(f"{key}: {val}" for key, val in f_values.items())
        print(f"Function values do not match: {mismatch_message}")

    print("Testing gradients...")

    try:
        # Check if all gradients are close
        if all(np.allclose(gradients["SUM of part funcs"].sum(), grad.sum(), rtol=1e-5) for grad in gradients.values()):
            print("Gradients match")
        else:
            mismatch_message = ", ".join(f"{key}: {grad.sum()}" for key, grad in gradients.items())
            print(f"Gradients do not match: {mismatch_message}")
    except Exception as e:
        print(f"Could not compare gradients due to error: {e}")
        mismatch_message = ", ".join(f"{key}: {grad.sum()}" for key, grad in gradients.items())
        print(f"Partial gradients: {mismatch_message}")

    fig, ax = plt.subplots(1, len(gradients), figsize=(5*len(gradients), 3.5))
    ims = []
    for i, grad in enumerate(gradients.values()):
        ims.append(ax[i].imshow(grad.as_array().sum(axis=0)))
        ax[i].set_title(f"Gradient of {list(gradients.keys())[i]}")
        ax[i].axis("off")
        plt.colorbar(ims[i], ax=ax[i])
    plt.tight_layout()
    plt.savefig("gradients.png")

#%% Run
if __name__ == "__main__":

    msg = MessageRedirector()

    print("Running tests...")

    f_values, gradients = main(args)

    test(f_values, gradients)

    print("Tests complete")