#%% imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# SIRF imports
from sirf.STIR import (ImageData, AcquisitionData,MessageRedirector,
                       TruncateToCylinderProcessor, SeparableGaussianImageFilter,
                       make_Poisson_loglikelihood, AcquisitionSensitivityModel,
                       AcquisitionModel)

from sirf.contrib.partitioner import partitioner
AcquisitionData.set_storage_scheme('memory')

# CIL imports
from cil.framework import BlockDataContainer
from cil.optimisation.operators import BlockOperator, ZeroOperator, IdentityOperator
from cil.optimisation.functions import (KullbackLeibler,
                                        OperatorCompositionFunction,
                                        SumFunction,  BlockFunction)


#%% argparse
parser = argparse.ArgumentParser(description='BSREM')

parser.add_argument('--num_subsets', type=str, default="8,8", help='number of subsets')

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
from utilities.data import get_pet_data, get_spect_data
from utilities.functions import get_pet_am, get_spect_am
from utilities.sirf import get_s_inv_from_am, get_s_inv_from_obj

#%% Monkey patching
def geometry(self):
    return self.allocate(0)

ImageData.geometry = property(geometry) 

BlockDataContainer.get_uniform_copy = lambda self, n: BlockDataContainer(*[x.clone().fill(n) for x in self.containers])
BlockDataContainer.max = lambda self: max(d.max() for d in self.containers)

#%% Change to working directory - this is where the tmp_ files will be saved
os.chdir(args.working_path)

#%% Functions
def get_filters():
    cyl, gauss = TruncateToCylinderProcessor(), SeparableGaussianImageFilter()
    cyl.set_strictly_less_than_radius(True)
    gauss.set_fwhms((7,7,7))
    return cyl, gauss

def get_data(args):
    
    """ Get the data for the tests """
    
    # convert num_subsets to list of ints
    num_subsets = [int(i) for i in args.num_subsets.split(",")]

    cyl, gauss, = get_filters()
    ct = ImageData(os.path.join(args.data_path, "CT/ct_zoomed_smallFOV.hv"))
    # normalise the CT image
    ct+=(-ct).max()
    ct/=ct.max()    

    pet_data  = get_pet_data(args.data_path)
    spect_data  = get_spect_data(args.data_path)
    
    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    initial_estimates = BlockDataContainer(pet_data["initial_image"], spect_data["initial_image"])

    return pet_data, spect_data, initial_estimates, num_subsets

def get_sirf_objective(args, pet_data, spect_data, initial_estimates):

    """ Returns a SIRF objective function for the PET and SPECT data """

    def get_obj(data, get_am):
        
        am = get_am()
        sensitivity_factors = AcquisitionSensitivityModel(data['normalisation'])
        sensitivity_factors.set_up(data['acquisition_data'])
        am = get_am()
        am.set_acquisition_sensitivity(sensitivity_factors)
        am.set_additive_term(data['additive'])
        
        obj = make_Poisson_loglikelihood(data['acquisition_data'], acq_model=am)
        obj.set_up(data['initial_image'])

        return obj
    
    spect_data['normalisation'] = spect_data['acquisition_data'].get_uniform_copy(1)
    pet_obj = get_obj(pet_data, lambda: get_pet_am(not args.no_gpu, gauss_fwhm=(4.5,7.5,7.5), truncate_cylinder=True))
    spect_obj = get_obj(spect_data, lambda: get_spect_am(spect_data, None, keep_all_views_in_cache=args.keep_all_views_in_cache))

    return pet_obj, spect_obj

def get_block_objective(desired_image, other_image, obj_fun, order = 0):

    """ Returns a block CIL objective function for the given SIRF objective function """

    o2d_zero = ZeroOperator(other_image, desired_image)
    d2d_id = IdentityOperator(desired_image)

    if order == 0:
        return OperatorCompositionFunction(obj_fun, BlockOperator(d2d_id, o2d_zero, shape = (1,2)))
    elif order == 1:
        return OperatorCompositionFunction(obj_fun, BlockOperator(o2d_zero, d2d_id, shape = (1,2)))
    else:
        raise ValueError("Order must be 0 or 1")

def set_up_partitioned_objectives(pet_data, spect_data, pet_pll_obj_funs, spect_pll_obj_funs):

    """ Returns a CIL SumFunction for the partitioned objective functions """
    
    for obj_fun in pet_pll_obj_funs:
        obj_fun.set_up(pet_data['initial_image'])

    for obj_fun in spect_pll_obj_funs:
        obj_fun.set_up(spect_data['initial_image'])
    
    return pet_pll_obj_funs, spect_pll_obj_funs

def set_up_kl_objectives(pet_data, spect_data, pet_datas, spect_datas, pet_ams, spect_ams):

    """ Returns a CIL SumFunction using KL objective functions for the PET and SPECT data and acq models """
    
    for d, am in zip(pet_datas, pet_ams):
        am.set_up(d, pet_data['initial_image'])

    for d, am in zip(spect_datas, spect_ams):
        am.set_up(d, spect_data['initial_image'])

    pet_norms = get_subset_data(pet_data['normalisation'], num_subsets=len(pet_datas), stagger = "staggered")

    pet_ads = [am.get_additive_term()*norm for am, norm in zip(pet_ams, pet_norms)]
    spect_ads = [am.get_additive_term() for am in spect_ams]

    pet_ams = [am.get_linear_acquisition_model() for am in pet_ams]
    spect_ams = [am.get_linear_acquisition_model() for am in spect_ams]

    pet_pll_obj_funs = [OperatorCompositionFunction(KullbackLeibler(data, eta=add), am) for data, add, am in zip(pet_datas, pet_ads, pet_ams)]
    spect_pll_obj_funs = [OperatorCompositionFunction(KullbackLeibler(data, eta=add), am) for data, add, am in zip(spect_datas, spect_ads, spect_ams)]

    return pet_pll_obj_funs, spect_pll_obj_funs

def get_subset_data(data, num_subsets, stagger = "staggered"):
        
    views=data.dimensions()[2]
    indices = list(range(views))
    partitions_idxs = partitioner.partition_indices(num_subsets, indices, stagger = stagger)
    datas = [data.get_subset(partitions_idxs[i]) for i in range(num_subsets)]

    return datas

#%% Main
def main(args):

    # Get the data

    print("Setting up data...")

    pet_data, spect_data, initial_estimates, num_subsets = get_data(args)



    # Pure SIRF objective function

    pet_obj, spect_obj = get_sirf_objective(args, pet_data, spect_data, initial_estimates)

    print("Calculating objective function using SIRF objective functions...")

    manual_full_obj_val = pet_obj(pet_data["initial_image"]) + spect_obj(spect_data["initial_image"])

    manual_obj_grad = BlockDataContainer(pet_obj.gradient(pet_data["initial_image"]),
                                                spect_obj.gradient(spect_data["initial_image"]))
    
    print("Calculating inverse sensitivities using SIRF objective functions...")
    pet_obj_sens = pet_obj.get_subset_sensitivity(0)
    spect_obj_sens = spect_obj.get_subset_sensitivity(0)

    pet_obj_sens_arr = np.reciprocal(pet_obj_sens.as_array(), where=pet_obj_sens.as_array()!=0)
    spect_obj_sens_arr = np.reciprocal(spect_obj_sens.as_array(), where=spect_obj_sens.as_array()!=0)

    pet_obj_sens_arr = np.nan_to_num(pet_obj_sens_arr)
    spect_obj_sens_arr = np.nan_to_num(spect_obj_sens_arr)

    pet_obj_sens.fill(pet_obj_sens_arr)
    spect_obj_sens.fill(spect_obj_sens_arr)

    s_inv_obj_full = BlockDataContainer(pet_obj_sens, spect_obj_sens)

    del pet_obj, spect_obj
    
    # Partitioned objective functions

    print("Setting up partitioned objective functions...")

    pet_datas, pet_ams, pet_pll_obj_funs = partitioner.data_partition(pet_data['acquisition_data'], pet_data['additive'], 
                                                        pet_data['normalisation'], num_batches=num_subsets[0], mode = "staggered",
                                                        create_acq_model=lambda: get_pet_am(not args.no_gpu, gauss_fwhm=(4.5,7.5,7.5),truncate_cylinder=True))
    
    # Unfortately need to redo this to avoid SPECT bug
    # defining here so we know we are using the same partitioning
    def get_partitioned_spect():
        return partitioner.data_partition(spect_data['acquisition_data'], spect_data['additive'],
                                        spect_data['acquisition_data'].get_uniform_copy(1), num_batches=num_subsets[1], mode = "staggered",
                                        create_acq_model=lambda: get_spect_am(spect_data, None, keep_all_views_in_cache=args.keep_all_views_in_cache,
                                                                              gauss_fwhm=(6.2,6.2,6.2)))

    _ ,_ , spect_pll_obj_funs = get_partitioned_spect()

    print("Setting up partitioned objective functions...")
    pet_pll_obj_funs, spect_pll_obj_funs = set_up_partitioned_objectives(pet_data, spect_data, pet_pll_obj_funs, spect_pll_obj_funs)

    print("Calculating inverse sensitivities using SIRF objective functions...")
    s_inv_obj = get_s_inv_from_obj([pet_pll_obj_funs, spect_pll_obj_funs], initial_estimates)

    # This is where stuff goes wrong
    pet_pll_obj_funs = [get_block_objective(pet_data['initial_image'], spect_data['initial_image'], obj_fun, order=0) for obj_fun in pet_pll_obj_funs]
    spect_pll_obj_funs = [get_block_objective(spect_data['initial_image'], pet_data['initial_image'], obj_fun, order=1) for obj_fun in spect_pll_obj_funs]

    all_plls = pet_pll_obj_funs + spect_pll_obj_funs

    obj_sum = SumFunction(*all_plls)
    
    print("Calculating objective function using CIL SumFunction and SIRF PLL functions...")

    partitioned_obj_val = obj_sum(initial_estimates)
    partitioned_obj_grad = obj_sum.gradient(initial_estimates)

    del obj_sum, pet_pll_obj_funs, spect_pll_obj_funs

    spect_datas, spect_ams, _ = get_partitioned_spect()

    print("Setting up KL objective functions...")
    pet_kl_obj_funs, spect_kl_obj_funs = set_up_kl_objectives(pet_data, spect_data, pet_datas, spect_datas, pet_ams, spect_ams)

    print("Calculating inverse sensitivities using SIRF AcquisitionModels...")
    s_inv_am = get_s_inv_from_am([pet_ams, spect_ams], initial_estimates)

    pet_kl_obj_funs = [-get_block_objective(pet_data['initial_image'], spect_data['initial_image'], obj_fun, order=0) for obj_fun in pet_kl_obj_funs]
    spect_kl_obj_funs = [-get_block_objective(spect_data['initial_image'], pet_data['initial_image'], obj_fun, order=1) for obj_fun in spect_kl_obj_funs]

    all_kls = pet_kl_obj_funs + spect_kl_obj_funs

    kl_obj = SumFunction(*all_kls)

    print("Calculating objective function using CIL SumFunction and KL functions...")

    kl_obj_val = kl_obj(initial_estimates)
    kl_obj_grad = kl_obj.gradient(initial_estimates)

    del kl_obj, pet_ams, spect_ams, pet_datas, spect_datas

    # Full objective function

    print("Calculating objective function using FULL CIL BlockFunction with SIRF PLL functions...")

    _, _, pet_obj_fun = partitioner.data_partition(pet_data['acquisition_data'], pet_data['additive'], pet_data['normalisation'], 
                                                        num_batches=1, mode = "staggered",
                                                        create_acq_model=lambda: get_pet_am(not args.no_gpu, gauss_fwhm=(4.5,7.5,7.5), truncate_cylinder=True))
        
    _, _, spect_obj_fun = partitioner.data_partition(spect_data['acquisition_data'], spect_data['additive'], spect_data['acquisition_data'].get_uniform_copy(1), 
                                                        num_batches=1, mode = "staggered",
                                                        create_acq_model=lambda: get_spect_am(spect_data, None, keep_all_views_in_cache=args.keep_all_views_in_cache))

    pet_obj_fun[0].set_up(initial_estimates[0])
    spect_obj_fun[0].set_up(initial_estimates[1])

    full_obj = BlockFunction(pet_obj_fun[0], spect_obj_fun[0])

    full_obj_val = full_obj(initial_estimates)
    full_obj_grad = full_obj.gradient(initial_estimates)

    del pet_obj_fun, spect_obj_fun, _

    # Calculate function values
    f_values = {
        "SUM of partitioned functions": partitioned_obj_val,
        "FULL function": full_obj_val,
        "MANUAL SIRF using obj fun": manual_full_obj_val,
        "KL with SIRF am additional Block Op": kl_obj_val
    }

    # Compute gradients
    gradients = {
        "SUM of partitioned functions": partitioned_obj_grad,
        "FULL function": full_obj_grad,
        "MANUAL SIRF using obj fun": manual_obj_grad,
        "KL with SIRF am additional Block Op": kl_obj_grad
    }

    # inverse sensitivities
    s_invs = {
        "SIRF obj fun": s_inv_obj,
        "KL obj fun": s_inv_am,
        "FULL obj fun": s_inv_obj_full
    }

    return f_values, gradients, s_invs

#%% Test
def test(f_values, gradients, s_invs):

    # Check if all function values are close
    if all(np.allclose(f_values["SUM of partitioned functions"], val, rtol=1e-5) for val in f_values.values()):
        print("Function values match")
    else:
        mismatch_message = ", ".join(f"{key}: {val}" for key, val in f_values.items())
        print(f"Function values do not match: {mismatch_message}")

    print("Testing gradients...")

    try:
        # Check if all gradients are close
        if all(np.allclose(gradients["SUM of partitioned functions"].sum(), grad.sum(), rtol=1e-5) for grad in gradients.values()):
            print("Gradients match")
        else:
            mismatch_message = ", ".join(f"{key}: {grad.sum()}" for key, grad in gradients.items())
            print(f"Gradients do not match: {mismatch_message}")
    except Exception as e:
        print(f"Could not compare gradients due to error: {e}")
        mismatch_message = ", ".join(f"{key}: {grad.sum()}" for key, grad in gradients.items())
        print(f"Partial gradients: {mismatch_message}")

    fig, ax = plt.subplots(2, len(gradients), figsize=(5*len(gradients), 6))
    ims = []
    for i, grad in enumerate(gradients.values()):
        ims.append(ax[0, i].imshow(grad[0].as_array()[35]))
        ims.append(ax[1, i].imshow(grad[1].as_array()[56]))
        ax[0, i].set_title(f"Gradient {list(gradients.keys())[i]}")
        plt.colorbar(ims[-2], ax=ax[0, i])
        plt.colorbar(ims[-1], ax=ax[1, i])
        ax[0, i].set_xticks([])
        ax[1, i].set_xticks([])
    # add PET / SPECT to 1st column axis onl
    ax[0, 0].set_ylabel("PET")
    ax[1, 0].set_ylabel("SPECT")
    plt.savefig("gradients.png")

    fig, ax = plt.subplots(2, len(s_invs), figsize=(5*len(s_invs), 6))
    ims = []
    for i, s_inv in enumerate(s_invs.values()):
        ims.append(ax[0, i].imshow(s_inv[0].as_array()[35]))
        ims.append(ax[1, i].imshow(s_inv[1].as_array()[56]))
        ax[0, i].set_title(f"Inverse sensitivity {list(s_invs.keys())[i]}")
        plt.colorbar(ims[-2], ax=ax[0, i])
        plt.colorbar(ims[-1], ax=ax[1, i])
        ax[0, i].set_xticks([])
        ax[1, i].set_xticks([])
    # add PET / SPECT to 1st column axis onl
    ax[0, 0].set_ylabel("PET")
    ax[1, 0].set_ylabel("SPECT")
    plt.savefig("sensitivity.png")

#%% Run
if __name__ == "__main__":

    msg = MessageRedirector()

    print("Running tests...")

    f_values, gradients, s_invs = main(args)

    test(f_values, gradients, s_invs)

    print("Tests complete")