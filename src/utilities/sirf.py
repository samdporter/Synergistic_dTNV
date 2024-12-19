from cil.optimisation.operators import IdentityOperator, ZeroOperator, BlockOperator
from cil.optimisation.functions import KullbackLeibler, OperatorCompositionFunction
from cil.framework import BlockDataContainer
import numpy as np
from sirf.contrib.partitioner import partitioner
from sirf.STIR import TruncateToCylinderProcessor, SeparableGaussianImageFilter

def get_block_objective(desired_image, other_image, obj_fun, order = 0):

    """ Returns a block CIL objective function for the given SIRF objective function """

    # Set up zero operators
    o2d_zero = ZeroOperator(other_image, desired_image)
    d2d_id = IdentityOperator(desired_image)

    if order == 0:
        return OperatorCompositionFunction(obj_fun, BlockOperator(d2d_id, o2d_zero, shape = (1,2)))
    elif order == 1:
        return OperatorCompositionFunction(obj_fun, BlockOperator(o2d_zero, d2d_id, shape = (1,2)))
    else:
        raise ValueError("Order must be 0 or 1")

def set_up_partitioned_objectives(pet_data, spect_data, pet_obj_funs, spect_obj_funs):

    """ Returns a CIL SumFunction for the partitioned objective functions """
    
    for obj_fun in pet_obj_funs:
        obj_fun.set_up(pet_data['initial_image'])

    for obj_fun in spect_obj_funs:
        obj_fun.set_up(spect_data['initial_image'])
    
    return pet_obj_funs, spect_obj_funs

def set_up_kl_objectives(pet_data, spect_data, pet_datas, pet_norms, spect_datas, pet_ams, spect_ams):

    """ Returns a CIL SumFunction using KL objective functions for the PET and SPECT data and acq models """
    
    for d, am in zip(pet_datas, pet_ams):
        am.set_up(d, pet_data['initial_image'])

    for d, am in zip(spect_datas, spect_ams):
        am.set_up(d, spect_data['initial_image'])

    pet_ads = [am.get_additive_term()*norm for am, norm in zip(pet_ams, pet_norms)]
    spect_ads = [am.get_additive_term() for am in spect_ams] # Do I somehow need to apply the normalisation here?

    pet_ams = [am.get_linear_acquisition_model() for am in pet_ams]
    spect_ams = [am.get_linear_acquisition_model() for am in spect_ams]

    pet_obj_funs = [OperatorCompositionFunction(KullbackLeibler(data, eta=add+add.max()/1e3), am) for data, add, am in zip(pet_datas, pet_ads, pet_ams)]
    spect_obj_funs = [OperatorCompositionFunction(KullbackLeibler(data, eta=add+add.max()/1e3), am) for data, add, am in zip(spect_datas, spect_ads, spect_ams)]

    return pet_obj_funs, spect_obj_funs

def get_s_inv_from_obj(obj_funs, initial_estimates):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimates.get_uniform_copy(0)
    for i, el in enumerate(s_inv.containers):
        for obj_fun in obj_funs[i]:
            tmp = obj_fun.get_subset_sensitivity(0)
            tmp = tmp.maximum(0)
            el += tmp
        el_arr = el.as_array()
        el_arr = np.reciprocal(el_arr, where=el_arr!=0)
        el.fill(np.nan_to_num(el_arr))
    return s_inv

def get_s_inv_from_am(ams, initial_estimates):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimates*0
    for i, el in enumerate(s_inv.containers):
        for am in ams[i]:
            one = am.forward(initial_estimates[i]).get_uniform_copy(1)
            tmp = am.backward(one)
            tmp = tmp.maximum(0)
            el += tmp
        el_arr = el.as_array()
        el_arr = np.reciprocal(el_arr, where=el_arr!=0)
        el.fill(np.nan_to_num(el_arr))
    return s_inv

def compute_inv_hessian_diagonals(bdc, obj_funs_list):

    outputs = []

    for image, obj_funs in zip(bdc.containers, obj_funs_list):
        # Initialize uniform copies
        ones_image = image.get_uniform_copy(1)
        hessian_diag = ones_image.get_uniform_copy(0)

        # Accumulate Hessian contributions
        for obj_fun in obj_funs:
            hessian_diag += obj_fun.function.multiply_with_Hessian(
                image, ones_image
            )

        # Take absolute values and write the result
        hessian_diag = hessian_diag.abs()

        hessian_diag_arr = hessian_diag.as_array()
        hessian_diag.fill(np.reciprocal(hessian_diag_arr, where=hessian_diag_arr!=0))
        
        outputs.append(hessian_diag)

    return BlockDataContainer(*outputs)

def get_subset_data(data, num_subsets, stagger = "staggered"):
        
    views=data.dimensions()[2]
    indices = list(range(views))
    partitions_idxs = partitioner.partition_indices(num_subsets, indices, stagger = stagger)
    datas = [data.get_subset(partitions_idxs[i]) for i in range(num_subsets[0])]

    return datas

def get_filters():
    cyl, gauss = TruncateToCylinderProcessor(), SeparableGaussianImageFilter()
    cyl.set_strictly_less_than_radius(True)
    gauss.set_fwhms((7,7,7))
    return cyl, gauss
