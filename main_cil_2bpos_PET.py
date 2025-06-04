#!/usr/bin/env python3
#main_cil_2bpos_PET.py
import sys
import os
import numpy as np
import argparse
import pandas as pd
import logging
from types import MethodType
from typing import List, Any

# SIRF
from sirf.STIR import ImageData, AcquisitionData, MessageRedirector
from sirf.contrib.partitioner import partitioner
AcquisitionData.set_storage_scheme("memory")

# CIL
from cil.framework import BlockDataContainer
from cil.optimisation.operators import (
    BlockOperator, ZeroOperator, IdentityOperator,
    CompositionOperator, GradientOperator
)
from cil.optimisation.functions import (
    SVRGFunction, SumFunction, SmoothMixedL21Norm,
    OperatorCompositionFunction
)
from cil.optimisation.algorithms import ISTA
from cil.optimisation.utilities import Sampler

from main_functions import attach_prior_hessian

# Monkey-patch ISTA and L21 Hessian
def update_step(self):
    self.f.gradient(self.x_old, out=self.gradient_update)
    η = self.step_size_rule.get_step_size(self)
    if self.preconditioner:
        self.x_old.sapyb(1,
                         self.preconditioner.apply(self, self.gradient_update),
                         -η, out=self.x_old)
    else:
        self.x_old.sapyb(1, self.gradient_update, -η, out=self.x_old)
    self.g.proximal(self.x_old, η, out=self.x)
ISTA.update = update_step

def sml21n_inv_hessian_diag(self, x, out=None):
    denom = (x.power(2) + self.epsilon**2).power(1.5)
    diag = self.epsilon**2 / denom
    if out is not None:
        out[...] = diag
        return out
    return diag
SmoothMixedL21Norm.inv_hessian_diag = sml21n_inv_hessian_diag

def ocf_inv_hessian_diag(self, x, out=None):
    invn2 = 1/(self.operator.calculate_norm()**2)
    tmp = self.operator.direct(x)
    inv = self.operator.inv_hessian_diag(tmp) * invn2
    if out is not None:
        out.fill(self.operator.adjoint(inv, out=out))
        return out
    return self.operator.adjoint(inv, out=out)
OperatorCompositionFunction.inv_hessian_diag = MethodType(
    ocf_inv_hessian_diag, OperatorCompositionFunction
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=100000.)
    p.add_argument("--delta", type=float)
    p.add_argument("--num_subsets", type=int, default=9)
    p.add_argument("--no_prior", action="store_true")
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--initial_step_size", type=float, default=1.)
    p.add_argument("--relaxation_eta", type=float, default=0.02)
    p.add_argument("--pet_gauss_fwhm", type=float, nargs=3, default=(5.,5.,5.))
    p.add_argument(
        "--pet_data_path",
        default="/home/storage/prepared_data/oxford_patient_data/sirt3/PET"
    )
    p.add_argument(
        "--output_path",
        default="/home/sam/working/BSREM_PSMR_MIC_2024/results/test"
    )
    p.add_argument(
        "--source_path",
        default="/home/sam/working/BSREM_PSMR_MIC_2024/src"
    )
    p.add_argument(
        "--working_path",
        default="/home/sam/working/BSREM_PSMR_MIC_2024/tmp"
    )
    p.add_argument("--use_tof", action="store_true")
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--mode", choices=["joint","separate"], default="joint")
    return p.parse_args()

args = parse_args()
sys.path.append(args.source_path)

# utilities
from structural_priors.DirectionalOperator import DirectionalOperator
from utilities.data import get_pet_data_multiple_bed_pos
from utilities.functions import get_pet_am
from utilities.preconditioners import (
    BSREMPreconditioner, ImageFunctionPreconditioner,
    ClampedHarmonicMeanPreconditioner
)
from utilities.callbacks import (
    SaveImageCallback, SaveGradientUpdateCallback,
    PrintObjectiveCallback, SaveObjectiveCallback,
    SavePreconditionerCallback
)
from utilities.sirf import get_filters, get_sensitivity_from_subset_objs
from utilities.cil import BlockIndicatorBox, LinearDecayStepSizeRule, AdjointOperator
from utilities.shifts import CouchShiftOperator, ImageCombineOperator, get_couch_shift_from_sinogram

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

def save_args():
    os.makedirs(args.output_path, exist_ok=True)
    pd.DataFrame([vars(args)]).to_csv(
        os.path.join(args.output_path, "args.csv"),
        index=False
    )

def prepare_data():
    umap = ImageData(os.path.join(args.pet_data_path, "umap_zoomed.hv"))
    umap += (-umap).max()
    umap /= umap.max()
    pet = get_pet_data_multiple_bed_pos(
        args.pet_data_path,
        suffixes=["_f1b1","_f2b1"],
        tof=args.use_tof
    )
    cyl, gauss = get_filters()
    gauss.apply(pet["initial_image"])
    cyl.apply(pet["initial_image"])
    return umap, pet

def get_shift_ops(pet):
    beds = list(pet["bed_positions"].keys())
    # if only one bed, no shift needed
    if len(beds) == 1:
        ref = pet["bed_positions"][beds[0]]["template_image"]
        I = IdentityOperator(ref)
        return I, [I], [I]
    # else multi-bed as before
    shifts = [
        get_couch_shift_from_sinogram(
            pet["bed_positions"][b]["acquisition_data"]
        )
        for b in beds
    ]
    ops = [
        CouchShiftOperator(pet["bed_positions"][b]["template_image"], sh)
        for b, sh in zip(beds, shifts)
    ]
    shifted = [
        op.direct(pet["bed_positions"][b]["template_image"])
        for b, op in zip(beds, ops)
    ]
    comb   = ImageCombineOperator(BlockDataContainer(*shifted))
    unsh   = [AdjointOperator(op) for op in ops]
    choose = [
        BlockOperator(*[
            IdentityOperator(shifted[i]) if i == j
            else ZeroOperator(shifted[j], shifted[i])
            for j in range(len(beds))
        ], shape=(1, len(beds)))
        for i in range(len(beds))
    ]
    return AdjointOperator(comb), unsh, choose

def get_data_fidelity_joint(pet, get_am, subsets, uncomb, unsh, choose):
    beds = list(pet["bed_positions"].keys())
    # partition
    df_parts = []
    for b in beds:
        acq = pet["bed_positions"][b]["acquisition_data"]
        df_parts.append(partitioner.data_partition(
            acq,
            pet["bed_positions"][b]["additive"],
            pet["bed_positions"][b]["normalisation"],
            num_batches=subsets,
            mode="staggered",
            create_acq_model=get_am
        )[2])
    # setup
    for i, b in enumerate(beds):
        tmpl = pet["bed_positions"][b]["template_image"]
        for df in df_parts[i]:
            df.set_up(tmpl)
    # sensitivities
    sens_parts = [
        get_sensitivity_from_subset_objs(
            df_parts[i],
            pet["bed_positions"][b]["template_image"]
        )
        for i, b in enumerate(beds)
    ]
    # combine and invert
    sens_comb = uncomb.adjoint(BlockDataContainer(*[
        unsh[i].adjoint(sp) for i, sp in enumerate(sens_parts)
    ]))
    s_inv = sens_comb.clone()
    arr = sens_comb.as_array()
    s_inv.fill(np.reciprocal(arr, where=arr!=0))
    cyl, _ = get_filters(); cyl.apply(s_inv)
    # build operator list
    all_funs = []
    for i, part in enumerate(df_parts):
        for df in part:
            comp = CompositionOperator(unsh[i], choose[i], uncomb)
            all_funs.append(OperatorCompositionFunction(df, comp))
    return all_funs, s_inv, sens_parts

def get_data_fidelity_separate(pet, get_am, subsets):
    beds = list(pet["bed_positions"].keys())
    df_parts_list, sens_parts, s_inv_parts = [], [], []
    for b in beds:
        # partition
        acq = pet["bed_positions"][b]["acquisition_data"]
        parts = partitioner.data_partition(
            acq,
            pet["bed_positions"][b]["additive"],
            pet["bed_positions"][b]["normalisation"],
            num_batches=subsets,
            mode="staggered",
            create_acq_model=get_am
        )[2]
        tmpl = pet["bed_positions"][b]["template_image"]
        for df in parts:
            df.set_up(tmpl)
        df_parts_list.append(parts)
        # sens
        sens = get_sensitivity_from_subset_objs(parts, tmpl)
        sens_parts.append(sens)
        # inverse
        inv = sens.clone()
        arr = sens.as_array()
        inv.fill(np.reciprocal(arr, where=arr!=0))
        cyl, _ = get_filters(); cyl.apply(inv)
        s_inv_parts.append(inv)
    return df_parts_list, sens_parts, s_inv_parts

def get_prior(pet, init, attn):
    grad = GradientOperator(init)
    dirop = DirectionalOperator(grad.direct(attn))
    comp = CompositionOperator(dirop, grad)
    comp.calculate_norm = MethodType(lambda self: grad.calculate_norm(), comp)
    return OperatorCompositionFunction(SmoothMixedL21Norm(epsilon=1e-6), comp)

def run_recon(all_funs, init, attn, s_inv, pet):
    if not args.no_prior:
        pr = get_prior(pet, init, attn)
        pr = -args.alpha / len(all_funs) * pr
        attach_prior_hessian(pr, epsilon=1e-3)
        all_funs = [SumFunction(f, pr) for f in all_funs]
    U = len(all_funs)
    maxv = init.max(); eps = maxv/1e3
    bs = BSREMPreconditioner(s_inv, 1, np.inf,
                             epsilon=eps, max_vals=maxv, smooth=True)
    if args.no_prior:
        pre = bs
    else:
        ip = ImageFunctionPreconditioner(pr.inv_hessian_diag, 1., U,
                                         freeze_iter=np.inf, epsilon=eps)
        pre = ClampedHarmonicMeanPreconditioner([bs, ip],
                                                update_interval=U,
                                                freeze_iter=U*10)
    probs = [1/U]*U
    f_obj = -SVRGFunction(all_funs,
                          sampler=Sampler.random_with_replacement(U, prob=probs),
                          snapshot_update_interval=2*U,
                          store_gradients=True)
    algo = ISTA(initial=init, f=f_obj, g=BlockIndicatorBox(0, np.inf),
                preconditioner=pre,
                step_size=LinearDecayStepSizeRule(args.initial_step_size,
                                                  args.relaxation_eta),
                update_objective_interval=U)
    algo.run(args.num_epochs*U, verbose=1, callbacks=[
        SaveImageCallback(os.path.join(args.output_path,"image"), U),
        SaveGradientUpdateCallback(os.path.join(args.output_path,"gradient"), U),
        SavePreconditionerCallback(os.path.join(args.output_path,"preconditioner"), U),
        PrintObjectiveCallback(U),
        SaveObjectiveCallback(os.path.join(args.output_path,"objective"), U),
    ])
    pd.DataFrame(algo.loss).to_csv(
        os.path.join(args.output_path, f"obj_a_{args.alpha}.csv"),
        index=False
    )
    return algo.x_old

def main():
    configure_logging()
    os.chdir(args.working_path)
    msg = MessageRedirector()
    save_args()

    umap, pet = prepare_data()
    get_am = lambda: get_pet_am(not args.no_gpu,
                                gauss_fwhm=args.pet_gauss_fwhm)

    uncomb, unsh, choose = get_shift_ops(pet)

    if args.mode == "joint":
        funs, s_inv, sens_parts = get_data_fidelity_joint(
            pet, get_am, args.num_subsets, uncomb, unsh, choose
        )
        init = pet["initial_image"].clone()
        attn = pet["attenuation"].clone()
        recon = run_recon(funs, init, attn, s_inv, pet)
        recon.write(os.path.join(args.output_path, "recon_joint.hv"))
        return

    # separate mode
    df_list, sens_parts, s_inv_parts = get_data_fidelity_separate(
        pet, get_am, args.num_subsets
    )
    recons = {}
    beds = list(pet["bed_positions"].keys())
    for i, b in enumerate(beds):
        funs = df_list[i]
        init = pet["bed_positions"][b]["initial_image"].clone()
        attn = pet["bed_positions"][b]["attenuation"].clone()
        r = run_recon(funs, init, attn, s_inv_parts[i], pet)
        fname = f"recon{b}.hv"
        r.write(os.path.join(args.output_path, fname))
        recons[b] = r

    # align reconstructions & sens maps back to reference space
    aligned_recons = [
        unsh_i.adjoint(recons[b])
        for unsh_i, b in zip(unsh, beds)
    ]
    aligned_sens = [
        unsh_i.adjoint(sens_parts[i])
        for i, unsh_i in enumerate(unsh)
    ]

    aligned_combined = uncomb.adjoint(BlockDataContainer(*aligned_recons))

    # then do the weighted‐overlap merge
    combined = ImageCombineOperator.combine_images(
        reference=aligned_combined,
        images=BlockDataContainer(*aligned_recons),
        sens_images=BlockDataContainer(*aligned_sens),
        weight_overlap=True
    )
    combined.write(os.path.join(
        args.output_path, "recon_separate_weighted.hv"
    ))

if __name__ == "__main__":
    main()
