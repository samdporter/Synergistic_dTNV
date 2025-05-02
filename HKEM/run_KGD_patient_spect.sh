#!/bin/bash
set -e

# Paths
SCRIPT_DIR="/home/sam/working/BSREM_PSMR_MIC_2024"
DATA_DIR="/home/storage/prepared_data/oxford_patient_data/sirt3"
PET_DIR_SUFFIX="PET"
SPECT_DIR_SUFFIX="SPECT"
PET_OUT_SUFFIX="pet"
SPECT_OUT_SUFFIX="spect"
RECON_SCRIPT="${SCRIPT_DIR}/HKEM/hkem_1bpos.py"
RESAMPLE_SCRIPT="${SCRIPT_DIR}/HKEM/resample_to_pet.py"
# Common reconstruction args
EPOCH=30
SPECT_RES="0.923,0.03,False"
SAMPLING="sequential"
FREEZE_ITER=72
NUM_NEIGHBOURS=5
NUM_NON_ZERO_FEATURES=1
SIGMA_DM=5.0
SIGMA_DP=5.0
ONLY_2D=""
HYBRID="" #"--no_hybrid"

# 1. Run HKEM with SPECT
python3 "$RECON_SCRIPT" \
    --modality SPECT \
    --method ista \
    --num_subsets 12 \
    --num_epochs 12 \
    --data_path "${DATA_DIR}/${SPECT_DIR_SUFFIX}" \
    --out_path "${SCRIPT_DIR}/HKEM" \
    --out_suffix "${SPECT_OUT_SUFFIX}" \
    --source_path "${SCRIPT_DIR}/src" \
    --sampling "$SAMPLING" \
    --gauss_fwhm 6.2 6.2 6.2 \
    --spect_res $SPECT_RES \
    --freeze_iter "$FREEZE_ITER" \
    --num_neighbours "$NUM_NEIGHBOURS" \
    --num_non_zero_features "$NUM_NON_ZERO_FEATURES" \
    --sigma_m 1 \
    --sigma_p 0.1 \
    --sigma_dm "$SIGMA_DM" \
    --sigma_dp "$SIGMA_DP" \
    $ONLY_2D \
    $HYBRID

# 2. Resample SPECT result to PET space
python3 "$RESAMPLE_SCRIPT" \
    --pet_path "${DATA_DIR}/${PET_DIR_SUFFIX}/template_image.hv" \
    --spect_path "${SCRIPT_DIR}/HKEM/${SPECT_OUT_SUFFIX}/reconstruction_x.hv" \
    --transform_path "${DATA_DIR}/${SPECT_DIR_SUFFIX}/spect2pet.nii" \
    --output_path "${DATA_DIR}/${PET_DIR_SUFFIX}" \
    --source_path "${SCRIPT_DIR}/src"
