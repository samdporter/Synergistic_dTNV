#!/bin/bash
set -e

# Paths
SCRIPT_DIR="/home/sam/working/BSREM_PSMR_MIC_2024"
DATA_DIR="/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data"
PET_DIR_SUFFIX="PET/phantom"
SPECT_DIR_SUFFIX="SPECT/phantom_140"
PET_OUT_SUFFIX="pet"
SPECT_OUT_SUFFIX="spect"
RECON_SCRIPT="${SCRIPT_DIR}/HKEM/hkem_1bpos.py"
RESAMPLE_SCRIPT="${SCRIPT_DIR}/HKEM/resample_to_pet.py"
# Common reconstruction args
EPOCH=30
SPECT_RES="1.21,0.03,false"
SAMPLING="sequential"
FREEZE_ITER=72
NUM_NEIGHBOURS=5
NUM_NON_ZERO_FEATURES=1
SIGMA_DM=5.0
SIGMA_DP=5.0
ONLY_2D=""
HYBRID="" #"--no_hybrid"
PET_GUIDANCE="emission"

DO_SPECT=true
DO_ISTA=true
DO_KOSMAPOSL=true
DO_KEM=true

if [ "$DO_SPECT" = true ]; then
    # 1. Run HKEM with SPECT
    python3 "$RECON_SCRIPT" \
        --modality SPECT \
        --method ista \
        --num_subsets 12 \
        --num_epochs 12 \
        --data_path "${DATA_DIR}/${SPECT_DIR_SUFFIX}" \
        --out_path "${SCRIPT_DIR}/HKEM/anthro" \
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

fi

if [ "$DO_ISTA" = true ]; then
    # 3. Run HKEM with PET
    python3 "$RECON_SCRIPT" \
        --modality PET \
        --method ista \
        --num_subsets 9 \
        --num_epochs "$EPOCH" \
        --data_path "${DATA_DIR}/${PET_DIR_SUFFIX}" \
        --out_path "${SCRIPT_DIR}/HKEM/anthro" \
        --out_suffix "${PET_OUT_SUFFIX}" \
        --source_path "${SCRIPT_DIR}/src" \
        --sampling "$SAMPLING" \
        --gauss_fwhm 5.0 5.0 5.0 \
        --spect_res $SPECT_RES \
        --freeze_iter "$FREEZE_ITER" \
        --num_neighbours "$NUM_NEIGHBOURS" \
        --num_non_zero_features "$NUM_NON_ZERO_FEATURES" \
        --sigma_m 3 \
        --sigma_p 1 \
        --sigma_dm "$SIGMA_DM" \
        --sigma_dp "$SIGMA_DP" \
        --guidance "$PET_GUIDANCE" \
        $ONLY_2D \
        $HYBRID
    
    if [ "$DO_KEM" = true ]; then
        # 3.5 Run KEM with PET
        python3 "$RECON_SCRIPT" \
            --modality PET \
            --method ista \
            --num_subsets 9 \
            --num_epochs "$EPOCH" \
            --data_path "${DATA_DIR}/${PET_DIR_SUFFIX}" \
            --out_path "${SCRIPT_DIR}/KEM/anthro" \
            --out_suffix "${PET_OUT_SUFFIX}" \
            --source_path "${SCRIPT_DIR}/src" \
            --sampling "$SAMPLING" \
            --gauss_fwhm 5.0 5.0 5.0 \
            --spect_res $SPECT_RES \
            --freeze_iter "$FREEZE_ITER" \
            --num_neighbours "$NUM_NEIGHBOURS" \
            --num_non_zero_features "$NUM_NON_ZERO_FEATURES" \
            --sigma_m 1 \
            --sigma_p 0.1 \
            --sigma_dm "$SIGMA_DM" \
            --sigma_dp "$SIGMA_DP" \
            --guidance "attenuation" \
            $ONLY_2D \
            --no_hybrid
    fi
fi

if [ "$DO_KOSMAPOSL" = true ]; then
    # 4. Run HKEM with PET using STIR stuff
    python3 "$RECON_SCRIPT" \
        --modality PET \
        --method kosmaposl \
        --num_subsets 9 \
        --num_epochs "$EPOCH" \
        --data_path "${DATA_DIR}/${PET_DIR_SUFFIX}" \
        --out_path "${SCRIPT_DIR}/HKEM/anthro/kosmaposl" \
        --out_suffix "${PET_OUT_SUFFIX}" \
        --source_path "${SCRIPT_DIR}/src" \
        --sampling "$SAMPLING" \
        --gauss_fwhm 5.0 5.0 5.0 \
        --spect_res $SPECT_RES \
        --freeze_iter "$FREEZE_ITER" \
        --num_neighbours "$NUM_NEIGHBOURS" \
        --num_non_zero_features "$NUM_NON_ZERO_FEATURES" \
        --sigma_m 1 \
        --sigma_p 0.1 \
        --sigma_dm "$SIGMA_DM" \
        --sigma_dp "$SIGMA_DP" \
        --guidance "$PET_GUIDANCE" \
        $ONLY_2D \
        $HYBRID

    if [ "$DO_KEM" = true ]; then
        # 4.5 Run KEM with PET using STIR stuff
        python3 "$RECON_SCRIPT" \
            --modality PET \
            --method kosmaposl \
            --num_subsets 9 \
            --num_epochs "$EPOCH" \
            --data_path "${DATA_DIR}/${PET_DIR_SUFFIX}" \
            --out_path "${SCRIPT_DIR}/KEM/anthro/kosmaposl" \
            --out_suffix "${PET_OUT_SUFFIX}" \
            --source_path "${SCRIPT_DIR}/src" \
            --sampling "$SAMPLING" \
            --gauss_fwhm 5.0 5.0 5.0 \
            --spect_res $SPECT_RES \
            --freeze_iter "$FREEZE_ITER" \
            --num_neighbours "$NUM_NEIGHBOURS" \
            --num_non_zero_features "$NUM_NON_ZERO_FEATURES" \
            --sigma_m 3 \
            --sigma_p 1 \
            --sigma_dm "$SIGMA_DM" \
            --sigma_dp "$SIGMA_DP" \
            --guidance "attenuation" \
            $ONLY_2D \
            --no_hybrid
    fi
fi
