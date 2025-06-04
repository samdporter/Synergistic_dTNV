#!/bin/bash
# run_experiments.sh
# Bash script to run the reconstruction script with varying alpha and beta parameters.
# Each run will have a unique output directory.

# Define the list of alpha and beta values.
alphas=(32 128 512)
betas=(0 2 8 32 128 512)

# Set the base output directory.
BASE_OUTPUT="/home/sam/working/BSREM_PSMR_MIC_2024/results"

# Set the path to your Python script (update if necessary)
SCRIPT_PATH="/home/sam/working/BSREM_PSMR_MIC_2024/main_cil_1bpos.py"

# Loop over each combination of alpha and beta.
for alpha in "${alphas[@]}"; do
    for beta in "${betas[@]}"; do
        # Create a unique output directory with a timestamp.
        timestamp=$(date +%Y%m%d_%H%M%S)
        output_dir="${BASE_OUTPUT}/anthro/alpha_${alpha}_beta_${beta}_${timestamp}"
        echo "Running with alpha=${alpha}, beta=${beta}."
        echo "Output directory: ${output_dir}"
        
        mkdir -p "${output_dir}"
        
        # Run the Python script with the given parameters.
        python3 "${SCRIPT_PATH}" \
            --alpha "${alpha}" \
            --beta "${beta}" \
            --output_path "${output_dir}" \
            --tail_singular_values 2 \
            --num_epochs 20 \
            --initial_step_size 1 \
            #--use_kappa \
        
        # Pause for 1 second to ensure a new timestamp if needed.
        sleep 1
    done
done
