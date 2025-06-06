import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sirf.STIR import *
from image_stats import calculate_statistic_in_voi


# %% Define iteration lists and DTNV folder names (only PET images are processed)
dtnv_iters = list(range(21, 631, 21*5))     # DTNV iterations: 21, 42, …, 630

dtnv_root = "/home/sam/working/BSREM_PSMR_MIC_2024/results/old"
alphas = [64,128,256,512,1024,2056]
betas = [0.25,0.5,1,2,4,8]

dtnv_params = {}

# Pattern: alpha_{alpha}_beta_{beta}_<timestamp>
for folder in os.listdir(dtnv_root):
    for alpha in alphas:
        for beta in betas:
            pattern = fr"alpha_{alpha}_beta_{beta}(_|$)"
            if re.search(pattern, folder):
                key = f"{alpha}_{beta}"
                dtnv_params[key] = folder

pet_ellipsoid_image = ImageData("/home/sam/working/PSF_estimation/NEMAs/Mediso/PET/ellipsoid_image_p.hv")
spect_ellipsoid_image = ImageData("/home/sam/working/PSF_estimation/NEMAs/Mediso/SPECT/ellipsoid_image_s.hv")


diams_mm = [36,28,21,17,13,10] # diameters in mm of NEMA spheres
#spect_radii = [d/2*v for d, v in zip(diams_mm, spect_template_image.voxel_sizes())] # radii in mm
spect_radii=[tuple([d/2/v for v in  spect_ellipsoid_image.voxel_sizes()]) for d in diams_mm]
spect_centres = [(55, 49.5, 77.5), (55, 61.5, 70.75), (56, 61.75, 57.25), (55, 49.75, 50), (55.0, 36.5, 57), (55.0, 36.0, 70.0)]

pet_radii=[tuple([d/2/v for v in pet_ellipsoid_image.voxel_sizes()]) for d in diams_mm]
pet_centres = [(36, 86-6, 97.5-6), (36, 98-6, 90.5-6), (36, 98.5-6, 76-6), (36, 86-6, 69-6),(36, 73-6, 76-6), (36, 73-6, 91-6)] 

# Pre‐load the PET reference (ellipsoid) image and mask (assumed defined earlier)
pet_ref = pet_ellipsoid_image
pet_ref_mask = pet_ellipsoid_image.as_array() > 0

def calculate_rmse(image, ref, mask=None):
    truncation = TruncateToCylinderProcessor()
    truncation.set_strictly_less_than_radius(True)
    truncation.apply(image)
    truncation.apply(ref)
    diff = image - ref
    # Convert the difference to a NumPy array if necessary
    diff_np = diff.as_array() if hasattr(diff, "as_array") else diff
    # Apply the mask and compute RRMSE only over the masked voxels
    squared_diff = np.power(diff_np, 2)
    if mask is None:
        mask = np.ones(squared_diff.shape, dtype=bool)
    rmse = np.sqrt(squared_diff[mask].sum() / np.count_nonzero(mask))
    return rmse

# %% Helper function: compute metrics for a given PET image.
def compute_metrics(image_path):
    image = ImageData(image_path)
    # Compute overall RRMSE then square to obtain RMSE.
    rmse_unmasked = calculate_rmse(image, pet_ref)
    rmse_masked   = calculate_rmse(image, pet_ref, pet_ref_mask)
    # For each sphere (using pet_centres and pet_radii from earlier)
    sphere_means, sphere_covs = [], []
    for c, r in zip(pet_centres, pet_radii):
        mean_val = calculate_statistic_in_voi(image.as_array(), c, list(r), "mean", show=False)
        cov_val  = calculate_statistic_in_voi(image.as_array(), c, list(r), "cov", show=False)
        sphere_means.append(mean_val)
        sphere_covs.append(cov_val)
    return {"rmse_unmasked": rmse_unmasked, "rmse_masked": rmse_masked,
            "sphere_means": sphere_means, "sphere_covs": sphere_covs}

# %% Collect metrics for HKEM reconstructions.
hkem_results = []
for it in range(1):
    metrics = compute_metrics(f"/home/sam/working/BSREM_PSMR_MIC_2024/HKEM/ista/pet/reconstruction_x.hv")
    metrics["iteration"] = it
    hkem_results.append(metrics)

# %% Utility: convert a list of results (one per iteration) into a pandas DataFrame.
def results_to_df(results, num_spheres):
    data = {"iteration": [], "rmse_unmasked": [], "rmse_masked": []}
    for s in range(num_spheres):
        data[f"mean_sphere{s+1}"] = []
        data[f"cov_sphere{s+1}"]  = []
    for res in results:
        data["iteration"].append(res["iteration"])
        data["rmse_unmasked"].append(res["rmse_unmasked"])
        data["rmse_masked"].append(res["rmse_masked"])
        for s in range(num_spheres):
            data[f"mean_sphere{s+1}"].append(res["sphere_means"][s])
            data[f"cov_sphere{s+1}"].append(res["sphere_covs"][s])
    return pd.DataFrame(data)

hkem_df = results_to_df(hkem_results, len(pet_centres))

# %% Collect metrics for each DTNV reconstruction (for each alpha–beta folder).
dtnv_dfs = {}  # dictionary to store DataFrames for each parameter combination
for key, folder in dtnv_params.items():
    results = []
    for it in dtnv_iters:
        print(f"Processing {folder} at iteration {it}")
        metrics = compute_metrics(os.path.join(dtnv_root, folder, f"image_0_{it}.hv"))
        metrics["iteration"] = it
        results.append(metrics)
    dtnv_dfs[key] = results_to_df(results, len(pet_centres))

# %% Create plots.
# --- For HKEM ---
# Iteration vs. Mean (for each sphere)
for s in range(len(pet_centres)):
    plt.figure()
    plt.plot(hkem_df["iteration"], hkem_df[f"mean_sphere{s+1}"], marker='o')
    plt.xlabel("Iteration")
    plt.ylabel(f"Mean (Sphere {s+1})")
    plt.title("HKEM: Iteration vs. Mean")
    plt.grid(True)
    plt.savefig(os.path.join(dtnv_root, f"mean_sphere{s+1}.png"))
    plt.close()
    

# Iteration vs. Coefficient of Variation (for each sphere)
for s in range(len(pet_centres)):
    plt.figure()
    plt.plot(hkem_df["iteration"], hkem_df[f"cov_sphere{s+1}"], marker='o')
    plt.xlabel("Iteration")
    plt.ylabel(f"Cov (Sphere {s+1})")
    plt.title("HKEM: Iteration vs. Coefficient of Variation")
    plt.grid(True)
    plt.savefig(os.path.join(dtnv_root, f"cov_sphere{s+1}.png"))
    plt.close()
    

# Iteration vs. RMSE (masked and unmasked; overall image)
plt.figure()
plt.plot(hkem_df["iteration"], hkem_df["rmse_unmasked"], label="RMSE Unmasked", marker='o')
plt.plot(hkem_df["iteration"], hkem_df["rmse_masked"], label="RMSE Masked", marker='o')
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("HKEM: Iteration vs. RMSE")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(dtnv_root, "rmse.png"))
plt.close()


# Scatter: RMSE versus average CoV (compute average over all spheres)
hkem_df["avg_cov"] = hkem_df[[f"cov_sphere{s+1}" for s in range(len(pet_centres))]].mean(axis=1)
plt.figure()
plt.scatter(hkem_df["avg_cov"], hkem_df["rmse_unmasked"], label="Unmasked")
plt.scatter(hkem_df["avg_cov"], hkem_df["rmse_masked"], label="Masked")
plt.xlabel("Average Coefficient of Variation")
plt.ylabel("RMSE")
plt.title("HKEM: RMSE vs. Average Coefficient of Variation")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(dtnv_root, "rmse_vs_cov.png"))
plt.close()


# Scatter: For each sphere, Mean versus CoV (over iterations)
for s in range(len(pet_centres)):
    plt.figure()
    plt.scatter(hkem_df[f"mean_sphere{s+1}"], hkem_df[f"cov_sphere{s+1}"])
    plt.xlabel(f"Mean (Sphere {s+1})")
    plt.ylabel(f"Cov (Sphere {s+1})")
    plt.title(f"HKEM: Mean vs. Cov (Sphere {s+1})")
    plt.grid(True)
    plt.savefig(os.path.join(dtnv_root, f"mean_vs_cov_sphere{s+1}.png"))
    plt.close()
    

# --- For DTNV (example plots for one alpha–beta combination) ---
# Here we show an example for the "128_8" folder.
df_dtnv = dtnv_dfs["128_8"]
plt.figure()
plt.plot(df_dtnv["iteration"], df_dtnv["mean_sphere1"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Mean (Sphere 1)")
plt.title("DTNV (128_8): Iteration vs. Mean (Sphere 1)")
plt.grid(True)
plt.savefig(os.path.join(dtnv_root, "dtnv_mean_sphere1.png"))
plt.close()



# First, add an average CoV column to each DTNV DataFrame.
for key, df in dtnv_dfs.items():
    df["avg_cov"] = df[[f"cov_sphere{s+1}" for s in range(len(pet_centres))]].mean(axis=1)

# --- Plot: Iteration vs. Mean for each sphere (overlay all DTNV keys) ---
for s in range(len(pet_centres)):
    plt.figure(figsize=(10,6))
    for key, df in dtnv_dfs.items():
        plt.plot(df["iteration"], df[f"mean_sphere{s+1}"], marker='o', label=key)
    plt.xlabel("Iteration")
    plt.ylabel(f"Mean (Sphere {s+1})")
    plt.title(f"DTNV: Iteration vs. Mean (Sphere {s+1})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dtnv_root, f"mean_sphere{s+1}.png"))
    plt.close()
    

# --- Plot: Iteration vs. Coefficient of Variation for each sphere ---
for s in range(len(pet_centres)):
    plt.figure(figsize=(10,6))
    for key, df in dtnv_dfs.items():
        plt.plot(df["iteration"], df[f"cov_sphere{s+1}"], marker='o', label=key)
    plt.xlabel("Iteration")
    plt.ylabel(f"CoV (Sphere {s+1})")
    plt.title(f"DTNV: Iteration vs. CoV (Sphere {s+1})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dtnv_root, f"cov_sphere{s+1}.png"))
    plt.close()
    

# --- Plot: Iteration vs. Overall RMSE (masked and unmasked) ---
plt.figure(figsize=(10,6))
for key, df in dtnv_dfs.items():
    plt.plot(df["iteration"], df["rmse_unmasked"], marker='o', label=f"{key} Unmasked")
    plt.plot(df["iteration"], df["rmse_masked"], marker='x', label=f"{key} Masked")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("DTNV: Iteration vs. RMSE (Masked and Unmasked)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(dtnv_root, "rmse.png"))
plt.close()


# --- Scatter Plot: Average CoV vs. RMSE (masked and unmasked) ---
plt.figure(figsize=(10,6))
for key, df in dtnv_dfs.items():
    plt.scatter(df["avg_cov"], df["rmse_unmasked"], label=f"{key} Unmasked")
    plt.scatter(df["avg_cov"], df["rmse_masked"], label=f"{key} Masked", marker='x')
plt.xlabel("Average Coefficient of Variation")
plt.ylabel("RMSE")
plt.title("DTNV: RMSE vs. Average CoV")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(dtnv_root, "rmse_vs_cov.png"))
plt.close()


# --- Scatter Plot: Mean vs. CoV for each sphere ---
for s in range(len(pet_centres)):
    plt.figure(figsize=(10,6))
    for key, df in dtnv_dfs.items():
        plt.scatter(df[f"mean_sphere{s+1}"], df[f"cov_sphere{s+1}"], label=key)
    plt.xlabel(f"Mean (Sphere {s+1})")
    plt.ylabel(f"CoV (Sphere {s+1})")
    plt.title(f"DTNV: Mean vs. CoV (Sphere {s+1})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dtnv_root, f"mean_vs_cov_sphere{s+1}.png"))
    plt.close()
    

# save the dataframes
hkem_df.to_csv("hkem_df.csv")
for key, df in dtnv_dfs.items():
    df.to_csv(f"dtnv_df_{key}.csv")