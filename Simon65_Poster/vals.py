# %%
from sirf.STIR import *
from image_stats import calculate_statistic_in_voi

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_spheres_on_image(image, centres, radii, title, region=None):
    """
    Plot ellipses representing spheres on orthogonal slices of an image with anisotropic voxels.
    
    Parameters:
        image   - image object with .as_array() and .voxel_sizes() returning (z, y, x)
        centres - list of (z, y, x) centres in voxel indices
        radii   - list of (rz, ry, rx) physical radii (in voxels, i.e., mm / voxel size)
        title   - figure title
        region  - optional region specification as a tuple (z_spec, y_spec, x_spec) where each element
                  is either an int (for a fixed slice) or a slice. If provided, the image is cropped and
                  centre coordinates are adjusted accordingly.
    """
    img = image.as_array()
    voxel_sizes = image.voxel_sizes()  # (vz, vy, vx)
    
    # If region is provided, crop the image and update centres.
    if region is not None:
        img = img[region]
        new_centres = []
        for c in centres:
            new_c = []
            for i, spec in enumerate(region):
                if isinstance(spec, slice):
                    start = spec.start if spec.start is not None else 0
                    new_c.append(c[i] - start)
                else:  # fixed index: the coordinate becomes 0
                    new_c.append(0)
            new_centres.append(tuple(new_c))
        centres = new_centres

    nz, ny, nx = img.shape
    
    # Compute central slices of the (possibly cropped) image.
    z_slice = nz // 2
    y_slice = ny // 2
    x_slice = nx // 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    # --- XY plane ---
    ax = axs[0]
    ax.imshow(img[z_slice, :, :], cmap='gray')
    for c, r in zip(centres, radii):
        ry, rx = r[1], r[2]
        ellipse = patches.Ellipse((c[2], c[1]), 2*rx, 2*ry,
                                  edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(ellipse)
    ax.set_title(f'XY (z={z_slice})')
    ax.set_aspect(voxel_sizes[1] / voxel_sizes[2])

    # --- XZ plane ---
    ax = axs[1]
    ax.imshow(img[:, y_slice, :], cmap='gray')
    for c, r in zip(centres, radii):
        rz, rx = r[0], r[2]
        ellipse = patches.Ellipse((c[2], c[0]), 2*rx, 2*rz,
                                  edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(ellipse)
    ax.set_title(f'XZ (y={y_slice})')
    ax.set_aspect(voxel_sizes[0] / voxel_sizes[2])

    # --- YZ plane ---
    ax = axs[2]
    ax.imshow(img[:, :, x_slice], cmap='gray')
    for c, r in zip(centres, radii):
        rz, ry = r[0], r[1]
        ellipse = patches.Ellipse((c[1], c[0]), 2*ry, 2*rz,
                                  edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(ellipse)
    ax.set_title(f'YZ (x={x_slice})')
    ax.set_aspect(voxel_sizes[0] / voxel_sizes[1])

    plt.tight_layout()
    plt.show()


# %%
pet_osem = ImageData(f"/home/storage/prepared_data/phantom_data/nema_phantom_data/PET/recon_ss9_ep12.hv")
spect_osem = ImageData(f"/home/storage/prepared_data/phantom_data/nema_phantom_data/SPECT/recon_ss12_ep12.hv")

gauss = SeparableGaussianImageFilter()
gauss.set_fwhms((5.0, 5.0, 5.0))

pet_osem_smoothed = gauss.process(pet_osem)
spect_osem_smoothed = gauss.process(spect_osem)

# %%
fig, ax = plt.subplots(1,2, figsize=(10,5))
im = ax[0].imshow(pet_osem.as_array()[35])
ax[0].set_title('PET')
fig.colorbar(im, ax=ax[0])
im = ax[1].imshow(pet_osem_smoothed.as_array()[35])
ax[1].set_title('Smoothed PET')
fig.colorbar(im, ax=ax[1])


# %%
pet_osem.shape

# %%
diams_mm = [36,28,21,17,13,10] # diameters in mm of NEMA spheres
#spect_radii = [d/2*v for d, v in zip(diams_mm, spect_template_image.voxel_sizes())] # radii in mm
spect_radii=[tuple([d/2/v for v in  spect_osem.voxel_sizes()]) for d in diams_mm]
spect_centres = [(55, 49.5, 77.5), (55, 61.5, 70.75), (56, 61.75, 57.25), (55, 49.75, 50), (55.0, 36.5, 57), (55.0, 36.0, 70.0)]

pet_radii=[tuple([d/2/v for v in pet_osem.voxel_sizes()]) for d in diams_mm]
pet_centres = [(36, 86-6, 97.5-6), (36, 98-6, 90.5-6), (36, 98.5-6, 76-6), (36, 86-6, 69-6),(36, 73-6, 76-6), (36, 73-6, 91-6)] 

# %%
z_slice = 35         # fixed slice in the z-dimension
y_min, y_max = 50, -50  # y-range
x_min, x_max = 50, -50  # x-range
region = (slice(20,-20), slice(y_min, y_max), slice(x_min, x_max))
plot_spheres_on_image(pet_osem_smoothed, pet_centres, pet_radii, "PET OSEM Smoothed", region)


# %%
pet_osem_sphere_double_radii_mean_osem_smoothed = calculate_statistic_in_voi(
    pet_osem_smoothed.as_array(),
    pet_centres[0],
    tuple([r for r in pet_radii[0]]),  # Original large sphere
    statistic_func="spillover_correction",
    large_sphere_radii=tuple([1.5 * r for r in pet_radii[0]]),  # Double radius
    small_sphere_radii=tuple([r for r in pet_radii[0]]),  # Original large sphere
    show=True
)
pet_osem_sphere_double_radii_std_mean_osem_smoothed = calculate_statistic_in_voi(
    pet_osem_smoothed.as_array(),
    pet_centres[0],
    tuple([r for r in pet_radii[0]]),  # Original large sphere
    statistic_func="spillover_correction_std",
    large_sphere_radii=tuple([1.5 * r for r in pet_radii[0]]),  # Double radius
    small_sphere_radii=tuple([r for r in pet_radii[0]]),  # Original large sphere
    show=False
)

spect_osem_sphere_double_radii_mean_osem_smoothed = calculate_statistic_in_voi(
    spect_osem_smoothed.as_array(),
    spect_centres[0],
    tuple([r for r in spect_radii[0]]),  # Original large sphere
    statistic_func="spillover_correction",
    large_sphere_radii=tuple([1.5 * r for r in spect_radii[0]]),  # Double radius
    small_sphere_radii=tuple([r for r in spect_radii[0]]),  # Original large sphere
    show=True
)

spect_osem_sphere_double_radii_std_mean_osem_smoothed = calculate_statistic_in_voi(
    spect_osem_smoothed.as_array(),
    spect_centres[0],
    tuple([r for r in spect_radii[0]]),  # Original large sphere
    statistic_func="spillover_correction_std",
    large_sphere_radii=tuple([1.5 * r for r in spect_radii[0]]),  # Double radius
    small_sphere_radii=tuple([r for r in spect_radii[0]]),  # Original large sphere
    show=False
)

pet_osem_sphere_half_radii_mean_osem = calculate_statistic_in_voi(pet_osem.as_array(), pet_centres[0], [r/2 for r in pet_radii[0]], "mean", show=False)
pet_osem_sphere_half_radii_std_mean_osem = calculate_statistic_in_voi(pet_osem.as_array(), pet_centres[0], [r/2 for r in pet_radii[0]], "std_mean", show=False)

spect_osem_sphere_half_radii_mean_osem = calculate_statistic_in_voi(spect_osem.as_array(), spect_centres[0], [r/2 for r in spect_radii[0]], "mean", show=False)
spect_osem_sphere_half_radii_std_mean_osem = calculate_statistic_in_voi(spect_osem.as_array(), spect_centres[0], [r/2 for r in spect_radii[0]], "std_mean", show=False)

pet_osem_sphere_double_radii_mean_osem = calculate_statistic_in_voi(
    pet_osem.as_array(), 
    pet_centres[0], 
    tuple([r for r in pet_radii[0]]),  # Original large sphere
    statistic_func="spillover_correction", 
    large_sphere_radii=tuple([1.5 * r for r in pet_radii[0]]),  # Double radius
    small_sphere_radii=tuple([r for r in pet_radii[0]]),  # Original large sphere
    show=False
)
pet_osem_sphere_double_radii_std_mean_osem = calculate_statistic_in_voi(
    pet_osem.as_array(), 
    pet_centres[0], 
    tuple([r for r in pet_radii[0]]),  # Original large sphere
    statistic_func="spillover_correction_std", 
    large_sphere_radii=tuple([1.5 * r for r in pet_radii[0]]),  # Double radius
    small_sphere_radii=tuple([r for r in pet_radii[0]]),  # Original large sphere
    show=False
)

spect_osem_sphere_double_radii_mean_osem = calculate_statistic_in_voi(
    spect_osem.as_array(), 
    spect_centres[0], 
    tuple([r for r in spect_radii[0]]),  # Original large sphere
    statistic_func="spillover_correction", 
    large_sphere_radii=tuple([1.5 * r for r in spect_radii[0]]),  # Double radius
    small_sphere_radii=tuple([r for r in spect_radii[0]]),  # Original large sphere
    show=False
)
spect_osem_sphere_double_radii_std_mean_osem = calculate_statistic_in_voi(
    spect_osem.as_array(), 
    spect_centres[0], 
    tuple([r for r in spect_radii[0]]),  # Original large sphere
    statistic_func="spillover_correction_std", 
    large_sphere_radii=tuple([1.5 * r for r in spect_radii[0]]),  # Double radius
    small_sphere_radii=tuple([r for r in spect_radii[0]]),  # Original large sphere
    show=False
)

# %%
# rpint means and stds 
print(f"PET OSEM - Half Radii Mean: {pet_osem_sphere_half_radii_mean_osem} +/- {pet_osem_sphere_half_radii_std_mean_osem}")
print(f"PET OSEM - Double Radii Mean: {pet_osem_sphere_double_radii_mean_osem} +/- {pet_osem_sphere_double_radii_std_mean_osem}")

print(f"SPECT OSEM - Half Radii Mean: {spect_osem_sphere_half_radii_mean_osem} +/- {spect_osem_sphere_half_radii_std_mean_osem}")
print(f"SPECT OSEM - Double Radii Mean: {spect_osem_sphere_double_radii_mean_osem} +/- {spect_osem_sphere_double_radii_std_mean_osem}")


# %%
# print smoothed means and stds
print(f"PET OSEM Smoothed - Double Radii Mean: {pet_osem_sphere_double_radii_mean_osem_smoothed} +/- {pet_osem_sphere_double_radii_std_mean_osem_smoothed}")
print(f"SPECT OSEM Smoothed - Double Radii Mean: {spect_osem_sphere_double_radii_mean_osem_smoothed} +/- {spect_osem_sphere_double_radii_std_mean_osem_smoothed}")

# %%
spect_iteration = 144
spect_hkem = ImageData(f"/home/sam/working/BSREM_PSMR_MIC_2024/HKEM/spect/ista_{spect_iteration}.hv")
pet_iteration0 = 108
pet_hkem = ImageData(f"/home/sam/working/BSREM_PSMR_MIC_2024/HKEM/pet/ista_{pet_iteration0}.hv")
pet_iteration1 = 270
pet_hkem1 = ImageData(f"/home/sam/working/BSREM_PSMR_MIC_2024/HKEM/pet/ista_{pet_iteration1}.hv")

# %%
plt.imshow(pet_hkem.as_array()[35, :, :])

# %%
# find mean and std of all spheres for PET and SPECT hkem
pet_hkem_sphere_radii_mean = [calculate_statistic_in_voi(pet_hkem.as_array(), pet_centres[i], [r for r in pet_radii[i]], "mean", show=False) for i in range(len(pet_centres))]
pet_hkem_sphere_radii_std_mean = [calculate_statistic_in_voi(pet_hkem.as_array(), pet_centres[i], [r for r in pet_radii[i]], "std_mean", show=False) for i in range(len(pet_centres))]

spect_hkem_sphere_radii_mean = [calculate_statistic_in_voi(spect_hkem.as_array(), spect_centres[i], [r for r in spect_radii[i]], "mean", show=False) for i in range(len(spect_centres))]
spect_hkem_sphere_radii_std_mean = [calculate_statistic_in_voi(spect_hkem.as_array(), spect_centres[i], [r for r in spect_radii[i]], "std_mean", show=False) for i in range(len(spect_centres))]

# %%
# print all
print("PET HKEM - Mean and Std of all spheres:")
for i in range(len(pet_centres)):
    print(f"Sphere {i+1}: {pet_hkem_sphere_radii_mean[i]} +/- {pet_hkem_sphere_radii_std_mean[i]}")
print("SPECT HKEM - Mean and Std of all spheres:")
for i in range(len(spect_centres)):
    print(f"Sphere {i+1}: {spect_hkem_sphere_radii_mean[i]} +/- {spect_hkem_sphere_radii_std_mean[i]}")

# %%
iteration = 315


pet_dtnv_128_4 = ImageData(f"/home/sam/working/BSREM_PSMR_MIC_2024/results/alpha_128_beta_4_20250327_114207/image_0_{iteration}.hv")
pet_dtnv_64_4 = ImageData(f"/home/sam/working/BSREM_PSMR_MIC_2024/results/alpha_64_beta_4_20250327_051738/image_0_{iteration}.hv")
pet_dtnv_128_0_5 = ImageData(f"/home/sam/working/BSREM_PSMR_MIC_2024/results/alpha_128_beta_0.5_20250327_082918/image_0_{iteration}.hv")
pet_dtnv_64_05 = ImageData(f"/home/sam/working/BSREM_PSMR_MIC_2024/results/alpha_64_beta_0.5_20250327_020634/image_0_{iteration}.hv")
plt.imshow(pet_dtnv_128_4.as_array()[35, :, :])

# %%
pet_dtnv_128_4_sphere_radii_mean = [calculate_statistic_in_voi(pet_dtnv_128_4.as_array(), pet_centres[i], [r for r in pet_radii[i]], "mean", show=True) for i in range(len(pet_centres))]
pet_dtnv_128_4_sphere_radii_std_mean = [calculate_statistic_in_voi(pet_dtnv_128_4.as_array(), pet_centres[i], [r for r in pet_radii[i]], "std_mean", show=False) for i in range(len(pet_centres))]

pet_dtnv_64_sphere_radii_std_mean  = [calculate_statistic_in_voi(pet_dtnv_64_4.as_array(), pet_centres[i], [r for r in pet_radii[i]], "std_mean", show=False) for i in range(len(pet_centres))]
pet_dtnv_64_sphere_radii_mean = [calculate_statistic_in_voi(pet_dtnv_64_4.as_array(), pet_centres[i], [r for r in pet_radii[i]], "mean", show=False) for i in range(len(pet_centres))]

pet_dtnv_128_0_5_sphere_radii_std_mean  = [calculate_statistic_in_voi(pet_dtnv_128_0_5.as_array(), pet_centres[i], [r for r in pet_radii[i]], "std_mean", show=False) for i in range(len(pet_centres))]
pet_dtnv_128_0_5_sphere_radii_mean = [calculate_statistic_in_voi(pet_dtnv_128_0_5.as_array(), pet_centres[i], [r for r in pet_radii[i]], "mean", show=False) for i in range(len(pet_centres))]

pet_dtnv_64_0_5_sphere_radii_std_mean  = [calculate_statistic_in_voi(pet_dtnv_64_05.as_array(), pet_centres[i], [r for r in pet_radii[i]], "std_mean", show=False) for i in range(len(pet_centres))]
pet_dtnv_64_0_5_sphere_radii_mean = [calculate_statistic_in_voi(pet_dtnv_64_05.as_array(), pet_centres[i], [r for r in pet_radii[i]], "mean", show=False) for i in range(len(pet_centres))]

# %%
# print all pet_dtnv
print("PET DTNV 128_4 - Mean and Std of all spheres:")
for i in range(len(pet_centres)):
    print(f"Sphere {i+1}: {pet_dtnv_128_4_sphere_radii_mean[i]} +/- {pet_dtnv_128_4_sphere_radii_std_mean[i]}")
    
print("PET DTNV 128_0.5 - Mean and Std of all spheres:")
for i in range(len(pet_centres)):
    print(f"Sphere {i+1}: {pet_dtnv_128_0_5_sphere_radii_mean[i]} +/- {pet_dtnv_128_0_5_sphere_radii_std_mean[i]}")
    
print("PET DTV 64_4 - Mean and Std of all spheres:")
for i in range(len(pet_centres)):
    print(f"Sphere {i+1}: {pet_dtnv_64_sphere_radii_mean[i]} +/- {pet_dtnv_64_sphere_radii_std_mean[i]}")
    
print("PET DTV 64_0.5 - Mean and Std of all spheres:")  
for i in range(len(pet_centres)):
    print(f"Sphere {i+1}: {pet_dtnv_64_0_5_sphere_radii_mean[i]} +/- {pet_dtnv_64_0_5_sphere_radii_std_mean[i]}")

# %%
import numpy as np
def calculate_rmse(image, ref, mask=None):
    truncation = TruncateToCylinderProcessor()
    truncation.set_strictly_less_than_radius(True)
    truncation.apply(image)
    truncation.apply(ref)
    diff = image - ref
    # Convert the difference to a NumPy array if necessary
    diff_np = diff.as_array() if hasattr(diff, "as_array") else diff
    # Apply the mask and compute RMSE only over the masked voxels
    squared_diff = np.power(diff_np, 2)
    if mask is None:
        mask = np.ones(squared_diff.shape, dtype=bool)
    rmse = np.sqrt(squared_diff[mask].sum() / np.count_nonzero(mask))
    return rmse

# %%
pet_ellipsoid_image = ImageData("/home/sam/working/PSF_estimation/NEMAs/Mediso/PET/ellipsoid_image_p.hv")
spect_ellipsoid_image = ImageData("/home/sam/working/PSF_estimation/NEMAs/Mediso/SPECT/ellipsoid_image_s.hv")

# %%
pet_ellipsoid_image_mask = pet_ellipsoid_image.as_array() > 0
spect_ellipsoid_image_mask = spect_ellipsoid_image.as_array() > 0

# %%
rmse_pet_osem = calculate_rmse(pet_osem, pet_ellipsoid_image)
rmse_pet_osem_smoothed = calculate_rmse(pet_osem_smoothed, pet_ellipsoid_image)
rmse_pet_hkem = calculate_rmse(pet_hkem, pet_ellipsoid_image)
rmse_pet_dtnv_128_4 = calculate_rmse(pet_dtnv_128_4, pet_ellipsoid_image)
rmse_pet_dtnv_64_4 = calculate_rmse(pet_dtnv_64_4, pet_ellipsoid_image)
rmse_pet_dtnv_128_0_5 = calculate_rmse(pet_dtnv_128_0_5, pet_ellipsoid_image)
rmse_pet_dtnv_64_0_5 = calculate_rmse(pet_dtnv_64_05, pet_ellipsoid_image)

rmse_pet_osem_masked = calculate_rmse(pet_osem, pet_ellipsoid_image, pet_ellipsoid_image_mask)
rmse_pet_osem_smoothed_masked = calculate_rmse(pet_osem_smoothed, pet_ellipsoid_image, pet_ellipsoid_image_mask)
rmse_pet_hkem_masked = calculate_rmse(pet_hkem, pet_ellipsoid_image, pet_ellipsoid_image_mask)
rmse_pet_dtnv_128_4_masked = calculate_rmse(pet_dtnv_128_4, pet_ellipsoid_image, pet_ellipsoid_image_mask)
rmse_pet_dtnv_64_4_masked = calculate_rmse(pet_dtnv_64_4, pet_ellipsoid_image, pet_ellipsoid_image_mask)
rmse_pet_dtnv_128_0_5_masked = calculate_rmse(pet_dtnv_128_0_5, pet_ellipsoid_image, pet_ellipsoid_image_mask)
rmse_pet_dtnv_64_0_5_masked = calculate_rmse(pet_dtnv_64_05, pet_ellipsoid_image, pet_ellipsoid_image_mask)

# %%
# print all
print("RMSE PET OSEM: ", rmse_pet_osem)
print("RMSE PET OSEM Smoothed: ", rmse_pet_osem_smoothed)
print("RMSE PET HKEM: ", rmse_pet_hkem)
print("RMSE PET DTNV 128_4: ", rmse_pet_dtnv_128_4)
print("RMSE PET DTNV 64_4: ", rmse_pet_dtnv_64_4)
print("RMSE PET DTNV 128_0.5: ", rmse_pet_dtnv_128_0_5)
print("RMSE PET DTNV 64_0.5: ", rmse_pet_dtnv_64_0_5)
print("RMSE PET OSEM Masked: ", rmse_pet_osem_masked)
print("RMSE PET OSEM Smoothed Masked: ", rmse_pet_osem_smoothed_masked)
print("RMSE PET HKEM Masked: ", rmse_pet_hkem_masked)
print("RMSE PET DTNV 128_4 Masked: ", rmse_pet_dtnv_128_4_masked)
print("RMSE PET DTNV 64_4 Masked: ", rmse_pet_dtnv_64_4_masked)
print("RMSE PET DTNV 128_0.5 Masked: ", rmse_pet_dtnv_128_0_5_masked)
print("RMSE PET DTNV 64_0.5 Masked: ", rmse_pet_dtnv_64_0_5_masked)

# %%
# add all to a table
import pandas as pd
data = {
    "Method": ["PET OSEM", "PET OSEM Smoothed", "PET HKEM", "PET DTNV 128_4", "PET DTNV 64_4", "PET DTNV 128_0.5", "PET DTNV 64_0.5"],
    "RMSE": [rmse_pet_osem, rmse_pet_osem_smoothed, rmse_pet_hkem, rmse_pet_dtnv_128_4, rmse_pet_dtnv_64_4, rmse_pet_dtnv_128_0_5, rmse_pet_dtnv_64_0_5],
    "RMSE Masked": [rmse_pet_osem_masked, rmse_pet_osem_smoothed_masked, rmse_pet_hkem_masked, rmse_pet_dtnv_128_4_masked, rmse_pet_dtnv_64_4_masked, rmse_pet_dtnv_128_0_5_masked, rmse_pet_dtnv_64_0_5_masked]
}

df = pd.DataFrame(data)

# find the lowest rmse for each column
lowest_rmse = df.loc[df['RMSE'].idxmin()]
lowest_rmse_masked = df.loc[df['RMSE Masked'].idxmin()]
print("Lowest RMSE: ", lowest_rmse)
print("Lowest RMSE Masked: ", lowest_rmse_masked)


