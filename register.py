#%% Import the required libraries
from sirf.STIR import ImageData, SeparableGaussianImageFilter
from sirf.Reg import NiftyResample, AffineTransformation, NiftyAladinSym
from cil.framework import BlockDataContainer

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

import argparse

parser = argparse.ArgumentParser(description='register')

parser.add_argument('--data_path', type=str, default="/home/storage/oxford_patient_data/prepared_data/sirt3", help='data path')
parser.add_argument('--output_path', type=str, default="//home/storage/oxford_patient_data/prepared_data/sirt3", help='output path')
parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/src', help='source path')
parser.add_argument('--working_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/tmp', help='working path')
parser.add_argument('--show_plot', type=bool, default=True, help='show plot')
parser.add_argument('--levels', type=int, default=4, help='number of levels')
parser.add_argument('--levels_to_perform', type=int, default=4, help='number of levels to perform')
parser.add_argument('--iterations', type=int, default=50, help='number of iterations')
# zooms in z,y,x
parser.add_argument('--zooms', type=float, nargs=3, default=[1, 2, 2], help='zooms in z,y,x')

args, _ = parser.parse_known_args()

sys.path.insert(0, args.source_path)
from src.utilities.cil import CouchShiftOperator, ImageCombineOperator, get_couch_shift_from_sinogram

#%% Load the images
pet_img_files = sorted(glob.glob(os.path.join(args.data_path, "PET", "ctac", "*.img")))

if pet_img_files:
    pet_ct_from_dicom = ImageData(pet_img_files[0])
else:
    raise FileNotFoundError("No .img files found in the specified directory.")

spect_dcm_files = sorted(glob.glob(os.path.join(args.data_path, "SPECT", "ctac", "*.dcm")))

if spect_dcm_files:
    spect_ct_from_dicom = ImageData(spect_dcm_files[0])
else:
    raise FileNotFoundError("No .dcm files found in the specified directory.")

def range_0_1(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)

pet_ct_from_dicom_array = pet_ct_from_dicom.as_array()
# crop image to len(z)/zoom(1),  len(y)/zoom(2), len(x)/zoom(3)
min_z, max_z = pet_ct_from_dicom_array.shape[0]/args.zooms[0]//2, pet_ct_from_dicom_array.shape[0]/args.zooms[0]/2+pet_ct_from_dicom_array.shape[0]%args.zooms[0]
min_y, max_y = pet_ct_from_dicom_array.shape[1]/args.zooms[1]//2, pet_ct_from_dicom_array.shape[1]/args.zooms[1]/2+pet_ct_from_dicom_array.shape[1]%args.zooms[1]
min_x, max_x = pet_ct_from_dicom_array.shape[2]/args.zooms[2]//2, pet_ct_from_dicom_array.shape[2]/args.zooms[2]/2+pet_ct_from_dicom_array.shape[2]%args.zooms[2]
pet_ct_from_dicom_array = pet_ct_from_dicom_array[int(min_z):int(max_z), int(min_y):int(max_y), int(min_x):int(max_x)]

pet_ct_from_dicom_zoomed = ImageData()
pet_ct_from_dicom_zoomed.initialise(pet_ct_from_dicom_array.shape, pet_ct_from_dicom.voxel_sizes(), pet_ct_from_dicom.get_geometrical_info().get_offset())
pet_ct_from_dicom_zoomed.fill(pet_ct_from_dicom_array)

pet_ct_from_dicom_zoomed.write(os.path.join(args.output_path, "PET", "ctac", "pet_ct_from_dicom_zoomed.hv"))

if args.show_plot:
    plt.imshow(pet_ct_from_dicom_zoomed.as_array()[pet_ct_from_dicom_zoomed.shape[0] // 2, :, :])
    plt.show()

# HU values so we need to set the minimum to -1000
pet_ct_from_dicom_zoomed.maximum(-1000, out=pet_ct_from_dicom)
spect_ct_from_dicom.maximum(-1000, out=spect_ct_from_dicom)

pet_ct_from_dicom_zoomed = range_0_1(pet_ct_from_dicom)
spect_ct_from_dicom = range_0_1(spect_ct_from_dicom)

spect_template_image = ImageData(os.path.join(args.data_path, "SPECT", "template_image.hv"))
spect_template_image_length = spect_template_image.dimensions()[0]*spect_template_image.voxel_sizes()[0]

#%% We need to shift the CT image to match the SPECT image
spect_ct_length = spect_ct_from_dicom.dimensions()[0]*spect_ct_from_dicom.voxel_sizes()[0]
spect_ct_width = spect_ct_from_dicom.dimensions()[1]*spect_ct_from_dicom.voxel_sizes()[1]

spect_ct_from_dicom = CouchShiftOperator.modify_pixel_offset(spect_ct_from_dicom, -spect_ct_width/2, 1)
spect_ct_from_dicom = CouchShiftOperator.modify_pixel_offset(spect_ct_from_dicom, -spect_ct_width/2, 2)
spect_ct_from_dicom = CouchShiftOperator.modify_pixel_offset(spect_ct_from_dicom, (spect_template_image_length - spect_ct_length)/2, 3)

spect_ct = spect_ct_from_dicom.zoom_image_as_template(spect_template_image, 'preserve_values')

#%% We need to flip the SPECT image in x and z
spect_array = spect_ct.as_array()
# needs flipping in x and z
spect_array = np.flip(spect_array, axis=2)
spect_array = np.flip(spect_array, axis=0)
spect_ct.fill(spect_array)

print("PET image dimensions: ", pet_ct_from_dicom_zoomed.dimensions())
print("PET image voxel sizes: ", pet_ct_from_dicom_zoomed.voxel_sizes())
print("SPECT image dimensions: ", spect_ct.dimensions())
print("SPECT image voxel sizes: ", spect_ct.voxel_sizes())

#%% Resample the SPECT image to the PET image
reg = NiftyAladinSym()
reg.set_reference_image(pet_ct_from_dicom_zoomed)
reg.set_floating_image(spect_ct)
reg.set_parameter('SetPerformRigid', '1')
reg.set_parameter('SetPerformAffine', '0')
# set number of levels
reg.set_parameter('SetNumberOfLevels', str(args.levels))
reg.set_parameter('SetLevelsToPerform', str(args.levels_to_perform))
reg.set_parameter('SetMaxIterations', str(args.iterations))
reg.process()
spect_registered = reg.get_output()

#%% let's check the results
pet_array = pet_ct_from_dicom_zoomed.as_array()
spect_registered_array = spect_registered.as_array()

if args.show_plot:
    titles = ['PET CT', 'SPECT CT', 'SPECT registered to PET CT']
    arrays = [pet_array, spect_array, spect_registered_array]

    fig, ax = plt.subplots(3, len(arrays), figsize=(len(arrays) * 3, 9))

    # Loop through the subplots and plot the images
    for j in range(len(arrays)):
        for i in range(3):
            if i == 0:
                img = arrays[j][arrays[j].shape[0] // 2, :, :]
            elif i == 1:
                img = arrays[j][:, arrays[j].shape[1] // 2, :]
            else:
                img = arrays[j][:, :, arrays[j].shape[2] // 2]
            
            ax[i, j].imshow(img)
            ax[i, j].set_title(titles[j])
            ax[i, j].axis('off')

    plt.show()

pet_template_f1b1 = ImageData(os.path.join(args.data_path, "PET", "template_image_f1b1.hv"))
pet_template_f2b1 = ImageData(os.path.join(args.data_path, "PET", "template_image_f2b1.hv"))

shift_f1b1 = get_couch_shift_from_sinogram(os.path.join(args.data_path, "PET", "non_tof", "prompts_f1b1.hs"))
shift_f2b1 = get_couch_shift_from_sinogram(os.path.join(args.data_path, "PET", "non_tof", "prompts_f2b1.hs"))

pet_f1b1 = CouchShiftOperator.modify_pixel_offset(pet_template_f1b1, shift_f1b1, 3)
pet_f2b1 = CouchShiftOperator.modify_pixel_offset(pet_template_f2b1, shift_f2b1, 3)

combine = ImageCombineOperator(BlockDataContainer(pet_f1b1, pet_f2b1), offset_z=shift_f2b1)
combined = combine.direct(BlockDataContainer(pet_f1b1, pet_f2b1))

pet_ct_zoomed2pet = pet_ct_from_dicom_zoomed.zoom_image_as_template(combined, 'preserve_values')

#%% Resample the PET zoomed image to the PET reference image
reg = NiftyAladinSym()
reg.set_reference_image(pet_ct_from_dicom_zoomed)
reg.set_floating_image(pet_ct_zoomed2pet)
reg.set_parameter('SetPerformRigid', '1')
reg.set_parameter('SetPerformAffine', '0')
# set number of levels
reg.set_parameter('SetNumberOfLevels', str(args.levels))
reg.set_parameter('SetLevelsToPerform', str(args.levels_to_perform))
reg.set_parameter('SetMaxIterations', str(args.iterations))
reg.process()
pet_registered = reg.get_output()

#%% let's check the results
pet_registered_array = pet_registered.as_array()

if args.show_plot:
    titles = ['PET CT', 'PET CT zoomed to PET reference', 'PET registered to PET reference']
    arrays = [pet_array, pet_ct_zoomed2pet.as_array(), pet_registered_array]

    fig, ax = plt.subplots(3, len(arrays), figsize=(len(arrays) * 3, 9))

    # Loop through the subplots and plot the images
    for j in range(len(arrays)):
        for i in range(3):
            if i == 0:
                img = arrays[j][arrays[j].shape[0] // 2, :, :]
            elif i == 1:
                img = arrays[j][:, arrays[j].shape[1] // 2, :]
            else:
                img = arrays[j][:, :, arrays[j].shape[2] // 2]
            
            ax[i, j].imshow(img)
            ax[i, j].set_title(titles[j])
            ax[i, j].axis('off')

    plt.show()