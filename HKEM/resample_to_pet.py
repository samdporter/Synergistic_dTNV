import argparse
import sys
import os

from sirf.Reg import NiftiImageData3DDisplacement
from sirf.STIR import ImageData

def get_args():
    
    parser = argparse.ArgumentParser(description="PET to SPECT conversion")
    parser.add_argument(
        "--pet_path",
        type=str,
        default="/home/storage/prepared_data/phantom_data/nema_phantom_data/PET/template_image.hv",
        help="Path to PET data."
    )
    parser.add_argument(
        "--transform_path",
        type=str,
        default="/home/storage/prepared_data/phantom_data/nema_phantom_data/SPECT/spect2pet.nii",
        help="Path to transformation data."
    )
    parser.add_argument(
        "--spect_path",
        type=str,
        default="/home/sam/working/BSREM_PSMR_MIC_2024/HKEM/spect/reconstruction_x.hv",
        help="Path to SPECT data."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/home/sam/working/BSREM_PSMR_MIC_2024/src",
        help="Path to source code."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/storage/prepared_data/phantom_data/nema_phantom_data/PET",
        help="Path to output data."
    )
    
    return parser.parse_args()

def main():
    
    args = get_args()
    sys.path.append(args.source_path)
    from utilities.nifty import NiftyResampleOperator
    
    pet_template = ImageData(args.pet_path)
    spect_recon = ImageData(args.spect_path)
    
    transform = NiftiImageData3DDisplacement(args.transform_path)
    
    resampler = NiftyResampleOperator(
        reference=pet_template,
        floating=spect_recon,
        transform=transform
    )
    
    spect_recon2pet = resampler.direct(spect_recon)
    
    spect_recon2pet.write(os.path.join(args.output_path, "spect.hv"))
    
if __name__ == "__main__":
    
    main()
    