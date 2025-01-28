import argparse
import sys
import os
import numpy as np

from sirf.STIR import ImageData

parser = argparse.ArgumentParser(description='test_shift_operator')

parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/src', help='source path')
parser.add_argument('--data_path', type=str, default="/home/sam/working/BSREM_PSMR_MIC_2024/data", help='data path')

args = parser.parse_args()

sys.path.insert(0, args.source_path)

from utilities.cil import CouchShiftOperator

def test_couch_shift_operator():
    # Load an image
    image = ImageData(os.path.join(args.data_path, "emission.hv"))

    # Create a couch shift operator
    shift = CouchShiftOperator(image, 10)

    # Shift the image
    shifted_image = shift.direct(image)

    # check 1st dim of shifted image origin is 10
    assert shifted_image.get_geometrical_info().get_offset()[2] == -10

    shifted_image_out = image.clone()
    shift.direct(image, shifted_image_out)

    assert shifted_image_out.get_geometrical_info().get_offset()[2] == -10

    # Revert the shift
    reverted_image = shift.adjoint(shifted_image)

    assert reverted_image.get_geometrical_info().get_offset()[2] == 0

    reverted_image_out = image.clone()
    shift.adjoint(shifted_image, reverted_image_out)

    assert reverted_image_out.get_geometrical_info().get_offset()[2] == 0

    # Check that the reverted image is the same as the original image
    assert np.allclose(reverted_image.as_array(), image.as_array())

# run if this script is executed
if __name__ == '__main__':
    test_couch_shift_operator()
    print("All tests passed.")
