import sys
import os
from pathlib import Path
from natsort import natsorted
import argparse
import cv2
import numpy as np
from PIL import Image

sys.path.append("../utils/")
from traffic_utils import get_mask_from_bbs

parser = argparse.ArgumentParser(description='Generate masks')
parser.add_argument('images', type=str, help='path to images')
parser.add_argument('boxes', type=str, help='path to boxes')
parser.add_argument('--pretty', dest='prettify', action='store_true',
                    default=False,
                    help='Prettify the masks (default: false)')
opt = parser.parse_args()

images_paths = natsorted(Path(opt.images).glob('*.png'), key=str)
bbs_pats = natsorted(Path(opt.boxes).glob('*.txt'), key=str)

assert len(images_paths) == len(bbs_pats), "The number of image and bounding boxes should be the same"

if opt.prettify:
    save_directory = "masks_generated_pretty"
else:
    save_directory = "masks_generated"
os.mkdir(save_directory)

for idx in range(len(images_paths)):
    file_name = str(images_paths[idx]).split('/')[-1].split('.')[0]

    image = Image.open(str(images_paths[idx]))
    with open(bbs_pats[idx]) as f:
        bounding_boxes = [line.rstrip().split(' ') for line in f]

    mask = get_mask_from_bbs(image.size, bounding_boxes)
    mask = Image.fromarray(mask).convert('RGB')

    if opt.prettify:
        mask = np.array(mask)*50
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        cv2.imwrite(save_directory + '/' + file_name + '.png', mask)
    else:
        mask.save(save_directory + '/' + file_name + '.png')


