import sys
import os

import numpy as np
from PIL import Image
from pathlib import Path


save_directory = 'masked_images/'
os.mkdir(save_directory)

mask = np.array(Image.open('./tools/mask.jpg'))

for idx, img_path in enumerate(Path(sys.argv[1]).glob('*.jpg')):
    if idx % 100 == 0:
        print(f'{idx} images generated with mask')

    image_name = str(img_path).split('/')[-1]
    image = np.array(Image.open(img_path))
    image[mask == 0] = 0

    image = Image.fromarray(image)
    image.save(save_directory + image_name)
