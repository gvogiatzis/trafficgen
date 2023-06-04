import sys
import os
import cv2
import subprocess
import numpy as np

from natsort import natsorted
from alive_progress import alive_bar
from pathlib import Path

if len(sys.argv) != 4:
    print(f'Usage mode: python3 {sys.argv[1]} path_to_gt_images path_to_result_images name_of_output_video')
    sys.exit(0)

results_path = Path(sys.argv[-2])
gt_path = Path(sys.argv[-3])

result_image_list = results_path.glob('*.*')
result_image_list = natsorted(result_image_list, key=str)

gt_image_list = gt_path.glob('*.*')
gt_image_list = natsorted(gt_image_list, key=str)

print("Loading images...")
with alive_bar(len(result_image_list), dual_line=True, force_tty=True, title='Dataset') as bar:
    img_array = []
    for idx, r_img_path in enumerate(result_image_list):
        r_img = cv2.imread(str(r_img_path))
        r_img = cv2.resize(r_img, (640, 384), interpolation=cv2.INTER_AREA)
        height, width, _ = r_img.shape
        size_r = (width, height)

        gt_img = cv2.imread(str(gt_image_list[idx]))
        gt_img = cv2.resize(gt_img, (640, 384), interpolation=cv2.INTER_AREA)
        height, width, _ = gt_img.shape
        size_gt = (width, height)

        concat = np.concatenate((gt_img, r_img), axis=1)

        img_array.append(concat)
        bar()

print("Writing images into video")
out = cv2.VideoWriter(sys.argv[-1] + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (640*2, 384))

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

bashCommand = f'ffmpeg -i {sys.argv[-1] + ".avi"} -pix_fmt yuv420p {sys.argv[-1] + ".mp4"} '
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

os.remove(sys.argv[-1] + ".avi")

