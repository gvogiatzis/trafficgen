from sklearn.metrics import jaccard_score
import cv2
import numpy as np
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('path_real', type=str, help='path to real images')
parser.add_argument('path_generated', type=str, help='path to generated images')

args = parser.parse_args()


def pixel_accuracy(output, gt):
    accuracies = []
    for cls in range(5):
        a = np.array(gt)
        b = np.array(output)

        occurrences = np.count_nonzero(a == cls)
        if occurrences == 0:
            break
        a[a != cls] = -100
        b[b != cls] = -50

        diff = a - b
        correct = np.count_nonzero(diff == 0)

        accuracies.append(correct / occurrences)

    return accuracies


if __name__ == "__main__":
    pathlist = Path(args.path_real).glob('*.png')
    limit = None

    scores = []
    accus = []
    for idx, img_path in enumerate(pathlist):
        if idx == limit:
            break
        img_name = str(img_path).split('/')[-1]

        img_real = cv2.imread(str(img_path))
        img_real = cv2.resize(img_real, (640, 640), interpolation=cv2.INTER_AREA)
        img_real = img_real.astype(int)

        img_fake = cv2.imread(args.path_generated + '/' + img_name)
        img_fake = cv2.resize(img_fake, (640, 640), interpolation=cv2.INTER_AREA)
        img_fake = img_fake.astype(int)

        # Get metrics
        iou = jaccard_score(img_real.flatten(), img_fake.flatten(), average='micro')
        accu = pixel_accuracy(img_fake, img_real)
        # if len(iou) != 5:
        #     continue
        accus.append(accu)
        scores.append(iou)


        if idx % 20 == 0:
            print(f'{idx} images processed')

    print(f'Mean IoU is {np.array(scores).mean(axis=0)}')
    print(f'Mean pixel accuracy is {np.array(accus).mean(axis=0)}')

