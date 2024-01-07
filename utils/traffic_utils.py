import sys
import binascii
from collections import namedtuple

import random
import scipy.cluster
from skimage.transform import resize
import torch as th
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ###############################################################
# Utilities for creating the graphs
#################################################################
DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')
IMAGE_SIZE = (640, 384)
NUM_CLUSTERS = 3
YOLO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']

SPADE_LABELS = {'bus': YOLO_NAMES.index('bus'),
                'car': YOLO_NAMES.index('car'),
                'person': YOLO_NAMES.index('person'),
                'truck': YOLO_NAMES.index('truck')}

T_CLASSES = [c for c in SPADE_LABELS.values()]

COLOR_PALETTE = {'Black': [0, 0, 0], 'White': [255, 255, 255], 'Red': [255, 0, 0], 'Lime': [0, 255, 0],
                 'Blue': [0, 0, 255], 'Yellow': [255, 255, 0], 'Magenta': [255, 0, 255], 'Gray': [128, 128, 128]}
# 'Silver': [192, 192, 192], 'Gray': [128, 128, 128], 'Maroon': [128, 0, 0], 'Olive': [128, 128, 0],
# 'Green': [0, 128, 0], 'Purple': [128, 0, 128], 'Teal': [0, 128, 128], 'Navy': [0, 0, 128], 'Cyan': [0, 255, 255]}

graphData = namedtuple('graphData', ['src_nodes', 'dst_nodes', 'n_nodes', 'features'])
gridData = namedtuple('graphData', ['src_nodes', 'dst_nodes', 'n_nodes', 'features', 'ids'])
TrafficItem = namedtuple('TrafficItem', ['conf', 'xywh', 'cls', 'visualFeatures', 'dayTime'])


def get_features(alt, alt_feats):
    if alt_feats == 'img' or alt_feats == 'hue':
        image_features_length = 400
    elif alt_feats == 'mean':
        image_features_length = 3
    elif alt_feats == 'clusters':
        image_features_length = NUM_CLUSTERS * 4
    elif alt_feats == 'color':
        image_features_length = len(COLOR_PALETTE)
    else:
        print('Alternative for features not valid')
        sys.exit(0)

    if alt == '0':
        all_features = ['x_pos', 'y_pos', 'BBw', 'BBh'] + [x for x in SPADE_LABELS.keys()]
    elif alt == '1':
        all_features = ['x_pos', 'y_pos', 'BBw', 'BBh'] + [x for x in SPADE_LABELS.keys()] + ['grid']
    elif alt == '2':
        all_features = ['x_pos', 'y_pos', 'BBw', 'BBh', 'DT_cos', 'DT_sin'] + [x for x in SPADE_LABELS.keys()] + [
            'grid']
    else:
        print('Alternative for graph not valid')
        sys.exit(0)

    return len(all_features) + image_features_length, all_features, image_features_length


def closest_grid_nodes(grid_ids, x, y, w_i):
    if w_i > 50:
        radius = 3
    elif w_i > 30:
        radius = 2
    else:
        radius = 1

    c_x = round(x * (w_i - 1))
    c_y = round(y * (w_i - 1))

    range_c = np.arange(c_x - radius, c_x + radius + 1)
    range_r = np.arange(c_y - radius, c_y + radius + 1)

    grid_nodes = []
    for g_x in range_c:
        for g_y in range_r:
            p1 = np.array([g_x, g_y])
            p2 = np.array([x * (w_i - 1), y * (w_i - 1)])
            dist = np.linalg.norm(p1 - p2)
            if 0 <= g_x < w_i and 0 <= g_y < w_i and dist < radius:
                grid_nodes.append(grid_ids[g_x][g_y])

    return grid_nodes


# ###############################################################
# YOLO and image processing related utils
#################################################################

def get_features_from_bb(alt_feat, original_image, xywh, visualize=False):
    original_image = np.array(original_image)
    # Rescale coordinates of the bounding box
    tmp_xywh = np.copy(xywh)
    tmp_xywh[::2] = xywh[::2] * original_image.shape[1]
    tmp_xywh[1::2] = xywh[1::2] * original_image.shape[0]
    x, y, w, h = tmp_xywh

    # Crop the image
    cropped_image = original_image[int(y - h / 2):int(y - h / 2 + h), int(x - w / 2):int(x - w / 2 + w), :]
    if alt_feat == 'img':
        dim = (20, 20)
        resized = resize(cropped_image, dim)
        features = np.squeeze(resized.flatten() / 255)
    elif alt_feat == 'hue':
        # Get hue channel
        cropped_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        cropped_hue = cropped_hsv[:, :, 0]

        # Resize all images and flatten them
        dim = (20, 20)
        resized = resize(cropped_hue, dim)
        features = np.squeeze(resized.flatten() / 255)
    elif alt_feat == 'mean':
        features = np.mean(cropped_image, axis=(0, 1))
    elif alt_feat == 'clusters' or alt_feat == 'color':
        # Get clusters of most predominant colors
        ar = cropped_image
        shape = ar.shape
        ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

        # Get clusters
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

        vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
        counts, bins = np.histogram(vecs, len(codes))  # count occurrences

        probs = scipy.special.softmax(counts / np.max(counts))
        inds = probs.argsort()
        codes = codes[inds[::-1]]
        probs = probs[inds[::-1]]

        features = []
        for idx, c in enumerate(codes):
            features.append(probs[idx])
            for col in c:
                features.append(col / 255)

        features = np.array(features, dtype=float)
        if alt_feat == 'color':
            color_feats = features[-(NUM_CLUSTERS*4):].reshape(NUM_CLUSTERS, 4)
            mean_rgb = []
            pop_black = False
            for color in color_feats:
                rgb_code = color[-3:] * 255
                module = np.sqrt(rgb_code[0]**2 + rgb_code[1]**2 + rgb_code[2]**2)
                if module < 80 and not pop_black:
                    pop_black = True
                    continue
                mean_rgb.append(rgb_code)
            mean_rgb = np.array(mean_rgb).mean(axis=0)
            min_dist = np.inf
            final_color = None
            for color in COLOR_PALETTE:
                p_color = np.array(COLOR_PALETTE[color]).astype(np.float16)
                dist = np.linalg.norm(mean_rgb - p_color)
                if color == 'Gray' and dist > 25:
                    continue
                if dist < min_dist:
                    min_dist = dist
                    final_color = color
            features = np.zeros(len(COLOR_PALETTE))
            # features[list(COLOR_PALETTE.keys()).index("White")] = 1.
            features[list(COLOR_PALETTE.keys()).index(final_color)] = 1.

    else:
        print("Not valid feature alternative.")
        sys.exit(0)

    # if visualize:
    #     cv2.imshow('Original image', original_image)
    #     cv2.imshow('Resized', resized)
    #     cv2.imshow('Cropped image', cropped_image)
    #     key = cv2.waitKey(3000)  # pauses for 2 seconds before fetching next image
    #     if key == 27:
    #         cv2.destroyAllWindows()
    #         sys.exit(0)

    return features


def get_mask_from_bbs(resolution, bbs):
    mask = np.zeros((resolution[1], resolution[0]))
    for bb in bbs:
        cls = bb[0]
        xywh = np.array([float(x) for x in bb[1:-1]])

        xywh[::2] = xywh[::2] * resolution[0]
        xywh[1::2] = xywh[1::2] * resolution[1]
        x, y, w, h = xywh

        mask[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = T_CLASSES.index(int(float(cls))) + 1

    return mask


# ###############################################################
# Transformations for SPADE
#################################################################
def get_transform(resolution, method=transforms.InterpolationMode.BICUBIC, normalize=True, toTensor=True):
    transform_list = []

    osize = resolution
    transform_list.append(transforms.Resize(osize, interpolation=method))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
