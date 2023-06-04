import sys

import torch
import pickle
import cv2

import numpy as np
from traffic_utils import get_features

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load_model(opt):
    params = pickle.load(open(opt.weights + '/traffic.prms', 'rb'), fix_imports=True)

    batch_size, lr, patience, weight_decay, latent_channels, latent_resolution, dropout, residual, networks_params, \
    best_loss = params.values()
    gnn_params, cnn_params = networks_params.values()
    gnn_net, alt, gnn_hidden, num_heads, gnn_activation, gnn_final_activation, alpha, att_drop = gnn_params.values()
    cnn_hidden, depth_output, scale_factors, kernel_sizes, strides, paddings, cnn_activation, \
    cnn_final_activation = cnn_params.values()

    n_features, all_features = get_features(alt)

    if int(alt) >= 1:
        from models.select_model_grid import FinalModel
    else:
        from models.select_model import FinalModel

    model = FinalModel(limg_resolution=latent_resolution,
                       latent_channels=latent_channels,
                       num_features=n_features,
                       num_channels_out=depth_output,
                       gnn_hidden=gnn_hidden,
                       cnn_hidden=cnn_hidden,
                       activation_gnn=gnn_activation,
                       activation_cnn=cnn_activation,
                       final_activation_gnn=gnn_final_activation,
                       final_activation_cnn=cnn_final_activation,
                       gnn_type=gnn_net,
                       num_heads=num_heads,
                       attn_drop=att_drop,
                       scale_factors=scale_factors,
                       strides=strides,
                       kernel_sizes=kernel_sizes,
                       paddings=paddings,
                       residual=residual,
                       dropout=dropout,
                       alpha=alpha
                       )

    model.load_state_dict(torch.load(opt.weights + '/traffic.tch', map_location=device))
    model.to(device)
    model.eval()
    return model, alt


def reformat_mask(mask):
    mask = np.moveaxis(mask, 0, -1)
    mask = ((mask/np.max(mask)) * 255).astype('uint8')
    mask = cv2.resize(mask, (640, 384), interpolation=cv2.INTER_AREA)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return mask


def reformat_output(output):
    output = np.moveaxis(output, 0, -1)

    # min_zero = output - np.min(output)
    # norm = min_zero / np.max(min_zero)
    # output_norm = (norm*255).astype('uint8')


    output_norm = (output*255)
    result = (np.argmax(output_norm, axis=2))
    result = ((result/np.max(result)) * 255).astype('uint8')

    output_norm = (output*250)
    output_norm[output_norm<0] = 0
    output_norm[output_norm>250] = 250
    output_norm = output_norm.astype('uint8')


    for ch in range(output_norm.shape[-1]):
        cv2.imshow(str(ch), output_norm[:, :, ch])

    result = cv2.resize(result, (640, 384), interpolation=cv2.INTER_AREA)
    output = cv2.applyColorMap(result, cv2.COLORMAP_JET)

    return output


def calculate_error(results):
    errors = []
    for result in results:
        logits, masks = result

        epsilon = 1e-12
        predictions = np.clip(logits, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(masks * np.log(predictions + 1e-9)) / N
        errors.append(ce)

    final_error = np.array(errors).mean()
    return final_error


def visualize(image, output, gt):
    mask = reformat_mask(gt)
    result = reformat_output(output)

    gt_output = np.concatenate((mask, result), axis=1)

    cv2.imshow('Image', image)
    cv2.imshow("Ground Truth - Output", gt_output)

