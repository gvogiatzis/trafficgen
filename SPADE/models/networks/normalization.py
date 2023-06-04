"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm

sys.path.append("../")
from utils.traffic_utils import get_features

with open("../config/training_volumes.yaml", "r") as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(f'Error loading the hyperparameters: {exc}')
        sys.exit(0)

batch_size, lr, patience, weight_decay, latent_channels, latent_resolution, dropout, residual, networks_params = params.values()
gnn_params, cnn_params = networks_params.values()
gnn_net, alt_feats, alt, gnn_hidden, num_heads, gnn_activation, gnn_final_activation, alpha, att_drop = gnn_params.values()
cnn_hidden, depth_output, scale_factors, kernel_sizes, strides, paddings, cnn_activation, cnn_final_activation = cnn_params.values()

n_features, all_features, _ = get_features(alt, alt_feats)


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2

        if int(alt) > 0:
            from models.networks.condition_model_grid import FinalModel as ConditionModel
        else:
            from models.networks.condition_model import FinalModel as ConditionModel

        self.gnn_condition = ConditionModel(limg_resolution=latent_resolution,
                                            latent_channels=latent_channels,
                                            num_features=n_features,
                                            num_channels_out=nhidden,
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

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, feats, graph, latent=False):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        if latent:
            actv, latent_img = self.gnn_condition(feats, graph, latent=True)
        else:
            latent_img = None
            actv = self.gnn_condition(feats, graph)

        actv = F.interpolate(actv, size=x.size()[2:], mode='nearest')

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        if not latent:
            return out

        return out, latent_img
