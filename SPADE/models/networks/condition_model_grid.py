import sys
import math

import numpy as np
import cv2

from models.networks.gat import GAT
from models.networks.tcnn import TCNN

from utils.model_utils import *


class FinalModel(nn.Module):
    def __init__(self, limg_resolution, latent_channels, num_features, num_channels_out, gnn_hidden, cnn_hidden,
                 activation_gnn, activation_cnn, final_activation_gnn, final_activation_cnn, gnn_type, num_heads,
                 attn_drop, scale_factors, strides, kernel_sizes, paddings, residual, dropout=None, alpha=0.12):
        super(FinalModel, self).__init__()
        self.limg_resolution = limg_resolution
        self.latent_channels = latent_channels
        self.num_features = num_features
        self.residual = residual
        self.dropout = dropout

        self.gnn_hidden = gnn_hidden
        self.cnn_hidden = cnn_hidden

        self.activation_gnn = get_activation_function(activation_gnn)
        self.final_activation_gnn = get_activation_function(final_activation_gnn)
        self.activation_cnn = get_activation_function(activation_cnn)
        self.final_activation_cnn = get_activation_function(final_activation_cnn)

        self.attn_drop = attn_drop
        self.num_heads = num_heads
        self.alpha = alpha
        self.gnn_type = gnn_type

        self.scale_factors = scale_factors
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.paddings = paddings
        self.num_channels_out = num_channels_out

        self.gnn_object = GAT(len(self.gnn_hidden) + 1, self.num_features, self.latent_channels, self.gnn_hidden,
                              self.num_heads, self.activation_gnn, self.final_activation_gnn, self.dropout,
                              self.attn_drop, self.alpha, self.residual)

        self.tcnn_object = TCNN(self.cnn_hidden, self.latent_channels, self.num_channels_out, self.scale_factors,
                                self.strides, self.kernel_sizes, self.paddings, self.activation_cnn,
                                self.final_activation_cnn)

    def forward(self, data, input_graph, latent=False):

        x = self.gnn_object(data, input_graph)
        logits = x.squeeze()

        unbatched = dgl.unbatch(input_graph)
        cnn_input = torch.zeros([len(unbatched), self.latent_channels] + self.limg_resolution).to(data.device)

        n_nodes = 0
        for idx_batch, g in enumerate(unbatched):
            latent_image = logits[n_nodes:n_nodes+(self.limg_resolution[0] * self.limg_resolution[1]), :]
            latent_image = latent_image.view(self.limg_resolution[0], self.limg_resolution[1], self.latent_channels)
            latent_image = latent_image.permute(2, 0, 1)

            cnn_input[idx_batch, :, :, :] = latent_image
            n_nodes += g.number_of_nodes()

        output = self.tcnn_object(cnn_input)

        if not latent:
            return output  # Volume

        return output, cnn_input  # Volume and latent image
