import torch
import torch.nn as nn


class TCNN(nn.Module):
    def __init__(self, hidden_dims, num_channels_in, num_channels_out, scale_factors, strides, kernel_sizes, paddings,
                 activation, final_activation):
        super(TCNN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Upsample(scale_factor=scale_factors[0], mode='nearest'))
        self.layers.append(nn.ReflectionPad2d(1))
        self.layers.append(nn.Conv2d(in_channels=num_channels_in, out_channels=hidden_dims[0],
                                     kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0]))

        # Hidden layers
        for lyr in range(1, len(hidden_dims)):
            self.layers.append(nn.Upsample(scale_factor=scale_factors[lyr], mode='nearest'))
            self.layers.append(nn.ReflectionPad2d(1))
            self.layers.append(nn.Conv2d(in_channels=hidden_dims[lyr - 1], out_channels=hidden_dims[lyr],
                                         kernel_size=kernel_sizes[lyr], stride=strides[lyr], padding=paddings[lyr]))

        # Output layer
        self.layers.append(nn.Upsample(scale_factor=scale_factors[-1], mode='nearest'))
        self.layers.append(nn.ReflectionPad2d(1))
        self.layers.append(nn.Conv2d(in_channels=hidden_dims[-1], out_channels=num_channels_out,
                                     kernel_size=kernel_sizes[-1], stride=strides[-1], padding=paddings[-1]))

        self.activation = activation
        self.final_activation = final_activation

    def forward(self, inputs):
        x = inputs
        for nl in range(len(self.layers) - 1):
            x = self.layers[nl](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        if self.final_activation is not None:
            output = self.final_activation(x)
        else:
            output = x
        return output
