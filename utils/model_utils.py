import torch
import torch.nn as nn
import dgl


def get_activation_function(activation_name):
    if activation_name == 'elu':
        activation = torch.nn.ELU()
    elif activation_name == 'leaky_relu':
        activation = torch.nn.LeakyReLU()
    elif activation_name == 'relu':
        activation = torch.nn.ReLU()
    elif activation_name == 'tanh':
        activation = torch.nn.Tanh()
    elif activation_name == 'sigmoid':
        activation = torch.nn.Sigmoid()
    else:
        activation = None

    return activation
