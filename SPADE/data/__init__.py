"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import yaml
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset

sys.path.append("../")

from graph_generator import TrafficDataset
from dgl.dataloading import GraphDataLoader
from utils.training_utils import *

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


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported. 
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt):
    mode = 'train' if opt.isTrain else 'test'
    dataset = TrafficDataset("RussiaCrossroads", raw_dir='../raw_data', resolution=opt.load_size, alt=alt,
                             alt_feats=alt_feats, grid_width=latent_resolution[0], mode=mode, project=opt.project)

    dataloader = GraphDataLoader(dataset,
                                 batch_size=opt.batchSize,
                                 collate_fn=collate,
                                 shuffle=not opt.serial_batches,
                                 num_workers=int(opt.nThreads),
                                 drop_last=opt.isTrain)
    return dataloader
