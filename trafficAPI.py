import sys
from skimage.transform import resize

import numpy as np
import yaml
from dgl.dataloading import GraphDataLoader
import copy

sys.path.append("../SPADE/")
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from graph_generator import TrafficDataset

from utils.training_utils import collate
from utils.traffic_utils import TrafficItem, SPADE_LABELS, YOLO_NAMES, IMAGE_SIZE, COLOR_PALETTE, DEVICE

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


class TrafficAPI(object):
    def __init__(self):
        super().__init__()
        self.options = TestOptions().parse()
        self.options.name = 'traffic3d'
        self.options.checkpoints_dir = '../SPADE/checkpoints'
        # self.options.gpu_ids = '0' if (DEVICE == 'cuda') else '-1'
        self.options.load_size = 640
        self.options.crop_size = 640
        self.options.batchSize = 1
        self.options.no_instance = True
        self.options.label_nc = len(SPADE_LABELS) + 1
        self.options.semantic_nc = len(SPADE_LABELS) + 1
        self.options.dataset_mode = 'custom'
        self.options.contain_dontcare_label = False

        self.model = Pix2PixModel(self.options)
        self.model.eval()

    def generate_one_output(self, data):
        dataset = TrafficDataset("CustomRussiaCrossroads", resolution=self.options.load_size, alt=alt, raw_dir='',
                                 visual_feats=data, alt_feats=alt_feats,
                                 grid_width=latent_resolution[0], mode='test', project='demo')

        dataloader = GraphDataLoader(dataset,
                                     batch_size=1,
                                     collate_fn=collate,
                                     shuffle=not self.options.serial_batches,
                                     num_workers=int(self.options.nThreads),
                                     drop_last=self.options.isTrain)
        for idx, data_i in enumerate(dataloader):
            generated = self.model(data_i, mode='demo')
            final_image = np.squeeze(generated.cpu().detach().numpy())
            final_image = np.moveaxis(final_image, 0, -1)
            final_image = resize(final_image, IMAGE_SIZE)
            return final_image
        return None
