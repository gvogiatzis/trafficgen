import os
import sys
from pathlib import Path

import math
import torch
import torch as th
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
from natsort import natsorted

from utils.alternatives import *


class TrafficDataset(DGLDataset):

    def __init__(self, name='TrafficDataset', raw_dir='raw_data', save_dir='cache_dataset', project='volumes',
                 mode='train', debug=False, limit=None, resolution=256, alt='1', alt_feats='img', visual_feats=None,
                 grid_width=20, force_reload=False, verbose=False):

        self.force_reload = force_reload
        self.mode = mode
        self.project = project
        self.debug = debug
        self.limit = limit
        self.resolution = resolution
        self.alt = alt
        self.alt_feats = alt_feats
        self.grid_width = grid_width
        self.device = DEVICE

        if self.project == 'demo':
            self.visual_feats = [bb.visualFeatures for bb in visual_feats[0]]
            self.data_demo = visual_feats
        elif self.project == 'SUMMO':
            self.visual_feats = 'White'
            self.day_time = [0, 0, 0]
        else:
            self.visual_feats = visual_feats

        self.graphs, self.labels, self.data = [], [], dict()
        self.data['masks'] = []
        self.data['img_path'] = []
        self.data['scenario_name'] = name

        if self.alt == '0':
            # Alternative without grid
            self.dataloader = initialize_alt_0
            self.grid_graph_data = None
        elif self.alt == '1' or self.alt == '2':
            # Alternative with grid
            self.dataloader = initialize_alt_1
            self.grid_graph_data = generate_grid_graph_data(self.alt, self.alt_feats, self.grid_width)
            _, self.all_feats, _ = get_features(self.alt, self.alt_feats)
        else:
            print('Introduce a valid initialize alternative')
            sys.exit(-1)

        super(TrafficDataset, self).__init__(name=name, raw_dir=raw_dir, save_dir=save_dir,
                                             force_reload=force_reload, verbose=verbose)

    # ###############################################################
    # Implementation of abstract methods
    #################################################################
    def process(self):
        # process raw data to graphs, labels, splitting masks
        if self.project == 'demo':
            image_list = self.data_demo
        elif self.project == 'SUMMO':
            raw_path = Path(self.raw_dir + '/' + self.mode)
            image_list = (raw_path / 'sumo_boxes').glob('*.*')
            image_list = natsorted(image_list, key=str)
        else:
            raw_path = Path(self.raw_dir + '/' + self.mode)
            image_list = (raw_path / 'images').glob('*.*')
            image_list = natsorted(image_list, key=str)

        # Create the dataset
        for idx, img_path in enumerate(image_list):
            if self.limit is not None and idx >= self.limit:
                print(f'Stop loading graphs. Limit {self.limit} reached.')
                break

            if idx % 100 == 0 and self.project != 'demo':
                print(f'{idx} images loaded.')

            if self.alt == '2':
                if self.project == 'demo':
                    day_time = self.data_demo[0][0].dayTime
                elif self.project == 'SUMMO':
                    day_time = self.day_time
                else:
                    day_time = str(img_path).split('_')
                    day_time = [int(day_time[2]), int(day_time[3]), int(day_time[4].split('.')[0])]

            if self.project == "volumes":
                label, bounding_boxes, image = self._generate_labels(img_path)
            elif self.project == "SPADE":
                label, mask, bounding_boxes, image = self._generate_labels(img_path)
                self.data['masks'].append(mask)
            elif self.project == "demo" or self.project == "SUMMO":
                mask, bounding_boxes, image = self._generate_labels(img_path)
                self.data['masks'].append(mask)
            else:
                print(f'The project name "{self.project}" is not valid.')
                sys.exit(-1)

            image_graph_data = self.dataloader((image, bounding_boxes, self.verbose, self.alt, self.alt_feats,
                                                self.visual_feats))
            if self.grid_graph_data:
                if self.alt == '2':
                    final_graph_data = self._create_final_graph(image_graph_data, day_time)
                else:
                    final_graph_data = self._create_final_graph(image_graph_data)

            else:
                final_graph_data = image_graph_data

            graph = dgl.graph((final_graph_data.src_nodes, final_graph_data.dst_nodes),
                              num_nodes=final_graph_data.n_nodes, idtype=th.int32)
            graph.ndata['h'] = final_graph_data.features
            self.graphs.append(graph)

            if self.project != 'demo':
                self.data['img_path'].append(str(img_path))
                if self.project != 'SUMMO':
                    self.labels.append(label)

        if self.project == 'volumes':
            self.labels = th.tensor(np.array(self.labels), dtype=th.float64)
        elif self.project == 'SPADE':
            self.labels = th.stack(self.labels, dim=0)
            self.data['masks'] = th.stack(self.data['masks'], dim=0)

    def __getitem__(self, idx):
        if self.project == 'SPADE':
            return self.graphs[idx], self.labels[idx], self.data['masks'][idx], self.data['img_path'][idx]
        elif self.project == 'volumes':
            return self.graphs[idx], self.labels[idx]
        elif self.project == 'demo':
            return self.graphs[idx], self.data['masks'][idx]
        elif self.project == 'SUMMO':
            return self.graphs[idx], self.data['masks'][idx], self.data['img_path'][idx]
    def __len__(self):
        return len(self.graphs)

    def save(self):
        if self.debug or self.project == "demo" or self.project == "SUMMO":
            return
        # Generate paths
        graphs_path, info_path = tuple((self.save_dir + '/' + x) for x in self.get_dataset_name())
        os.makedirs(self.save_dir, exist_ok=True)

        # Save graphs and labels
        save_graphs(graphs_path, self.graphs, {'labels': self.labels})

        # Save additional info
        if self.project == 'volumes':
            save_info(info_path, {'img_path': self.data['img_path'],
                                  'scenario_name': self.data['scenario_name']})
        elif self.project == 'SPADE':
            save_info(info_path, {'img_path': self.data['img_path'],
                                  'scenario_name': self.data['scenario_name'],
                                  'masks': self.data['masks']})

    def load(self):
        # Generate paths
        graphs_path, info_path = tuple((self.save_dir + '/' + x) for x in self.get_dataset_name())

        # Load graphs
        self.graphs, label_dict = load_graphs(graphs_path)
        self.labels = label_dict['labels']

        # Load info
        self.data['img_path'] = load_info(info_path)['img_path']
        self.data['scenario_name'] = load_info(info_path)['scenario_name']

        if self.project == 'SPADE':
            self.data['masks'] = load_info(info_path)['masks']

        print(f'LOADED {len(self.graphs)} graphs')

    def get_dataset_name(self):
        if self.limit is None:
            limit = 'None'
        else:
            limit = str(self.limit)
        graphs_path = self.name + '_mode_' + self.mode + '_project_' + self.project + '_alt_' + self.alt + '_feats_' + self.alt_feats + '_limit_' + limit + '_res_' + str(self.resolution) + '.bin'
        info_path = self.name + '_mode_' + self.mode + '_project_' + self.project + '_alt_' + self.alt + '_feats_' + self.alt_feats + '_limit_' + limit + '_res_' + str(self.resolution) + '.info'
        return graphs_path, info_path

    def has_cache(self):
        # Generate paths
        graphs_path, info_path = tuple((self.save_dir + '/' + x) for x in self.get_dataset_name())
        if self.debug:
            return False
        return os.path.exists(graphs_path) and os.path.exists(info_path)

    # ###############################################################
    # Utility methods
    #################################################################

    def _generate_labels(self, img_path):
        # Get the bounding boxes
        if self.project == 'demo':
            bounding_boxes = []
            for d in img_path:
                bounding_boxes.append([d.cls] + d.xywh + [d.conf])
            image = img_path
            path_parts = None
        elif self.project == 'SUMMO':
            with open(img_path) as f:
                bounding_boxes = [line.rstrip().split(' ') for line in f]
            image = img_path
            path_parts = None
        else:
            image = Image.open(str(img_path))
            path_parts = str(img_path).split('/')
            bb_path = self.raw_dir + '/' + self.mode + '/boxes/' + path_parts[-1].split('.')[0] + ".txt"
            with open(bb_path) as f:
                bounding_boxes = [line.rstrip().split(' ') for line in f]

        # Get the label
        if self.project == 'volumes':
            label_path = self.raw_dir + '/' + self.mode + '/volumes/' + path_parts[-1] + ".pt"
            label = (th.load(label_path).squeeze() / 255.).cpu().detach().numpy()
            return label, bounding_boxes, image
        elif self.project == 'SPADE':
            # Label Image (mask)
            mask = get_mask_from_bbs(image.size, bounding_boxes)
            mask = Image.fromarray(mask).convert('RGB')
            transform_mask = get_transform([self.resolution, self.resolution],
                                           method=transforms.InterpolationMode.NEAREST,
                                           normalize=False)
            mask_tensor = transform_mask(mask) * 255.0
            mask_tensor = mask_tensor[0, :, :]
            mask_tensor = mask_tensor[None, :, :]
            mask_tensor[mask_tensor == 255] = len(SPADE_LABELS) + 1  # Plus the background label
            mask = mask_tensor

            # input image (real images)
            label = image.convert('RGB')
            transform_image = get_transform([self.resolution, self.resolution])
            label = transform_image(label)
            return label, mask, bounding_boxes, image
        elif self.project == 'demo' or self.project == 'SUMMO':
            # Label Image (mask)
            mask = get_mask_from_bbs(IMAGE_SIZE, bounding_boxes)
            mask = Image.fromarray(mask).convert('RGB')
            transform_mask = get_transform([self.resolution, self.resolution],
                                           method=transforms.InterpolationMode.NEAREST,
                                           normalize=False)
            mask_tensor = transform_mask(mask) * 255.0
            mask_tensor = mask_tensor[0, :, :]
            mask_tensor = mask_tensor[None, :, :]
            mask_tensor[mask_tensor == 255] = len(SPADE_LABELS) + 1  # Plus the background label
            mask = mask_tensor
            return mask, bounding_boxes, image
        else:
            print(f'The project name "{self.project}" is not valid.')
            sys.exit(-1)

    def _create_final_graph(self, graph_data, day_time=None):
        src_nodes = th.cat([self.grid_graph_data.src_nodes,
                            (graph_data.src_nodes + self.grid_graph_data.n_nodes)], dim=0)
        dst_nodes = th.cat([self.grid_graph_data.dst_nodes,
                            (graph_data.dst_nodes + self.grid_graph_data.n_nodes)], dim=0)

        n_nodes = self.grid_graph_data.n_nodes + graph_data.n_nodes
        features = th.cat([self.grid_graph_data.features, graph_data.features], dim=0)

        # Link each node in the room graph to the correspondent grid graph.
        for n_id in range(0, graph_data.n_nodes):
            x, y = graph_data.features[n_id][0:2].cpu().detach().numpy()
            closest_grid_nodes_id = closest_grid_nodes(self.grid_graph_data.ids, x, y, self.grid_width)

            for g_id in closest_grid_nodes_id:
                if g_id is not None:
                    src_nodes = th.cat([src_nodes, th.tensor([g_id], dtype=th.int32)], dim=0)
                    dst_nodes = th.cat([dst_nodes,
                                        th.tensor([n_id + self.grid_graph_data.n_nodes], dtype=th.int32)], dim=0)

                    src_nodes = th.cat([src_nodes,
                                        th.tensor([n_id + self.grid_graph_data.n_nodes], dtype=th.int32)], dim=0)
                    dst_nodes = th.cat([dst_nodes, th.tensor([g_id], dtype=th.int32)], dim=0)

        if day_time:
            sin, cos = self._from_time_to_degrees(day_time)
            for feat in features:
                feat[self.all_feats.index('DT_sin')] = sin
                feat[self.all_feats.index('DT_cos')] = cos

        return graphData(src_nodes, dst_nodes, n_nodes, features)

    @staticmethod
    def _from_time_to_degrees(time):
        all_seconds = 24.*60.*60.
        c_time = time[0] * 3600 + time[1] * 60 + time[2]
        angle = (c_time/all_seconds) * 2 * math.pi
        return math.sin(angle), math.cos(angle)


if __name__ == '__main__':
    dataset = TrafficDataset('crossroadsRussia', debug=True, project='SPADE', limit=10)
