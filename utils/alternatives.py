import sys

from alive_progress import alive_bar
from utils.traffic_utils import *


# Generate data for a grid of nodes
def generate_grid_graph_data(alt, alt_feats, grid_width):
    # Grid properties
    connectivity = 8  # Connections of each node
    node_ids = np.zeros((grid_width, grid_width), dtype=int)  # Array to store the IDs of each node
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Compute the number of nodes and initialize feature vectors
    n_features, all_features, _ = get_features(alt, alt_feats)
    n_nodes = grid_width ** 2
    features_gridGraph = th.zeros(n_nodes, n_features)

    max_used_id = -1
    for y in range(grid_width):
        for x in range(grid_width):
            max_used_id += 1
            node_id = max_used_id
            node_ids[x][y] = node_id

            # Self edges
            src_nodes.append(node_id)
            dst_nodes.append(node_id)

            if x < grid_width - 1:
                src_nodes.append(node_id)
                dst_nodes.append(node_id + 1)

                if connectivity == 8 and y > 0:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id - grid_width + 1)

            if x > 0:
                src_nodes.append(node_id)
                dst_nodes.append(node_id - 1)

                if connectivity == 8 and y < grid_width - 1:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id + grid_width - 1)

            if y < grid_width - 1:
                src_nodes.append(node_id)
                dst_nodes.append(node_id + grid_width)

                if connectivity == 8 and x < grid_width - 1:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id + grid_width + 1)

            if y > 0:
                src_nodes.append(node_id)
                dst_nodes.append(node_id - grid_width)

                if connectivity == 8 and x > 0:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id - grid_width- 1)

            x_pos = x / (grid_width - 1)
            y_pos = y / (grid_width - 1)
            features_gridGraph[node_id, all_features.index('grid')] = 1
            features_gridGraph[node_id, all_features.index('x_pos')] = x_pos
            features_gridGraph[node_id, all_features.index('y_pos')] = y_pos

    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    return gridData(src_nodes, dst_nodes, n_nodes, features_gridGraph, node_ids)


# ###############################################################
# Graph initialization alternatives
#################################################################
def initialize_alt_0(data):
    # Initialize variables
    image, bounding_boxes, verbose, alt, alt_feats, custom_feats = data

    #  Creation of the graph
    # Compute the number of nodes
    n_nodes = len(bounding_boxes)
    src_nodes, dst_nodes = [], []
    #      bb_size + One hot for classes + image features
    n_features, all_features, _ = get_features(alt, alt_feats)
    features = th.zeros(n_nodes, n_features)

    # Loop over the number of nodes to add features to each node and create edges
    image_xywhs = []
    for node_id in range(n_nodes):
        # Get variables
        cls = bounding_boxes[node_id][0]
        xywh = np.array([float(x) for x in bounding_boxes[node_id][1:-1]])  # Already normalised
        image_xywhs.append(xywh)

        if custom_feats:
            if len(custom_feats) == 1:
                feats_bb = custom_feats
            else:
                feats_bb = custom_feats[node_id]
        else:
            feats_bb = get_features_from_bb(alt_feats, image, xywh, visualize=False)

        # Add the features to the node
        try:
            features[node_id, 0:4] = th.tensor(xywh)
            features[node_id, T_CLASSES.index(int(float(cls))) + 4] = 1.
            features[node_id, len(all_features):] = th.tensor(feats_bb)
        except ValueError as ve:
            if verbose:
                print(f'{YOLO_NAMES[int(float(cls))]} not used, skipping the creation of the node')
            continue

        # Create self edges
        src_nodes.append(node_id)
        dst_nodes.append(node_id)

    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    return graphData(src_nodes, dst_nodes, n_nodes, features)


def initialize_alt_1(data):
    # Initialize variables
    image, bounding_boxes, verbose, alt, alt_feats, custom_feats = data

    #  Creation of the graph
    # Compute the number of nodes
    n_nodes = len(bounding_boxes)
    src_nodes, dst_nodes = [], []
    #      bb_size + One hot for classes + image features
    n_features, all_features, _ = get_features(alt, alt_feats)
    features = th.zeros(n_nodes, n_features)

    # Loop over the number of nodes to add features to each node and create edges
    image_xywhs = []
    for node_id in range(n_nodes):
        # Get variables
        cls = bounding_boxes[node_id][0]
        xywh = np.array([float(x) for x in bounding_boxes[node_id][1:-1]])  # Already normalised
        image_xywhs.append(xywh)

        if custom_feats is not None:
            if isinstance(custom_feats, str):
                feats_bb = np.zeros(len(COLOR_PALETTE))
                feats_bb[list(COLOR_PALETTE.keys()).index(custom_feats)] = 1.
            else:
                feats_bb = custom_feats[node_id]
        else:
            feats_bb = get_features_from_bb(alt_feats, image, xywh, visualize=False)

        # Add the features to the node
        try:
            features[node_id, 0:4] = th.tensor(xywh)
            features[node_id, T_CLASSES.index(int(float(cls))) + 4] = 1.
            features[node_id, len(all_features):] = th.tensor(feats_bb)
        except ValueError as ve:
            if verbose:
                print(f'{YOLO_NAMES[int(float(cls))]} not used, skipping the creation of the node')
            continue

    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    return graphData(src_nodes, dst_nodes, n_nodes, features)
