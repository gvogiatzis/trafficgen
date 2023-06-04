import sys
import dgl
import torch
import signal

from sklearn.metrics import mean_squared_error
import numpy as np

# ###############################################################
# Training utils
#################################################################
# Variables
if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

ctrl_c_counter = 0
stop_training = False


# Functions
def describe_model(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def collate(batch):
    if len(batch[0]) == 4:
        masks = torch.unsqueeze(batch[0][2], dim=0)
        image_paths = [batch[0][3]]
    elif len(batch[0]) == 3:
        image_paths = [batch[0][2]]

    graphs = [batch[0][0]]
    labels = torch.unsqueeze(batch[0][1], dim=0)

    for data in batch[1:]:
        if len(batch[0]) == 4:
            graph, label, mask, img_path = data
            masks = torch.cat([masks, torch.unsqueeze(mask, dim=0)], dim=0)
            image_paths.append(img_path)
        elif len(batch[0]) == 3:
            graph, label, img_path = data
            image_paths.append(img_path)
        else:
            graph, label = data
        graphs.append(graph)
        labels = torch.cat([labels, torch.unsqueeze(label, dim=0)], dim=0)

    batched_graphs = dgl.batch(graphs)

    if len(batch[0]) == 4:
        return batched_graphs, labels, masks, image_paths
    elif len(batch[0]) == 3:
        return batched_graphs, labels, image_paths
    else:
        return batched_graphs, labels


def evaluate(feats, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        output = model(feats.float(), subgraph)

        loss_data = loss_fcn(output.float().to(device), labels.to(device))
        return loss_data.item()


def signal_handler(sig, frame):
    global stop_training
    global ctrl_c_counter
    ctrl_c_counter += 1
    if ctrl_c_counter == 3:
        stop_training = True
    if ctrl_c_counter >= 5:
        sys.exit(-1)
    print('If you press Ctr+c  3 times we will stop saving the training ({} times)'.format(ctrl_c_counter))
    print('If you press Ctr+c >5 times we will stop NOT saving the training ({} times)'.format(ctrl_c_counter))
