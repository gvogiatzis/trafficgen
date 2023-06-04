import sys

import torch


def count_parameters(m):
    params = 0
    for key in m.keys():
        params += m[key].numel()
    return params


if __name__ == "__main__":
    model_G = torch.load(sys.argv[1] + '/latest_net_G.pth')
    model_D = torch.load(sys.argv[1] + '/latest_net_G.pth')

    params_G = count_parameters(model_G)
    params_D = count_parameters(model_D)
    print(f'Number of generator parameters: {params_G}')
    print(f'Number of discriminator parameters: {params_D}')
    print(f'Number of total parameters: {params_G + params_D}')


