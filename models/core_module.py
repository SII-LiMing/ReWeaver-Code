import torch.nn as nn
from models.pc_encoder import PctComplexStitch
import torch.nn.functional as F
import torch

import torch
from torch import nn, Tensor
import math


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sin=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sin_activation = sin

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if(self.sin_activation):
              x = layer(x).sin() if i < self.num_layers - 1 else layer(x)
            else:
              x = F.leaky_relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MLP_hn(nn.Module): #hypernets of MLP
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, input_dim_fea):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        h = [hidden_dim] * (num_layers - 1)
        h_plus = [hidden_dim + 1] * (num_layers - 1)
        self.layers_dims = list(zip([input_dim + 1] + h_plus, h + [output_dim]))
        self.layers_size = [a * b for a,b in self.layers_dims]

        self.layer1 = nn.Linear(input_dim_fea, 1024)
        self.layer2 = nn.Linear(1024, sum(self.layers_size))
    
    def forward(self, x, feature):
        net_par = self.layer1(feature)
        net_par = F.relu(net_par)
        net_par = self.layer2(net_par)
        net_par = net_par / math.sqrt(self.hidden_dim)

        net_par_layers = torch.split(net_par, self.layers_size, dim=-1)
        for i in range(len(self.layers_size)):
            layer_par = net_par_layers[i].view(feature.shape[0], feature.shape[1] ,self.layers_dims[i][0], self.layers_dims[i][1])
            layer_par = layer_par.to(x.device)
            x = torch.einsum('...ij,...jk->...ik', x, layer_par[...,:-1,:]) + layer_par[...,-1:,:]
            if i < self.num_layers - 1:
                x = F.leaky_relu(x)
        return x