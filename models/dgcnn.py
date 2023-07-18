#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    activation layer
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class Conv1dLayer(Seq):
    def __init__(self, channels, act='relu', norm=True, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv1d(channels[i - 1], channels[i], 1, bias=bias))
            if norm:
                m.append(nn.BatchNorm1d(channels[i]))
            if act:
                m.append(act_layer(act))
        super(Conv1dLayer, self).__init__(*m)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature




class DGCNNView(nn.Module):
    def __init__(self, args, dim=512):
        super(DGCNNView, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.bn3 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm1d(self.dim)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim*2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv3 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   act_mod(**act_mod_args))
        self.conv5 = nn.Sequential(nn.Conv1d(self.dim * 3, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))




    def forward(self, x, pos):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(1)

        return x1



class DGCNNViewLight(nn.Module):
    def __init__(self, args, dim=512):
        super(DGCNNViewLight, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm1d(self.dim)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim*2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))

        self.conv5 = nn.Sequential(nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))

        self.bn5 = nn.BatchNorm1d(self.dim)




    def forward(self, x, pos):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(1)

        return x1


class DGCNNViewLight2(nn.Module):
    def __init__(self, args, dim=512):
        super(DGCNNViewLight2, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm1d(self.dim)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim*2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv5 = nn.Sequential(nn.Conv1d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))




    def forward(self, x, pos):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]


        x = torch.cat((x1, x2), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(1)

        return x1


class DGCNNViewMLP(nn.Module):
    def __init__(self, args, dim=512):
        super(DGCNNViewMLP, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.dim = dim
        self.bn5 = nn.BatchNorm1d(self.dim)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv5 = nn.Sequential(nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))

        self.bn5 = nn.BatchNorm1d(self.dim)




    def forward(self, x, pos):
        batch_size = x.size(0)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(1)

        return x1