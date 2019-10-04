

try: from utils import *
except: from Assignment2.Ours.utils import *

import math
import torch
from torch import nn
from torch.nn import functional as F

# Define the RNAN network
class Network(nn.Module):
    def __init__(self, depth, n_maps, n_channel):
        super(Network, self).__init__()

        conv = self.base_conv
        kernel_size = 3

        # build layers
        self.conv1 = conv(n_channel, n_maps, kernel_size)
        self.conv2 = conv(n_maps, n_channel, kernel_size)
        self.nl_res1 = nl_res_module(conv, n_maps, kernel_size)
        self.res_layers_list = [res_att_module(conv, n_maps, kernel_size) for x in range(depth - 2)]
        self.res_layers_list.append(conv(n_maps, n_maps, kernel_size))
        self.res_layers = nn.Sequential(*self.res_layers_list)
        self.nl_res2 = nl_res_module(conv, n_maps, kernel_size)

    # define a basic type of conv as in paper
    def base_conv(self, _in, _out, kernel_size):
        return nn.Conv2d(_in, _out, kernel_size, padding=(kernel_size // 2), bias=True)

    def forward(self, x):
        # residual mask_branch
        res = self.nl_res1(self.conv1(x))
        res = self.res_layers(res)
        res = self.nl_res2(res)
        # the image only after a conv layer
        src = self.conv2(res)
        # return the restored image
        return x + src


# Define the basic residual module
class base_res(nn.Module):
    def __init__(self, conv, n_maps, kernel_size):

        super(base_res, self).__init__()
        self.conv1 = conv(n_maps, n_maps, kernel_size)
        self.relu1 = nn.ReLU(True)
        self.conv2 = conv(n_maps, n_maps, kernel_size)

    def forward(self, x):
        res = self.relu1(self.conv1(x))
        res = self.conv2(res)
        res = res.mul(1) # keep original resolution

        return x + res

# Trunk branch as proposed
class trunk_branch(nn.Module):
    def __init__(self, conv, n_maps, kernel_size):
        super(trunk_branch, self).__init__()
        self.res1 = base_res(conv, n_maps, kernel_size)
        self.res2 = base_res(conv, n_maps, kernel_size)

    def forward(self, x):
        return self.res2(self.res1(x))


# Mask branch as proposed
class mask_branch(nn.Module):
    def __init__(self, conv, n_maps, kernel_size):
        super(mask_branch, self).__init__()

        self.res1 = base_res(conv, n_maps, kernel_size)
        self.conv1 = nn.Conv2d(n_maps, n_maps, 3, stride=2, padding=1)
        self.res2_1 = base_res(conv, n_maps, kernel_size)
        self.res2_2 = base_res(conv, n_maps, kernel_size)
        self.conv_t1 = nn.ConvTranspose2d(n_maps, n_maps, 6, stride=2, padding=2)
        self.res3 = base_res(conv, n_maps, kernel_size)
        self.nin_conv = nn.Conv2d(n_maps, n_maps, 1, padding=0, bias=True)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        res1_out = self.res1(x)
        conv1_out = self.conv1(res1_out)
        res2_out = self.res2_2(self.res2_1(conv1_out))
        convt1_out = self.conv_t1(res2_out)
        res_sum = res1_out + convt1_out
        res3_out = self.res3(res_sum)

        return self.sig(self.nin_conv(res3_out))

# Residual module with attention
class res_att_module(nn.Module):
    def __init__(self, conv, n_maps, kernel_size):
        super(res_att_module, self).__init__()
        # define layers
        self.res1 = base_res(conv, n_maps, kernel_size)
        self.t_branch1 = (trunk_branch(conv, n_maps, kernel_size))
        self.m_branch1 = (mask_branch(conv, n_maps, kernel_size))
        self.res2 = base_res(conv, n_maps, kernel_size)
        self.res3 = base_res(conv, n_maps, kernel_size)
        self.conv = conv(n_maps, n_maps, kernel_size)

    def forward(self, x):
        out_r_1 = self.res1(x)
        t_out = self.t_branch1(out_r_1)
        m_out = self.m_branch1(out_r_1)
        product_mt_out = t_out * m_out
        sum_out = product_mt_out + out_r_1
        sum_out = self.res3(self.res2(sum_out))
        sum_out = self.conv(sum_out)
        return sum_out


# Non-local residual module with attention
class nl_res_module(nn.Module):
    def __init__(self, conv, n_maps, kernel_size):
        super(nl_res_module, self).__init__()
        self.res1 = base_res(conv, n_maps, kernel_size)
        self.t_branch1 = (trunk_branch(conv, n_maps, kernel_size))
        self.m_branch1 = (mask_branch(conv, n_maps, kernel_size))
        self.res2 = base_res(conv, n_maps, kernel_size)
        self.res3 = base_res(conv, n_maps, kernel_size)
        self.conv = conv(n_maps, n_maps, kernel_size)

    def forward(self, x):
        out_r_1 = self.res1(x)
        t_out = self.t_branch1(out_r_1)
        m_out = self.m_branch1(out_r_1)
        product_mt_out = t_out * m_out
        sum_out = product_mt_out + out_r_1
        sum_out = self.res3(self.res2(sum_out))
        sum_out = self.conv(sum_out)
        return sum_out



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        n_channel, n_filters = 3, 48

        layers = []

        # insert all layers of discriminator
        # essentially discriminator is a CNN-based encoder
        layers.append(nn.Conv2d(n_channel, n_filters, (4, 6), 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(n_filters, n_filters*2, (4, 6), 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(n_filters*2))
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(n_filters*2, n_filters*4, (4, 6), 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(n_filters*4))
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(n_filters*4, n_filters*8, (4, 6), 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(n_filters*8))
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(n_filters*8, 1, (4, 5), 1, 0, bias=False))
        layers.append(nn.Sigmoid())

        # build model
        self.body = nn.Sequential(* layers)

    def forward(self, x):
        # ff
        x = self.body(x)
        # change shape
        x = x.view(-1, 1).squeeze(1)
        return x

