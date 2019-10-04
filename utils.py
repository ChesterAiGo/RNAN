
from torch import nn, optim
from torch.nn import init
from torch.utils import data
from math import exp, pow
from torch.autograd import Variable
from time import time
import numpy as np
import torch
import math
import torch.nn.functional as F


# calculate PSNR based on MSE value
def calculate_psnr(MSE):
    return 20 * math.log10(1.0 / MSE)


# define the SSIM loss
class SSIM_loss(torch.nn.Module):
    def __init__(self):
        super(SSIM_loss, self).__init__()
        # window size, c1 and c2 are directly taken from the paper
        self.win_size, self.n_channel, self.sigma = 11, 3, 1.5
        self.c1, self.c2 = 0.01 ** 2, 0.03 ** 2
        self.window = self.make_window()

    # make sliding window of SSIM
    def make_window(self):
        G = []
        for x in range(self.win_size):
            top = -(x - self.win_size // 2) ** 2
            bottom = float(2 * self.sigma ** 2)
            G.append(exp(top / bottom))

        G = torch.Tensor(G)

        # normalize
        win_1d = G / G.sum()

        # change dimension
        win_1d = win_1d.unsqueeze(1)

        win_2d = win_1d.mm(win_1d.t()).float().unsqueeze(0).unsqueeze(0)
        win_final = Variable(win_2d.expand(self.n_channel, 1, self.win_size, self.win_size).contiguous())

        return win_final

    def forward(self, I1, I2):
        # send to GPU
        self.window = self.window.cuda(I1.get_device())

        # feed images through window
        mu_1 = F.conv2d(I1, self.window, padding=self.win_size // 2, groups=self.n_channel)
        mu_2 = F.conv2d(I2, self.window, padding=self.win_size // 2, groups=self.n_channel)

        # calculate factors
        mu_1_sq = mu_1.pow(2)
        mu_2_sq = mu_2.pow(2)
        mu_12 = mu_1 * mu_2
        sigma1_sq = F.conv2d(I1 * I1, self.window, padding=self.win_size // 2, groups=self.n_channel) - mu_1_sq
        sigma2_sq = F.conv2d(I2 * I2, self.window, padding=self.win_size // 2, groups=self.n_channel) - mu_2_sq
        sigma12 = F.conv2d(I1 * I2, self.window, padding=self.win_size // 2, groups=self.n_channel) - mu_12

        # calculate two terms in the fraction
        top = (2 * mu_12  + self.c1) * (2 * sigma12 + self.c2)
        bottom = (mu_1_sq + mu_2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2)
        return (top/bottom).mean()


# this function was used to re-arrange the data into the structure that our code uses
def rearrange_data():
    import os

    # define paths
    src_path = '/Users/ChesterAiGo/Desktop/All_Files/Projects/Ongoing_Projects/ELEC5306/Assignment2/ELEC5306_DATA/'
    train_list = src_path + 'temp_sep_trainlist.txt'
    val_list = src_path + 'temp_sep_validationlist.txt'
    src_imgs = src_path + "vimeo_part_crop/"
    q40_imgs = src_path + "vimeo_part_q40_crop/"
    q37_imgs = src_path + "input_vimeo_part_hevc_qp37_crop/"

    dst_path = 'data/test_data_q40/'

    # global counter as index of all images
    glb_cnt = 0

    # use carefully
    with open(val_list, 'r') as f:
        content = [x.strip() for x in f.readlines()]

        # acquire all possible folders
        for each_folder in content:
            folder_path = q40_imgs + each_folder + "/"

            # acquire all images
            for each_img in os.listdir(folder_path):
                if('png' in each_img or 'jpg' in each_img):

                    # move image
                    os.rename(folder_path + each_img, dst_path + str(glb_cnt) + "_" + each_img)
                    glb_cnt += 1