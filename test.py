# from choose_dataset import DatasetChooser
# from ST_HRN import ST_HRN
# from HMR import HMR
# from MS_STHRN import MS_STHRN
# from torch.utils.data import DataLoader
# import torch
# import utils
# import torch.optim as optim
# import os
# from utils import Progbar
# from loss import loss as Loss
# import numpy as np
# import datetime
# import scipy.io as sio
# from plot_animation import plot_animation
# import config
# from argparse import ArgumentParser
# from torchviz import make_dot
# from tensorboardX import SummaryWriter
# def choose_net(config):
#
#     if config.model == 'ST_HRN':
#         net = ST_HRN(config)
#     elif config.model == 'HMR':
#         net = HMR(config)
#     elif config.model == 'MS_STHRN':
#         net = MS_STHRN(config)
#     return net
#
# if __name__ == '__main__':
#     gpu = [1]
#     dataset = 'Human'
#     action = 'walking'
#     datatype = 'lie'
#     training = True
#     visualize = False
#     config = config.TrainConfig(dataset, datatype, action, gpu, training, visualize)
#     choose = DatasetChooser(config)
#     train_dataset, bone_length = choose(train=True)
#     train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
#
#     device = torch.device("cuda:"+str(config.device_ids[0]) if torch.cuda.is_available() else "cpu")
#     print('Device {} will be used to save parameters'.format(device))
#     torch.cuda.manual_seed_all(112858)
#
#     net = choose_net(config)
#     net.to(device)
#
#
#     for i, data in enumerate(train_loader, 0):
#         encoder_inputs = data['encoder_inputs'].float().to(device)  ## 前t-1帧
#         decoder_inputs = data['decoder_inputs'].float().to(device)  ## t-1到t-1+output_window_size帧
#         decoder_outputs = data['decoder_outputs'].float().to(device)  ## t帧到以后的
#         prediction = net(encoder_inputs, decoder_inputs, train=True)
#         print(prediction)
#
#         g = make_dot(prediction)
#         g.render('net', view=False)
#         break



#########################

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter  # 用于进行可视化
from torchviz import make_dot
class modelViz(nn.Module):
    def __init__(self):
        super(modelViz, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 10, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        return x



if __name__  == "__main__":
    # 首先来搭建一个模型
    modelviz = modelViz()
    # 创建输入
    sampledata = torch.rand(1, 3, 4, 4)
    # 看看输出结果对不对
    out = modelviz(sampledata)
    print(out)  # 测试有输出，网络没有问题
    g = make_dot(out)
    g.render('modelviz', view=False)  # 这种方式会生成一个pdf文件