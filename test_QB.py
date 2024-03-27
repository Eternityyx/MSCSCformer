#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a implementation of validation code of this paper:
# FrMLNet: Framelet-based Multi-level Network for Pansharpening
# author: Tingting Wang
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
import h5py
from transformer import *
from util import *
from dataset import *
import math
import os
import pickle
import sys
import importlib
import scipy.io
import time


importlib.reload(sys)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicesList = [torch.cuda.current_device()]
dtype = torch.cuda.FloatTensor
ms_channels = 4
pan_channel = 1
test_bs = 1
epoch = 700
LR = 0.0001
iter_num = 5
scale = 3
dim = 64
heads = 2
num_blocks = 3
## nc = 4

checkpoint_condition = 'val_best'
NET = 'transformer'
condition = f'v1_qb_{LR}_{iter_num}_{scale}_{dim}_{heads}_{num_blocks}_ps8'

checkpoint = f'../../disk3/xmm/{NET}/checkpoint/{condition}/{NET}_'


if __name__ == "__main__":
    ##### read dataset #####
    test_bs = 1
    tmpPath = checkpoint + f"QB_17139_{condition}_{checkpoint_condition}.pth"
    ReducedData = 'test_qb_multiExm2'

    SaveReducedDataPath = f"../../disk3/xmm/test_result/QB_Reduced/{NET}_{condition}_{checkpoint_condition}.mat"
    # SaveReducedDataPath = f"../../disk3/xmm/{NET}/{NET}_{condition}_reduce.mat"
    test_Reduced_data_name = f'../../disk3/xmm/dataset/testing/{ReducedData}.h5'
    test_Reduced_data = h5py.File(test_Reduced_data_name, 'r')
    test_Reduced_dataset = new_dataset(test_Reduced_data)
    del test_Reduced_data
    test_Reduced_dataloader = torch.utils.data.DataLoader(test_Reduced_dataset, batch_size=test_bs, shuffle=False)

    FullData = 'test_qb_OrigScale_multiExm2'
    SaveFullDataPath = f"../../disk3/xmm/test_result/QB_Full/{NET}_{condition}_{checkpoint_condition}.mat"
    # SaveFullDataPath = f"../../disk3/xmm/{NET}/{NET}_{condition}_full.mat"
    test_Full_data_name = f'../../disk3/xmm/dataset/testing/{FullData}.h5'
    test_Full_data = h5py.File(test_Full_data_name, 'r')
    test_Full_dataset = new_full_dataset(test_Full_data)
    del test_Full_data
    test_Full_dataloader = torch.utils.data.DataLoader(test_Full_dataset, batch_size=test_bs, shuffle=False)

    #fc = 32 ## AFEFPNN
    #CNN =  LapPanNet(nc,fc) ## AFEFPNN
    CNN = mscsc(ms_channels=ms_channels, pan_channel=pan_channel, iter_num=iter_num, scale=scale, dim=dim, heads=heads, num_blocks=num_blocks)
    CNN = nn.DataParallel(CNN, device_ids=devicesList).cuda()

    CNN.load_state_dict(torch.load(tmpPath))
    # print(CNN)
    CNN.eval()
    reduced_count = 0
    for index, data in enumerate(test_Reduced_dataloader):
        gtVar = Variable(data[0]).type(dtype)
        panVar = Variable(data[1]).type(dtype)
        lmsVar = Variable(data[2]).type(dtype)
        msVar = Variable(data[3]).type(dtype)
        with torch.no_grad():
            output = CNN(msVar, panVar, lmsVar)
        #output = CNN(panVar, lmsVar)+ lmsVar   ## MSDCNN/AFEFPNN
        netOutput_np = output.cpu().data.numpy()
        # lms_np = data[2].numpy()
        # ms_np = data[3].numpy()
        # pan_np = data[1].numpy()
        # gt_np = data[0].numpy()
        if reduced_count == 0:
            Output_np = netOutput_np
            # ms = ms_np
            # lms = lms_np
            # pan = pan_np
            # gt = gt_np
        else:
            Output_np = np.concatenate((netOutput_np, Output_np), axis=0)
            # ms = np.concatenate((ms_np, ms), axis=0)
            # lms = np.concatenate((lms_np, lms), axis=0)
            # pan = np.concatenate((pan_np, pan), axis=0)
            # gt = np.concatenate((gt_np, gt), axis=0)
        reduced_count = reduced_count + 1
    scipy.io.savemat(SaveReducedDataPath, {'sr': Output_np})
    #scipy.io.savemat(SaveDataPath,{'QB256':Output_np, 'GT256': gt})

    full_count = 0
    for index, data in enumerate(test_Full_dataloader):
        panVar = Variable(data[0]).type(dtype)
        lmsVar = Variable(data[1]).type(dtype)
        msVar = Variable(data[2]).type(dtype)
        with torch.no_grad():
            output = CNN(msVar, panVar, lmsVar)
        #output = CNN(panVar, lmsVar)+ lmsVar   ## MSDCNN/AFEFPNN
        netOutput_np = output.cpu().data.numpy()
        # lms_np = data[1].numpy()
        # ms_np = data[2].numpy()
        # pan_np = data[0].numpy()
        if full_count == 0:
            Output_np = netOutput_np
            # ms = ms_np
            # lms = lms_np
            # pan = pan_np
        else:
            Output_np = np.concatenate((netOutput_np, Output_np), axis=0)
            # ms = np.concatenate((ms_np, ms), axis=0)
            # lms = np.concatenate((lms_np, lms), axis=0)
            # pan = np.concatenate((pan_np, pan), axis=0)
        full_count = full_count + 1
    scipy.io.savemat(SaveFullDataPath, {'sr': Output_np})
    total = sum(p.numel() for p in CNN.parameters())
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f'{NET}_{condition}_{checkpoint_condition} test qb completely!, 总参数量为：{total / 1e6:.4f}M')
