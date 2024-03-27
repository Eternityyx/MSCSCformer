#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import h5py
import torch.utils.data as data
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import spectral as spy
import torch.nn as nn


class my_dataset(data.Dataset):
    def __init__(self, mat_data):
        gt_set = mat_data['gt'][...]
        gt_set = np.transpose(gt_set, (3, 0, 1, 2))
        pan_set = mat_data['pan'][...]
        pan_set = np.transpose(pan_set, (2, 0, 1))
        pan_set = pan_set[:, np.newaxis, :, :]
        ms_set = mat_data['ms'][...]
        ms_set = np.transpose(ms_set, (3, 0, 1, 2))
        lms_set = mat_data['usms'][...]
        lms_set = np.transpose(lms_set, (3, 0, 1, 2))

        # 将图片的H和W改为32的倍数，32为2的下采样次幂
        if gt_set.shape[2] % 8 != 0:
            gt_set = gt_set[:, :, :-1, :]
        if gt_set.shape[3] % 8 != 0:
            gt_set = gt_set[:, :, :, :-1]
        if pan_set.shape[2] % 8 != 0:
            pan_set = pan_set[:, :, :-1, :]
        if pan_set.shape[3] % 8 != 0:
            pan_set = pan_set[:, :, :, :-1]
        if ms_set.shape[2] % 8 != 0:
            ms_set = ms_set[:, :, :-1, :]
        if ms_set.shape[3] % 8 != 0:
            ms_set = ms_set[:, :, :, :-1]
        if lms_set.shape[2] % 8 != 0:
            lms_set = lms_set[:, :, :-1, :]
        if lms_set.shape[3] % 8 != 0:
            lms_set = lms_set[:, :, :, :-1]

        self.gt_set = np.array(gt_set, dtype=np.float32) / 1.
        self.pan_set = np.array(pan_set, dtype=np.float32) / 1.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 1.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 1.

    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return gt, pan, lms, ms

    def __len__(self):
        return self.gt_set.shape[0]

class my_full_dataset(data.Dataset):
    def __init__(self, mat_data):
        pan_set = mat_data['pan'][...]
        pan_set = np.transpose(pan_set, (2, 0, 1))
        pan_set = pan_set[:, np.newaxis, :, :]
        ms_set = mat_data['ms'][...]
        ms_set = np.transpose(ms_set, (3, 0, 1, 2))
        lms_set = mat_data['usms'][...]
        lms_set = np.transpose(lms_set, (3, 0, 1, 2))

        # 将图片的H和W改为32的倍数，32为2的下采样次幂
        if pan_set.shape[2] % 8 != 0:
            pan_set = pan_set[:, :, :-1, :]
        if pan_set.shape[3] % 8 != 0:
            pan_set = pan_set[:, :, :, :-1]
        if ms_set.shape[2] % 8 != 0:
            ms_set = ms_set[:, :, :-1, :]
        if ms_set.shape[3] % 8 != 0:
            ms_set = ms_set[:, :, :, :-1]
        if lms_set.shape[2] % 8 != 0:
            lms_set = lms_set[:, :, :-1, :]
        if lms_set.shape[3] % 8 != 0:
            lms_set = lms_set[:, :, :, :-1]

        self.pan_set = np.array(pan_set, dtype=np.float32) / 1.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 1.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 1.

    def __getitem__(self, index):
        pan = self.pan_set[index, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return pan, lms, ms

    def __len__(self):
        return self.pan_set.shape[0]

class new_dataset(data.Dataset):
    def __init__(self, mat_data):
        gt_set = mat_data['gt'][...]
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.gt_set = np.array(gt_set, dtype=np.float32) / 2047.
        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047.

        # self.gt_set = (np.array(gt_set, dtype=np.float32) + 1) / 2048.
        # self.pan_set = (np.array(pan_set, dtype=np.float32) + 1) / 2048.
        # self.ms_set = (np.array(ms_set, dtype=np.float32) + 1) / 2048.
        # self.usms_set = F.interpolate(self.ms_set, scale_factor=4, mode='bilinear')
        # self.lms_set = (np.array(lms_set, dtype=np.float32) + 1) / 2048.


    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        # usms = self.usms_set[index, :, :, :]
        # self.lms_set[self.lms_set <= 0] = 1 / 2048
        lms = self.lms_set[index, :, :, :]
        return gt, pan, lms, ms

    def __len__(self):
        # return self.gt_set.shape[0]
        return 4

class new_full_dataset(data.Dataset):
    def __init__(self, mat_data):
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047.

        # self.pan_set = (np.array(pan_set, dtype=np.float32) + 1) / 2048.
        # self.ms_set = (np.array(ms_set, dtype=np.float32) + 1) / 2048.
        # self.usms_set = F.interpolate(self.ms_set, scale_factor=4, mode='bilinear')
        # self.lms_set = (np.array(lms_set, dtype=np.float32) + 1) / 2048.

    def __getitem__(self, index):
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        # usms = self.usms_set[index, :, :, :]
        # usms = F.interpolate(self.lms_set[index, :, :, :], scale_factor=4, mode='bilinear')
        lms = self.lms_set[index, :, :, :]
        return pan, lms, ms

    def __len__(self):
        # return self.pan_set.shape[0]
        return 4


class gf_dataset(data.Dataset):
    def __init__(self, mat_data):
        gt_set = mat_data['gt'][...]
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.gt_set = np.array(gt_set, dtype=np.float32) / 1023.
        self.pan_set = np.array(pan_set, dtype=np.float32) / 1023.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 1023.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 1023.

    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return gt, pan, lms, ms

    def __len__(self):
        return self.gt_set.shape[0]

class gf_full_dataset(data.Dataset):
    def __init__(self, mat_data):
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.pan_set = np.array(pan_set, dtype=np.float32) / 1023.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 1023.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 1023.

    def __getitem__(self, index):
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return pan, lms, ms

    def __len__(self):
        return self.pan_set.shape[0]


if __name__ == "__main__":
    validation_data_name = '../../disk3/xmm/dataset/training/train_qb.h5'  # your data path
    validation_data = h5py.File(validation_data_name, 'r')
    validation_dataset = new_dataset(validation_data)
    del validation_data
    data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)
    for index, item in enumerate(data_loader):
        print(item[0].max(), item[0].min())
        print(item[1].max(), item[1].min())
        # usms = F.interpolate(item[3], scale_factor=4, mode='bilinear')
        # print(usms.max(), usms.min())
        print(item[3].max(), item[3].min())


        # item[0]=item[0].permute(0,2,3,1)
        # item[1] = item[1].permute(0, 2, 3, 1)
        # item[2] = item[2].permute(0, 2, 3, 1)
        # item[3] = item[3].permute(0, 2, 3, 1)
        # print(item[0].size())
        # print(item[1].size())
        # print(item[2].size())
        # print(item[3].size())
        # view1=spy.imshow(data=item[0][0,:,:,:].numpy(),bands=(0,1,2),title='gt')
        # view2=spy.imshow(data=item[1][0,:,:,:].numpy(),title='pan')
        # view3=spy.imshow(data=item[2][0,:,:,:].numpy(),bands=(0,1,2),title='lms')
        # view4=spy.imshow(data=item[3][0,:,:,:].numpy(),bands=(0,1,2),title='ms')

        # plt.subplot(2, 2, 1)
        # plt.imshow(item[0][0,:,:,:])
        # plt.title('ground truth')
        # plt.subplot(2, 2, 2)
        # plt.imshow(item[1][0,:,:,:])
        # plt.title('pan image')
        # plt.subplot(2, 2, 3)
        # plt.imshow(item[2][0,:,:,:])
        # plt.title('lms image')
        # plt.subplot(2, 2, 4)
        # plt.imshow(item[3][0,:,:,:])
        # plt.title('ms image')
        # plt.show()
        if index==10:break
