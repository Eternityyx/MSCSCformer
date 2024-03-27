#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a implementation of training code of this paper:
# FrMLNet: Framelet-based Multi-level Network for Pansharpening
# author: Tingting Wang
"""
from __future__ import print_function

import matplotlib.pyplot as plt
import torch.optim
import torch.utils.data as data
from torch.autograd import Variable
import h5py
from skimage.metrics import peak_signal_noise_ratio as PSNR
from transformer import *
from util import *
from dataset import *
import math
import os
import pickle
import sys
import scipy.io as sio
from sewar.no_ref import qnr
from PIL import Image
import spectral as spy
import importlib
from torch.utils.tensorboard import SummaryWriter
import smtplib
import email
# 负责构造文本
from email.mime.text import MIMEText
# 负责构造图片
from email.mime.image import MIMEImage
# 负责将多个对象集合起来
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import Header


importlib.reload(sys)

## parameters setting and network selection ##
train_bs = 6
val_bs = 6
test_bs = 1
ms_channels = 8
pan_channel = 1
epoch = 700
LR = 0.0001
iter_num = 5
scale = 3
dim = 64
heads = 2
num_blocks = 3


NET = 'transformer'
condition = f'v1_wv3_{LR}_{iter_num}_{scale}_{dim}_{heads}_{num_blocks}'

# condition = f'decent1_qb_k5_{LR}_{iter_nums1}_{iter_nums2}_{alpha}_{beta}'
checkpoint = f'../../../data/zmy/{NET}/checkpoint/{condition}'
# plt_save = f'../../data/zmy/{NET}/output_img/{condition}'
writer = SummaryWriter(f'./log/{condition}')

validRecord = {"epoch": [], "LOSS": [], "PSNR": [], "SAM": [], "ERGAS": []}
testRecord = {"epoch": [], "QNR": []}

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
devicesList = [0, 1, 2]
dtype = torch.cuda.FloatTensor
MAEloss = torch.nn.L1Loss(reduction='mean').type(dtype)
MAE_sumloss = torch.nn.L1Loss(reduction='sum').type(dtype)
MSEloss = torch.nn.MSELoss(reduction='mean').type(dtype)

name = 1

CNN = mscsc(ms_channels=ms_channels, pan_channel=pan_channel, iter_num=iter_num, scale=scale, dim=dim, heads=heads, num_blocks=num_blocks)
# CNN = deepnet(ms_channels=ms_channels, pan_channel=pan_channel, iter_nums1=iter_nums1, iter_nums2=iter_nums2)
total = sum(p.numel() for p in CNN.parameters())
print(f'总参数量为：{total / 1e6:.2f}M')
CNN = nn.DataParallel(CNN, device_ids=devicesList).cuda()

class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

## parameters setting ##

def validation(dataloader):
    global name
    sum_psnr = 0
    sum_sam = 0
    sum_ergas = 0
    sum_loss = 0
    count = 0
    CNN.eval()
    for index, data in enumerate(dataloader):
        count += (data[0].shape[0] / val_bs)

        gtVar = Variable(data[0]).type(dtype)
        panVar = Variable(data[1]).type(dtype)
        lmsVar = Variable(data[2]).type(dtype)
        msVar = Variable(data[3]).type(dtype)

        with torch.no_grad():
            output = CNN(msVar, panVar, lmsVar)
            basic_loss = MAEloss(output, gtVar)
            loss = basic_loss
            sum_loss += loss

        output = output.cpu().data.numpy()

        msVar = msVar.cpu().data.numpy()
        gtLabel_np = gtVar.cpu().data.numpy()
        samValue = SAM(gtLabel_np, output)
        ergasValue = ERGAS(output, gtLabel_np, msVar)
        psnrValue = PSNR(gtLabel_np, output)
        sum_sam += samValue
        sum_psnr += psnrValue
        sum_ergas += ergasValue

    avg_psnr = sum_psnr / count
    avg_sam = sum_sam / count
    avg_ergas = sum_ergas / count
    avg_loss = sum_loss / count

    print(f'val_psnr:{avg_psnr:.4f} val_sam:{avg_sam:.4f} val_ergas:{avg_ergas:.4f} val_loss:{avg_loss:.7f}')
    return avg_psnr, avg_sam, avg_ergas, avg_loss

def test(test_Full_dataloader):
    global name
    sum_qnr = 0
    count = 0
    CNN.eval()

    for index, data in enumerate(test_Full_dataloader):
        count += 1

        panVar = Variable(data[0]).type(dtype)
        lmsVar = Variable(data[1]).type(dtype)
        msVar = Variable(data[2]).type(dtype)

        with torch.no_grad():
            output = CNN(msVar, panVar, lmsVar)
            # output = CNN(panVar, lmsVar)+ lmsVar   ## MSDCNN/AFEFPNN

        output = output.squeeze(0).permute(1, 2, 0).cpu().data.numpy()
        msVar = msVar.squeeze(0).permute(1, 2, 0).cpu().data.numpy()
        panVar = panVar.squeeze(0).squeeze(0).cpu().data.numpy()

        qnrValue = qnr(panVar, msVar, output)
        sum_qnr += qnrValue

    avg_qnr = sum_qnr / count

    print(f'test_qnr:{avg_qnr:.4f}')
    return avg_qnr


if __name__ == "__main__":

    # try:
    # SMTP服务器,这里使用qq邮箱
    mail_host = "smtp.qq.com"
    # 发件人邮箱
    mail_sender = "1806163688@qq.com"
    # 邮箱授权码,注意这里不是邮箱密码
    mail_license = "fvansedoesmfcbej"
    # 收件人邮箱，可以为多个收件人
    mail_receivers = ["yeyongxu_3@163.com"]

    test_max = 0
    val_max = 0
    min = 1e9
    resume_train = False
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    ### read dataset ###
    traindata = 'train_wv3.h5'
    train_data_name = f'../../../data/zmy/dataset/training/{traindata}'
    train_data = h5py.File(train_data_name, 'r')
    train_dataset = new_dataset(train_data)
    trainsetSize = train_data['gt'].shape[0]
    del train_data
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4)

    validationdata = 'valid_wv3.h5'
    validation_data_name = f'../../../data/zmy/dataset/validation/{validationdata}'
    validation_data = h5py.File(validation_data_name, 'r')
    validation_dataset = new_dataset(validation_data)
    del validation_data
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=val_bs, shuffle=True, num_workers=4)

    FullData = 'test_wv3_OrigScale_multiExm2.h5'
    test_Full_data_name = f'../../../data/zmy/dataset/testing/{FullData}'
    test_Full_data = h5py.File(test_Full_data_name, 'r')
    test_Full_dataset = new_full_dataset(test_Full_data)
    del test_Full_data
    test_Full_dataloader = torch.utils.data.DataLoader(test_Full_dataset, batch_size=test_bs, shuffle=True)

    savemat_val_data_name = f'../../../data/zmy/{NET}/WV3_{condition}_val_data.mat'
    savemat_test_data_name = f'../../../data/zmy/{NET}/WV3_{condition}_test_data.mat'
    SaveFullDataPath = f"../../data/zmy/{NET}/{NET}_{condition}_{FullData}"
    savenet_data_name = checkpoint + f'/{{}}_WV3_{condition}_{{}}.pth'

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, CNN.parameters()), lr=LR, betas=(0.9, 0.999))

    weight_decay = 0
    if weight_decay > 0:
        reg_loss = Regularization(CNN, weight_decay, p=2).type(dtype)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for i in range(1, epoch + 1):
        count = 0
        CNN.train()
        for index, data in enumerate(train_dataloader):
            count += data[0].shape[0]
            optimizer.zero_grad()

            gtVar = Variable(data[0]).type(dtype)
            panVar = Variable(data[1]).type(dtype)
            lmsVar = Variable(data[2]).type(dtype)
            msVar = Variable(data[3]).type(dtype)

            output = CNN(msVar, panVar, lmsVar)
            basic_loss = MAEloss(output, gtVar)
            loss = basic_loss

            if loss.item() < min:
                min = loss.item()

            loss.backward()
            optimizer.step()
            # print(f'epoch:{i:04d} [{count:05d}/{trainsetSize:05d}] loss {loss.item():.8f}')
            print(f'epoch:{i:04d} [{count:05d}/{trainsetSize:05d}] basic_loss {(basic_loss.item()):.8f}')
            # torch.save(CNN.state_dict(),'./log_FrResPanNet_QB/FrResPanNet0519_QB16_1_01_01.pth'.format(i))

        if (i) % 2 == 0:
            print("")
            # validation(validation_dataloader)
            # psnr,sam,ergas = validation(test_dataloader)
            val_psnr, val_sam, val_ergas, val_loss = validation(validation_dataloader)
            test_qnr = test(test_Full_dataloader)

            validRecord["epoch"].append(i)
            validRecord["LOSS"].append(val_loss.item())
            validRecord["PSNR"].append(val_psnr)
            validRecord["SAM"].append(val_sam)
            validRecord["ERGAS"].append(val_ergas)
            testRecord["epoch"].append(i)
            testRecord["QNR"].append(test_qnr)
            if test_qnr > test_max:
                test_max = test_qnr
                torch.save(CNN.state_dict(), savenet_data_name.format(NET, 'test_best'))
            if val_psnr > val_max:
                val_max = val_psnr
                torch.save(CNN.state_dict(), savenet_data_name.format(NET, 'val_best'))
            sio.savemat(savemat_val_data_name, validRecord)
            sio.savemat(savemat_test_data_name, testRecord)

            writer.add_scalars(
                main_tag='loss',
                tag_scalar_dict={
                    'train_loss': basic_loss.item(),
                    'val_loss': val_loss.item()
                },
                global_step=i)

            writer.add_scalars(
                main_tag='PSNR',
                tag_scalar_dict={
                    'val_psnr': val_psnr,
                    'test_qnr': test_qnr
                },
                global_step=i)

        torch.save(CNN.state_dict(), savenet_data_name.format(NET, 'newest'))
        scheduler.step()
    # except Exception as e:
    #     error = MIMEMultipart('mixed')
    #
    #     # 邮件主题
    #     subject_content = f"""错误报告"""
    #     # 设置发送者,注意严格遵守格式,里面邮箱为发件人邮箱
    #     error["From"] = "阝栩丶诩☁<1806163688@qq.com>"
    #     # 设置接受者,注意严格遵守格式,里面邮箱为接受者邮箱
    #     error["To"] = "小旭旭<yeyongxu_3@163.com>"
    #     # 设置邮件主题
    #     error["Subject"] = Header(subject_content, 'utf-8')
    #
    #     # 邮件正文内容
    #     body_content = f"{e}"
    #     # 构造文本,参数1：正文内容，参数2：文本格式，参数3：编码方式
    #     message_text = MIMEText(body_content, "plain", "utf-8")
    #     # 向MIMEMultipart对象中添加文本对象
    #     error.attach(message_text)
    #
    #     # 创建SMTP对象
    #     stp = smtplib.SMTP()
    #     # 设置发件人邮箱的域名和端口，端口地址为25
    #     stp.connect(mail_host, 25)
    #     # set_debuglevel(1)可以打印出和SMTP服务器交互的所有信息
    #     stp.set_debuglevel(1)
    #     # 登录邮箱，传递参数1：邮箱地址，参数2：邮箱授权码
    #     stp.login(mail_sender, mail_license)
    #     # 发送邮件，传递参数1：发件人邮箱地址，参数2：收件人邮箱地址，参数3：把邮件内容格式改为str
    #     stp.sendmail(mail_sender, mail_receivers, error.as_string())

    print(f'最大test_psnr为：{val_max}, 最大qnr为：{test_qnr}')

    mm = MIMEMultipart('mixed')

    # 邮件主题
    subject_content = f"""{NET}_{condition}_WV3数据集训练结果"""
    # 设置发送者,注意严格遵守格式,里面邮箱为发件人邮箱
    mm["From"] = "阝栩丶诩☁<1806163688@qq.com>"
    # 设置接受者,注意严格遵守格式,里面邮箱为接受者邮箱
    mm["To"] = "小旭旭<yeyongxu_3@163.com>"
    # 设置邮件主题
    mm["Subject"] = Header(subject_content, 'utf-8')

    # 邮件正文内容
    body_content = f"""论文：《Deep Gradient Projection Networks for Pan-sharpening》https://arxiv.org/abs/2103.05946
    使用训练数据集为：{traindata}
    总参数量为：{total / 1e6:.2f}M
    总训练次数：{epoch:04d}
    训练batch_size为：{train_bs}
    使用学习率为：{LR}
    使用weight_decay为：{weight_decay}
    最大test_psnr为：{max:.4f}
    最小loss为：{min:.8f}
    网络结构为：
    {CNN}"""
    # 构造文本,参数1：正文内容，参数2：文本格式，参数3：编码方式
    message_text = MIMEText(body_content, "plain", "utf-8")
    # 向MIMEMultipart对象中添加文本对象
    mm.attach(message_text)

    # 创建SMTP对象
    stp = smtplib.SMTP()
    # 设置发件人邮箱的域名和端口，端口地址为25
    stp.connect(mail_host, 25)
    # set_debuglevel(1)可以打印出和SMTP服务器交互的所有信息
    stp.set_debuglevel(1)
    # 登录邮箱，传递参数1：邮箱地址，参数2：邮箱授权码
    stp.login(mail_sender, mail_license)
    # 发送邮件，传递参数1：发件人邮箱地址，参数2：收件人邮箱地址，参数3：把邮件内容格式改为str
    stp.sendmail(mail_sender, mail_receivers, mm.as_string())
    print("邮件发送成功")
    # 关闭SMTP对象
    stp.quit()
