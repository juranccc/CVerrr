import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import weights_init
from utils.callbacks import LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils_fit import fit_one_epoch
import torchvision.models as models
import torch.onnx
from torchsummary import summary
import cv2 as cv
if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda = True
    #-------------------------------#
    #   训练自己的数据集必须要修改的
    #   自己需要的分类个数+1，如2+1
    #-------------------------------#
    num_classes = 2
    #-------------------------------#
    #   所使用的的主干网络：
    #   mobilenet、xception
    #-------------------------------#
    backbone    = "xception"
    #-------------------------------------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   预训练权重对于99%的情况都必须要用，不用的话权值太过随机，特征提取效果不明显
    #   网络训练的结果也不会好，数据的预训练权重对不同数据集是通用的，因为特征是通用的
    #------------------------------------------------------------------------------------#
    model_path  = "model_data/ep148-loss0.038-val_loss0.031.pth"
    #-------------------------------#
    #   下采样的倍数8、16
    #   8要求更大的显存
    #-------------------------------#
    downsample_factor   = 16
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    input_shape         = [512, 512]
    #----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    Freeze_lr           = 5e-4
    #----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 5e-5
    #------------------------------#
    #   数据集路径
    #------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------#
    dice_loss       = False
    #--------------------------------------------------------------------------------------------#
    #   主干网络预训练权重的使用，这里的权值部分仅仅代表主干，下方的model_path代表整个模型的权值
    #   如果想从主干开始训练，可以把这里的pretrained=True，下方model_path的部分注释掉
    #--------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    #------------------------------------------------------#
    Freeze_Train    = True
    #------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0
    #------------------------------------------------------#
    num_workers     = 4
    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    if not pretrained:
        weights_init(model)

    print('Load weights {}.'.format(model_path))
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    features = []
    def hook(module, input, output):
        features.append(output.clone().detach())

    x = cv.imread('img/1_1_new.jpg').reshape(1,3,512,512)
    x = torch.from_numpy(x)
    x = torch.tensor(x, dtype=torch.float32)
    handle = model.backbone.conv2.register_forward_hook(hook)
    y = model(x)
    print(x)