#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import logging
from torchvision import transforms
import matplotlib.pyplot as plt

# In[2]:


max_epoch_num = 30
mini_batch_size = 10
lambda1 = 100
lambda2 = 10

# In[3]:


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        super(SegmentationDataset, self).__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        # 仅选择有效的图像文件
        self.images = [f for f in sorted(os.listdir(images_dir)) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.masks = [f for f in sorted(os.listdir(masks_dir)) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        #self.images = sorted(os.listdir(images_dir))
        #self.masks = sorted(os.listdir(masks_dir))
        assert len(self.images) == len(self.masks), "图像和掩码数量不匹配"
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        # 读取图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图像文件：{img_path}")
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # [1, H, W]

        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码文件：{mask_path}")
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # [1, H, W]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# In[4]:


class Generator1_CAN8(nn.Module):
    def __init__(self):
        super(Generator1_CAN8, self).__init__()
        chn = 64
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.leakyrelu5 = nn.LeakyReLU(0.2)
        self.leakyrelu6 = nn.LeakyReLU(0.2)
        self.leakyrelu7 = nn.LeakyReLU(0.2)
        self.leakyrelu8 = nn.LeakyReLU(0.2)
        
        # 编码器部分
        self.g1_conv1 = nn.Conv2d(1,     chn,   3, dilation=1, padding=1)
        self.g1_conv2 = nn.Conv2d(chn,   chn,   3, dilation=1, padding=1)
        self.g1_conv3 = nn.Conv2d(chn,   chn*2, 3, dilation=2, padding=2)
        self.g1_conv4 = nn.Conv2d(chn*2, chn*4, 3, dilation=4, padding=4)
        self.g1_conv5 = nn.Conv2d(chn*4, chn*8, 3, dilation=8, padding=8)
        
        # 解码器部分
        self.g1_conv6 = nn.Conv2d(chn*8, chn*4, 3, dilation=4, padding=4)
        self.g1_conv7 = nn.Conv2d(chn*4, chn*2, 3, dilation=2, padding=2)
        self.g1_conv8 = nn.Conv2d(chn*2, chn,   3, dilation=1, padding=1)
        self.g1_conv9 = nn.Conv2d(chn,   1,     1, dilation=1)
        
        # 批归一化层
        self.g1_bn1 = nn.BatchNorm2d(chn)
        self.g1_bn2 = nn.BatchNorm2d(chn)
        self.g1_bn3 = nn.BatchNorm2d(chn*2)
        self.g1_bn4 = nn.BatchNorm2d(chn*4)
        self.g1_bn5 = nn.BatchNorm2d(chn*8)
        self.g1_bn6 = nn.BatchNorm2d(chn*4)
        self.g1_bn7 = nn.BatchNorm2d(chn*2)
        self.g1_bn8 = nn.BatchNorm2d(chn)
    
    def forward(self, input_images):  # 输入[B, 1, 128, 128], 输出[B, 1, 128, 128]
        # 编码器部分
        net = self.g1_conv1(input_images)
        net = self.g1_bn1(net)
        net = self.leakyrelu1(net)
        
        net = self.g1_conv2(net)
        net = self.g1_bn2(net)
        net = self.leakyrelu2(net)
        
        net = self.g1_conv3(net)
        net = self.g1_bn3(net)
        net = self.leakyrelu3(net)
        
        net = self.g1_conv4(net)
        net = self.g1_bn4(net)
        net = self.leakyrelu4(net)
        
        net = self.g1_conv5(net)
        net = self.g1_bn5(net)
        net = self.leakyrelu5(net)
        
        # 解码器部分
        net = self.g1_conv6(net)
        net = self.g1_bn6(net)
        net = self.leakyrelu6(net)
        
        net = self.g1_conv7(net)
        net = self.g1_bn7(net)
        net = self.leakyrelu7(net)
        
        net = self.g1_conv8(net)
        net = self.g1_bn8(net)
        net = self.leakyrelu8(net)
        
        # 最后一层卷积
        output = self.g1_conv9(net)
        
        return output

# In[5]:


class Generator2_UCAN64(nn.Module):
    def __init__(self):
        super(Generator2_UCAN64, self).__init__()
        chn = 64
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.leakyrelu5 = nn.LeakyReLU(0.2)
        self.leakyrelu6 = nn.LeakyReLU(0.2)
        self.leakyrelu7 = nn.LeakyReLU(0.2)
        self.leakyrelu8 = nn.LeakyReLU(0.2)
        self.leakyrelu9 = nn.LeakyReLU(0.2)
        self.leakyrelu10 = nn.LeakyReLU(0.2)
        self.leakyrelu11 = nn.LeakyReLU(0.2)
        self.leakyrelu12 = nn.LeakyReLU(0.2)
        self.leakyrelu13 = nn.LeakyReLU(0.2)
        
        # 编码器部分
        self.g2_conv1 = nn.Conv2d(1,   chn, 3, dilation=1, padding=1)
        self.g2_conv2 = nn.Conv2d(chn, chn, 3, dilation=2, padding=2)
        self.g2_conv3 = nn.Conv2d(chn, chn, 3, dilation=4, padding=4)
        self.g2_conv4 = nn.Conv2d(chn, chn, 3, dilation=8, padding=8)
        self.g2_conv5 = nn.Conv2d(chn, chn, 3, dilation=16, padding=16)
        self.g2_conv6 = nn.Conv2d(chn, chn, 3, dilation=32, padding=32)
        self.g2_conv7 = nn.Conv2d(chn, chn, 3, dilation=64, padding=64)
        self.g2_conv8 = nn.Conv2d(chn, chn, 3, dilation=32, padding=32)
        
        # 解码器部分
        self.g2_conv9 = nn.Conv2d(chn*2, chn, 3, dilation=16, padding=16)
        self.g2_conv10 = nn.Conv2d(chn*2, chn, 3, dilation=8, padding=8)
        self.g2_conv11 = nn.Conv2d(chn*2, chn, 3, dilation=4, padding=4)
        self.g2_conv12 = nn.Conv2d(chn*2, chn, 3, dilation=2, padding=2)
        self.g2_conv13 = nn.Conv2d(chn*2, chn, 3, dilation=1, padding=1)
        self.g2_conv14 = nn.Conv2d(chn, 1,   1, dilation=1)
        
        # 批归一化层
        self.g2_bn1 = nn.BatchNorm2d(chn)
        self.g2_bn2 = nn.BatchNorm2d(chn)
        self.g2_bn3 = nn.BatchNorm2d(chn)
        self.g2_bn4 = nn.BatchNorm2d(chn)
        self.g2_bn5 = nn.BatchNorm2d(chn)
        self.g2_bn6 = nn.BatchNorm2d(chn)
        self.g2_bn7 = nn.BatchNorm2d(chn)
        self.g2_bn8 = nn.BatchNorm2d(chn)
        self.g2_bn9 = nn.BatchNorm2d(chn)
        self.g2_bn10 = nn.BatchNorm2d(chn)
        self.g2_bn11 = nn.BatchNorm2d(chn)
        self.g2_bn12 = nn.BatchNorm2d(chn)
        self.g2_bn13 = nn.BatchNorm2d(chn)
    
    def forward(self, input_images):  # 输入[B, 1, 128, 128], 输出[B, 1, 128, 128]
        # 编码器部分
        net1 = self.g2_conv1(input_images)
        net1 = self.g2_bn1(net1)
        net1 = self.leakyrelu1(net1)
        
        net2 = self.g2_conv2(net1)
        net2 = self.g2_bn2(net2)
        net2 = self.leakyrelu2(net2)
        
        net3 = self.g2_conv3(net2)
        net3 = self.g2_bn3(net3)
        net3 = self.leakyrelu3(net3)
        
        net4 = self.g2_conv4(net3)
        net4 = self.g2_bn4(net4)
        net4 = self.leakyrelu4(net4)
        
        net5 = self.g2_conv5(net4)
        net5 = self.g2_bn5(net5)
        net5 = self.leakyrelu5(net5)
        
        net6 = self.g2_conv6(net5)
        net6 = self.g2_bn6(net6)
        net6 = self.leakyrelu6(net6)
        
        net7 = self.g2_conv7(net6)
        net7 = self.g2_bn7(net7)
        net7 = self.leakyrelu7(net7)
        
        net8 = self.g2_conv8(net7)
        net8 = self.g2_bn8(net8)
        net8 = self.leakyrelu8(net8)
        
        # 拼接 net6 和 net8
        net9 = torch.cat([net6, net8], dim=1)  # [B, 128, 32, 32]
        
        net9 = self.g2_conv9(net9)
        net9 = self.g2_bn9(net9)
        net9 = self.leakyrelu9(net9)
        
        # 拼接 net5 和 net9
        net10 = torch.cat([net5, net9], dim=1)  # [B, 128, 32, 32]
        
        net10 = self.g2_conv10(net10)
        net10 = self.g2_bn10(net10)
        net10 = self.leakyrelu10(net10)
        
        # 拼接 net4 和 net10
        net11 = torch.cat([net4, net10], dim=1)  # [B, 128, 32, 32]
        
        net11 = self.g2_conv11(net11)
        net11 = self.g2_bn11(net11)
        net11 = self.leakyrelu11(net11)
        
        # 拼接 net3 和 net11
        net12 = torch.cat([net3, net11], dim=1)  # [B, 128, 32, 32]
        
        net12 = self.g2_conv12(net12)
        net12 = self.g2_bn12(net12)
        net12 = self.leakyrelu12(net12)
        
        # 拼接 net2 和 net12
        net13 = torch.cat([net2, net12], dim=1)  # [B, 128, 32, 32]
        
        net13 = self.g2_conv13(net13)
        net13 = self.g2_bn13(net13)
        net13 = self.leakyrelu13(net13)
        
        # 最后一层卷积
        net14 = self.g2_conv14(net13)
        
        return net14

# In[6]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.Tanh1 = nn.Tanh()
        self.Tanh2 = nn.Tanh()
        self.Softmax = nn.Softmax(dim=1)
        
        # 卷积层
        self.d_conv1 = nn.Conv2d(2,  24, 3, dilation=1, padding=1)
        self.d_conv2 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv3 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv4 = nn.Conv2d(24, 1,  3, dilation=1, padding=1)
        
        # 批归一化层
        self.d_bn1 = nn.BatchNorm2d(24)
        self.d_bn2 = nn.BatchNorm2d(24)
        self.d_bn3 = nn.BatchNorm2d(24)
        self.d_bn4 = nn.BatchNorm2d(1)
        self.d_bn5 = nn.BatchNorm2d(128)
        self.d_bn6 = nn.BatchNorm2d(64)
        self.d_bn7 = nn.BatchNorm2d(3)
        
        # 全连接层
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
    
    def forward(self, input_images):  # 输入[3B, 2, 128, 128], 输出[B, 1, 128, 128]
        # 最大池化层，减少空间维度
        net = F.max_pool2d(input_images, kernel_size=[2,2])  # [3B, 2, 64, 64]
        net = F.max_pool2d(net, kernel_size=[2,2])           # [3B, 2, 32, 32]
        
        # 卷积层和激活函数
        net = self.d_conv1(net)
        net = self.d_bn1(net)
        net = self.leakyrelu1(net)
        
        net = self.d_conv2(net)
        net = self.d_bn2(net)
        net = self.leakyrelu2(net)
        
        net = self.d_conv3(net)
        net = self.d_bn3(net)
        net = self.leakyrelu3(net)
        
        net = self.d_conv4(net)
        net = self.d_bn4(net)
        net1 = self.leakyrelu4(net)  # [3B, 1, 32, 32]
        
        # 展平操作
        net = net1.view(-1, 1024)  # [3B, 1024]
        
        # 全连接层1
        net = self.fc1(net)         # [3B, 128]
        net = net.unsqueeze(2).unsqueeze(3)  # [3B, 128, 1, 1]
        net = self.d_bn5(net)
        net = self.Tanh1(net)       # [3B, 128, 1, 1]
        
        # 展平操作
        net = net.view(-1, 128)     # [3B, 128]
        
        # 全连接层2
        net = self.fc2(net)         # [3B, 64]
        net = net.unsqueeze(2).unsqueeze(3)  # [3B, 64, 1, 1]
        net = self.d_bn6(net)
        net = self.Tanh2(net)       # [3B, 64, 1, 1]
        
        # 展平操作
        net = net.view(-1, 64)      # [3B, 64]
        
        # 全连接层3
        net = self.fc3(net)         # [3B, 3]
        net = net.unsqueeze(2).unsqueeze(3)  # [3B, 3, 1, 1]
        net = self.d_bn7(net)
        net = self.Softmax(net)     # [3B, 3, 1, 1]
        net = net.squeeze(3).squeeze(2)      # [3B, 3]
        
        # 分割输出
        realscore0, realscore1, realscore2 = torch.split(net, mini_batch_size, dim=0)
        feat0, feat1, feat2 = torch.split(net1, mini_batch_size, dim=0)
        
        # 特征距离
        featDist = torch.mean(torch.pow(feat1 - feat2, 2))
    
        return realscore0, realscore1, realscore2, featDist

# In[7]:


def create_logger(log_file):
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

# In[8]:


def checkpoint_state(model, optimizer, epoch, it):
    return {
        'epoch': epoch,
        'it': it,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

# In[9]:


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# In[10]:


def calculateF1Measure(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = target.astype(np.float32)
    TP = np.sum((pred_binary == 1) & (target_binary == 1))
    FP = np.sum((pred_binary == 1) & (target_binary == 0))
    FN = np.sum((pred_binary == 0) & (target_binary == 1))
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

# In[11]:


def main():
    # 保存输出的总路径
    root_result_dir = os.path.join('pytorch_outputs', 'results')
    os.makedirs(root_result_dir, exist_ok=True)
    metrics_dir = os.path.join(root_result_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    #os.makedirs(os.path.join(root_result_dir, 'models'), exist_ok=True)

    # 当前时间，日志文件的后缀
    time_suffix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    # 日志文件
    log_file = os.path.join(root_result_dir, f'log_train_g1g2_{time_suffix}.txt')
    logger = create_logger(log_file)
    # 定义dataset
    trainsplit = SegmentationDataset(
        images_dir='/root/autodl-fs/data/train/image',
        masks_dir='/root/autodl-fs/data/train/mask',
    )
    trainset = DataLoader(
        trainsplit,
        batch_size=mini_batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
    testsplit = SegmentationDataset(
        images_dir='/root/autodl-fs/data/MDvsFA_test/image',
        masks_dir='/root/autodl-fs/data/MDvsFA_test/mask',
    )
    testset_MDvsFA = DataLoader(
        testsplit,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
        drop_last=True
    )
    testsplit = SegmentationDataset(
        images_dir='/root/autodl-fs/data/SIRST_test/image',
        masks_dir='/root/autodl-fs/data/SIRST_test/mask',
    )
    testset_Sirst = DataLoader(
        testsplit,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
        drop_last=True
    )

    # 定义3个Model
    g1 = Generator1_CAN8().cuda()
    g2 = Generator2_UCAN64().cuda()
    dis = Discriminator().cuda()

    # 定义3个优化器
    optimizer_g1 = optim.Adam(g1.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_g2 = optim.Adam(g2.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(dis.parameters(), lr=1e-5, betas=(0.5, 0.999))

    # 定义loss
    loss1 = nn.BCEWithLogitsLoss()
    it = 0
    for epoch in range(0, max_epoch_num):
        # 调整学习率
        if (epoch + 1) % 10 == 0:
            for p in optimizer_g1.param_groups:
                p['lr'] *= 0.2
            for q in optimizer_g2.param_groups:
                q['lr'] *= 0.2
            for r in optimizer_d.param_groups:
                r['lr'] *= 0.2
            logger.info(f"Adjusted learning rates at epoch {epoch + 1}")

        # 训练一个周期
        logger.info(f'Now we are training epoch {epoch + 1}!')
        total_it_per_epoch = len(trainset)
        for bt_idx, data in enumerate(trainset):
            # 训练一个batch
            torch.cuda.empty_cache()  # 释放之前占用的缓存
            it += 1

            #  训练判别器
            dis.train() 
            g1.eval()
            g2.eval()
            optimizer_d.zero_grad()
            
            with torch.no_grad():
                # 将输入输出放到cuda上
                input_images, output_images = data  # [B, 1, 128, 128], [B, 1, 128, 128]
                input_images = input_images.cuda(non_blocking=True).float()
                output_images = output_images.cuda(non_blocking=True).float()
                
                # 生成器生成假图像
                g1_out = g1(input_images)  # [B, 1, 128, 128]
                g1_out = torch.clamp(g1_out, 0.0, 1.0)
                
                g2_out = g2(input_images)  # [B, 1, 128, 128]
                g2_out = torch.clamp(g2_out, 0.0, 1.0)

            # 准备判别器输入
            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1)  # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1)          # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1)          # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0)              # [3*B, 2, 128, 128]

            # 判别器前向传播
            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input)  # [B, 3], [B, 3], [B, 3], [1]

            # 真实标签为1，假标签为0
            gen_gt  = torch.cat([
                torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float(), 
                torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float(), 
                torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()
            ], dim=1)
            gen_gt1 = torch.cat([
                torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float(), 
                torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float(), 
                torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()
            ], dim=1)
            gen_gt2 = torch.cat([
                torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float(), 
                torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float(), 
                torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float()
            ], dim=1)

            # 计算损失
            ES0 = torch.mean(loss1(logits_real, gen_gt))
            ES1 = torch.mean(loss1(logits_fake1, gen_gt1))
            ES2 = torch.mean(loss1(logits_fake2, gen_gt2))
            disc_loss = ES0 + ES1 + ES2
            #logger.info(f"Discriminator loss: {disc_loss.item()}")
            disc_loss.backward() # 误差反向传播
            optimizer_d.step() # 更新参数

            #  训练生成器 G1
            dis.eval() 
            g1.train()
            g2.eval()
            optimizer_g1.zero_grad()

            # 生成器1生成假图像
            g1_out = g1(input_images)  # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)

            # 计算损失
            MD1 = torch.mean(torch.mul(torch.pow(g1_out - output_images, 2), output_images))
            FA1 = torch.mean(torch.mul(torch.pow(g1_out - output_images, 2), 1 - output_images))
            MF_loss1 = lambda1 * MD1 + FA1 

            # 准备判别器输入
            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1)  # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1)          # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1)          # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0)              # [3*B, 2, 128, 128]

            # 判别器前向传播
            with torch.no_grad():
                logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input)  # [B, 3], [B, 3], [B, 3], [1]

            # 生成器1的损失
            gen_adv_loss1 = torch.mean(loss1(logits_fake1, gen_gt))
            gen_loss1  = lambda1 * MF_loss1 + lambda2 * gen_adv_loss1 + Lgc
            #logger.info(f"Generator1 loss: {gen_loss1.item()}")

            # 反向传播和优化
            gen_loss1.backward()
            optimizer_g1.step()

            #  训练生成器 G2
            dis.eval() 
            g1.eval()
            g2.train()
            optimizer_g2.zero_grad()

            # 生成器2生成假图像
            g2_out = g2(input_images)  # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            # 计算损失
            MD2 = torch.mean(torch.mul(torch.pow(g2_out - output_images, 2), output_images))
            FA2 = torch.mean(torch.mul(torch.pow(g2_out - output_images, 2), 1 - output_images))
            MF_loss2 = MD2 + lambda2 * FA2

            # 准备判别器输入
            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1)  # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1)          # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1)          # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0)              # [3*B, 2, 128, 128]

            # 判别器前向传播
            with torch.no_grad():
                logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input)  # [B, 3], [B, 3], [B, 3], [1]

            # 生成器2的损失
            gen_adv_loss2 = torch.mean(loss1(logits_fake2, gen_gt))
            gen_loss2  = lambda1 * MF_loss2 + lambda2 * gen_adv_loss2 + Lgc
            #logger.info(f"Generator2 loss: {gen_loss2.item()}")

            # 反向传播和优化
            gen_loss2.backward()
            optimizer_g2.step()

            val_loss_g1_list = []
            val_false_ratio_g1_list = []
            val_detect_ratio_g1_list = []
            val_F1_g1_list = []

            val_loss_g2_list = []
            val_false_ratio_g2_list = []
            val_detect_ratio_g2_list = []
            val_F1_g2_list = []

            val_loss_g3_list = []
            val_false_ratio_g3_list = []
            val_detect_ratio_g3_list = []
            val_F1_g3_list = []

            val_loss_g1_list_sirst = []
            val_false_ratio_g1_list_sirst = []
            val_detect_ratio_g1_list_sirst = []
            val_F1_g1_list_sirst = []

            val_loss_g2_list_sirst = []
            val_false_ratio_g2_list_sirst = []
            val_detect_ratio_g2_list_sirst = []
            val_F1_g2_list_sirst = []

            val_loss_g3_list_sirst = []
            val_false_ratio_g3_list_sirst = []
            val_detect_ratio_g3_list_sirst = []
            val_F1_g3_list_sirst = []
            
            # 每100个批次进行一次测试
            if (bt_idx + 1) % 100 == 0:
                # 在测试集上测试
                sum_val_loss_g1 = 0
                sum_val_false_ratio_g1 = 0 
                sum_val_detect_ratio_g1 = 0
                sum_val_F1_g1 = 0
                #g1_time = 0
                
                sum_val_loss_g2 = 0
                sum_val_false_ratio_g2 = 0 
                sum_val_detect_ratio_g2 = 0
                sum_val_F1_g2 = 0
                #g2_time = 0
                
                sum_val_loss_g3 = 0
                sum_val_false_ratio_g3 = 0 
                sum_val_detect_ratio_g3 = 0
                sum_val_F1_g3 = 0

                #testset_MDvsFA测试集
                for bt_idx_test, data in enumerate(testset_MDvsFA):
                    dis.eval() 
                    g1.eval()
                    g2.eval()
                    optimizer_g1.zero_grad()
                    optimizer_g2.zero_grad()
                    optimizer_d.zero_grad()
                    # 将输入输出放到cuda上
                    input_images, output_images = data  # [1, 1, 128, 128], [1, 1, 128, 128]
                    input_images = input_images.cuda(non_blocking=True).float()
                    output_images = output_images.cuda(non_blocking=True).float()
                    # 生成器1输出
                    #stime = time.time()
                    g1_out = g1(input_images)  # [1, 1, 128, 128]
                    #etime = time.time()
                    #g1_time += etime - stime
                    #logger.info(f'testing {bt_idx_test}, g1 time is {etime - stime}')
                    g1_out = torch.clamp(g1_out, 0.0, 1.0)
                    # 生成器2输出
                    #stime = time.time()
                    g2_out = g2(input_images)  # [1, 1, 128, 128]
                    #etime = time.time()
                    #g2_time += etime - stime
                    #logger.info(f'testing {bt_idx_test}, g2 time is {etime - stime}')
                    g2_out = torch.clamp(g2_out, 0.0, 1.0)
                    # 生成器融合输出
                    g3_out = (g1_out + g2_out) / 2  # 取均值的方式进行融合
                    # 转换为numpy
                    output_images_np = output_images.cpu().numpy()
                    g1_out_np = g1_out.detach().cpu().numpy()
                    g2_out_np = g2_out.detach().cpu().numpy()
                    g3_out_np = g3_out.detach().cpu().numpy()
                    # 计算损失和指标
                    val_loss_g1 = np.mean(np.square(g1_out_np - output_images_np))
                    sum_val_loss_g1 += val_loss_g1
                    val_false_ratio_g1 = np.mean(np.maximum(0, g1_out_np - output_images_np))
                    sum_val_false_ratio_g1 += val_false_ratio_g1
                    val_detect_ratio_g1 = np.sum(g1_out_np * output_images_np) / np.maximum(np.sum(output_images_np), 1)
                    sum_val_detect_ratio_g1 += val_detect_ratio_g1
                    val_F1_g1 = calculateF1Measure(g1_out_np, output_images_np, 0.5)
                    sum_val_F1_g1 += val_F1_g1
                    # 计算生成器2的指标
                    val_loss_g2 = np.mean(np.square(g2_out_np - output_images_np))
                    sum_val_loss_g2 += val_loss_g2
                    val_false_ratio_g2 = np.mean(np.maximum(0, g2_out_np - output_images_np))
                    sum_val_false_ratio_g2 += val_false_ratio_g2
                    val_detect_ratio_g2 = np.sum(g2_out_np * output_images_np) / np.maximum(np.sum(output_images_np), 1)
                    sum_val_detect_ratio_g2 += val_detect_ratio_g2
                    val_F1_g2 = calculateF1Measure(g2_out_np, output_images_np, 0.5)
                    sum_val_F1_g2 += val_F1_g2
                    # 计算融合生成器的指标
                    val_loss_g3 = np.mean(np.square(g3_out_np - output_images_np))
                    sum_val_loss_g3 += val_loss_g3
                    val_false_ratio_g3 = np.mean(np.maximum(0, g3_out_np - output_images_np))
                    sum_val_false_ratio_g3 += val_false_ratio_g3
                    val_detect_ratio_g3 = np.sum(g3_out_np * output_images_np) / np.maximum(np.sum(output_images_np), 1)
                    sum_val_detect_ratio_g3 += val_detect_ratio_g3
                    val_F1_g3 = calculateF1Measure(g3_out_np, output_images_np, 0.5)
                    sum_val_F1_g3 += val_F1_g3

                # 记录每100个批次的结果
                testset_MDvsFA_len=len(testset_MDvsFA)
                
                val_loss_g1_list.append(sum_val_loss_g1 / testset_MDvsFA_len)
                val_false_ratio_g1_list.append(sum_val_false_ratio_g1 / testset_MDvsFA_len)
                val_detect_ratio_g1_list.append(sum_val_detect_ratio_g1 / testset_MDvsFA_len)
                val_F1_g1_list.append(sum_val_F1_g1 / testset_MDvsFA_len)

                val_loss_g2_list.append(sum_val_loss_g2 / testset_MDvsFA_len)
                val_false_ratio_g2_list.append(sum_val_false_ratio_g2 / testset_MDvsFA_len)
                val_detect_ratio_g2_list.append(sum_val_detect_ratio_g2 / testset_MDvsFA_len)
                val_F1_g2_list.append(sum_val_F1_g2 / testset_MDvsFA_len)

                val_loss_g3_list.append(sum_val_loss_g3 / testset_MDvsFA_len)
                val_false_ratio_g3_list.append(sum_val_false_ratio_g3 / testset_MDvsFA_len)
                val_detect_ratio_g3_list.append(sum_val_detect_ratio_g3 / testset_MDvsFA_len)
                val_F1_g3_list.append(sum_val_F1_g3 / testset_MDvsFA_len)

                sum_val_loss_g1_sirst = 0
                sum_val_false_ratio_g1_sirst = 0
                sum_val_detect_ratio_g1_sirst = 0
                sum_val_F1_g1_sirst = 0

                sum_val_loss_g2_sirst = 0
                sum_val_false_ratio_g2_sirst = 0
                sum_val_detect_ratio_g2_sirst = 0
                sum_val_F1_g2_sirst = 0

                sum_val_loss_g3_sirst = 0
                sum_val_false_ratio_g3_sirst = 0
                sum_val_detect_ratio_g3_sirst = 0
                sum_val_F1_g3_sirst = 0

                #testset_Sirst测试集
                for bt_idx_test, data in enumerate(testset_Sirst):
                    dis.eval() 
                    g1.eval()
                    g2.eval()
                    optimizer_g1.zero_grad()
                    optimizer_g2.zero_grad()
                    optimizer_d.zero_grad()

                    input_images, output_images = data  # [1, 1, 128, 128], [1, 1, 128, 128]
                    input_images = input_images.cuda(non_blocking=True).float()
                    output_images = output_images.cuda(non_blocking=True).float()

                    # 生成器1输出
                    g1_out = g1(input_images)  
                    g1_out = torch.clamp(g1_out, 0.0, 1.0)

                    # 生成器2输出
                    g2_out = g2(input_images)  
                    g2_out = torch.clamp(g2_out, 0.0, 1.0)

                    # 生成器融合输出
                    g3_out = (g1_out + g2_out) / 2 

                    # 转换为numpy
                    output_images_np = output_images.cpu().numpy()
                    g1_out_np = g1_out.detach().cpu().numpy()
                    g2_out_np = g2_out.detach().cpu().numpy()
                    g3_out_np = g3_out.detach().cpu().numpy()

                    # 计算损失和指标
                    val_loss_g1 = np.mean(np.square(g1_out_np - output_images_np))
                    sum_val_loss_g1_sirst += val_loss_g1
                    val_false_ratio_g1 = np.mean(np.maximum(0, g1_out_np - output_images_np))
                    sum_val_false_ratio_g1_sirst += val_false_ratio_g1
                    val_detect_ratio_g1 = np.sum(g1_out_np * output_images_np) / np.maximum(np.sum(output_images_np), 1)
                    sum_val_detect_ratio_g1_sirst += val_detect_ratio_g1
                    val_F1_g1 = calculateF1Measure(g1_out_np, output_images_np, 0.5)
                    sum_val_F1_g1_sirst += val_F1_g1

                    # 计算生成器2的指标
                    val_loss_g2 = np.mean(np.square(g2_out_np - output_images_np))
                    sum_val_loss_g2_sirst += val_loss_g2
                    val_false_ratio_g2 = np.mean(np.maximum(0, g2_out_np - output_images_np))
                    sum_val_false_ratio_g2_sirst += val_false_ratio_g2
                    val_detect_ratio_g2 = np.sum(g2_out_np * output_images_np) / np.maximum(np.sum(output_images_np), 1)
                    sum_val_detect_ratio_g2_sirst += val_detect_ratio_g2
                    val_F1_g2 = calculateF1Measure(g2_out_np, output_images_np, 0.5)
                    sum_val_F1_g2_sirst += val_F1_g2

                    # 计算融合生成器的指标
                    val_loss_g3 = np.mean(np.square(g3_out_np - output_images_np))
                    sum_val_loss_g3_sirst += val_loss_g3
                    val_false_ratio_g3 = np.mean(np.maximum(0, g3_out_np - output_images_np))
                    sum_val_false_ratio_g3_sirst += val_false_ratio_g3
                    val_detect_ratio_g3 = np.sum(g3_out_np * output_images_np) / np.maximum(np.sum(output_images_np), 1)
                    sum_val_detect_ratio_g3_sirst += val_detect_ratio_g3
                    val_F1_g3 = calculateF1Measure(g3_out_np, output_images_np, 0.5)
                    sum_val_F1_g3_sirst += val_F1_g3

                # 计算testset_Sirst的平均值
                testset_Sirst_len = len(testset_Sirst)

                val_loss_g1_list_sirst.append(sum_val_loss_g1_sirst / testset_Sirst_len)
                val_false_ratio_g1_list_sirst.append(sum_val_false_ratio_g1_sirst / testset_Sirst_len)
                val_detect_ratio_g1_list_sirst.append(sum_val_detect_ratio_g1_sirst / testset_Sirst_len)
                val_F1_g1_list_sirst.append(sum_val_F1_g1_sirst / testset_Sirst_len)

                val_loss_g2_list_sirst.append(sum_val_loss_g2_sirst / testset_Sirst_len)
                val_false_ratio_g2_list_sirst.append(sum_val_false_ratio_g2_sirst / testset_Sirst_len)
                val_detect_ratio_g2_list_sirst.append(sum_val_detect_ratio_g2_sirst / testset_Sirst_len)
                val_F1_g2_list_sirst.append(sum_val_F1_g2_sirst / testset_Sirst_len)

                val_loss_g3_list_sirst.append(sum_val_loss_g3_sirst / testset_Sirst_len)
                val_false_ratio_g3_list_sirst.append(sum_val_false_ratio_g3_sirst / testset_Sirst_len)
                val_detect_ratio_g3_list_sirst.append(sum_val_detect_ratio_g3_sirst / testset_Sirst_len)
                val_F1_g3_list_sirst.append(sum_val_F1_g3_sirst / testset_Sirst_len)
                
                # 记录并打印验证结果
                logger.info("======================== G1 results ============================")
                avg_val_loss_g1 = sum_val_loss_g1 / testset_MDvsFA_len
                avg_val_false_ratio_g1  = sum_val_false_ratio_g1 / testset_MDvsFA_len
                avg_val_detect_ratio_g1 = sum_val_detect_ratio_g1 / testset_MDvsFA_len
                avg_val_F1_g1 = sum_val_F1_g1 / testset_MDvsFA_len
                logger.info(f"Val L2 Loss G1: {avg_val_loss_g1:.4f}")
                logger.info(f"False Alarm Rate G1: {avg_val_false_ratio_g1:.4f}")
                logger.info(f"Detection Rate G1: {avg_val_detect_ratio_g1:.4f}")
                logger.info(f"F1 Measure G1: {avg_val_F1_g1:.4f}")
                #logger.info(f"G1 Time: {g1_time:.4f} seconds")
                logger.info("======================== G2 results ============================")
                avg_val_loss_g2 = sum_val_loss_g2 / testset_MDvsFA_len
                avg_val_false_ratio_g2  = sum_val_false_ratio_g2 / testset_MDvsFA_len
                avg_val_detect_ratio_g2 = sum_val_detect_ratio_g2 / testset_MDvsFA_len
                avg_val_F1_g2 = sum_val_F1_g2 / testset_MDvsFA_len
                logger.info(f"Val L2 Loss G2: {avg_val_loss_g2:.4f}")
                logger.info(f"False Alarm Rate G2: {avg_val_false_ratio_g2:.4f}")
                logger.info(f"Detection Rate G2: {avg_val_detect_ratio_g2:.4f}")
                logger.info(f"F1 Measure G2: {avg_val_F1_g2:.4f}")
                #logger.info(f"G2 Time: {g2_time:.4f} seconds")
                logger.info("======================== G3 results ============================")
                avg_val_loss_g3 = sum_val_loss_g3 / testset_MDvsFA_len
                avg_val_false_ratio_g3  = sum_val_false_ratio_g3 / testset_MDvsFA_len
                avg_val_detect_ratio_g3 = sum_val_detect_ratio_g3 / testset_MDvsFA_len
                avg_val_F1_g3 = sum_val_F1_g3 / testset_MDvsFA_len
                logger.info(f"Val L2 Loss G3: {avg_val_loss_g3:.4f}")
                logger.info(f"False Alarm Rate G3: {avg_val_false_ratio_g3:.4f}")
                logger.info(f"Detection Rate G3: {avg_val_detect_ratio_g3:.4f}")
                logger.info(f"F1 Measure G3: {avg_val_F1_g3:.4f}")

                # 记录并打印验证结果
                logger.info("======================== G1 results for testset_Sirst ============================")
                avg_val_loss_g1_sirst = sum_val_loss_g1_sirst / testset_Sirst_len
                avg_val_false_ratio_g1_sirst = sum_val_false_ratio_g1_sirst / testset_Sirst_len
                avg_val_detect_ratio_g1_sirst = sum_val_detect_ratio_g1_sirst / testset_Sirst_len
                avg_val_F1_g1_sirst = sum_val_F1_g1_sirst / testset_Sirst_len
                logger.info(f"Val L2 Loss G1 (Sirst): {avg_val_loss_g1_sirst:.4f}")
                logger.info(f"False Alarm Rate G1 (Sirst): {avg_val_false_ratio_g1_sirst:.4f}")
                logger.info(f"Detection Rate G1 (Sirst): {avg_val_detect_ratio_g1_sirst:.4f}")
                logger.info(f"F1 Measure G1 (Sirst): {avg_val_F1_g1_sirst:.4f}")
                #logger.info(f"G1 Time (Sirst): {g1_time:.4f} seconds")
                
                logger.info("======================== G2 results for testset_Sirst ============================")
                avg_val_loss_g2_sirst = sum_val_loss_g2_sirst / testset_Sirst_len
                avg_val_false_ratio_g2_sirst = sum_val_false_ratio_g2_sirst / testset_Sirst_len
                avg_val_detect_ratio_g2_sirst = sum_val_detect_ratio_g2_sirst / testset_Sirst_len
                avg_val_F1_g2_sirst = sum_val_F1_g2_sirst / testset_Sirst_len
                logger.info(f"Val L2 Loss G2 (Sirst): {avg_val_loss_g2_sirst:.4f}")
                logger.info(f"False Alarm Rate G2 (Sirst): {avg_val_false_ratio_g2_sirst:.4f}")
                logger.info(f"Detection Rate G2 (Sirst): {avg_val_detect_ratio_g2_sirst:.4f}")
                logger.info(f"F1 Measure G2 (Sirst): {avg_val_F1_g2_sirst:.4f}")
                #logger.info(f"G2 Time (Sirst): {g2_time:.4f} seconds")

                logger.info("======================== G3 results for testset_Sirst ============================")
                avg_val_loss_g3_sirst = sum_val_loss_g3_sirst / testset_Sirst_len
                avg_val_false_ratio_g3_sirst = sum_val_false_ratio_g3_sirst / testset_Sirst_len
                avg_val_detect_ratio_g3_sirst = sum_val_detect_ratio_g3_sirst / testset_Sirst_len
                avg_val_F1_g3_sirst = sum_val_F1_g3_sirst / testset_Sirst_len
                logger.info(f"Val L2 Loss G3 (Sirst): {avg_val_loss_g3_sirst:.4f}")
                logger.info(f"False Alarm Rate G3 (Sirst): {avg_val_false_ratio_g3_sirst:.4f}")
                logger.info(f"Detection Rate G3 (Sirst): {avg_val_detect_ratio_g3_sirst:.4f}")
                logger.info(f"F1 Measure G3 (Sirst): {avg_val_F1_g3_sirst:.4f}")
                #logger.info(f"G3 Time (Sirst): {g3_time:.4f} seconds")

                
                # 保存模型检查点
                #ckpt_name1 = os.path.join(root_result_dir, 'models', f'g1_epoch_{epoch + 1}_batch_{bt_idx + 1}.pth')
                #ckpt_name2 = os.path.join(root_result_dir, 'models', f'g2_epoch_{epoch + 1}_batch_{bt_idx + 1}.pth')
                #ckpt_name3 = os.path.join(root_result_dir, 'models', f'dis_epoch_{epoch + 1}_batch_{bt_idx + 1}.pth')
                #save_checkpoint(checkpoint_state(g1, optimizer_g1, epoch + 1, it), filename=ckpt_name1)
                #save_checkpoint(checkpoint_state(g2, optimizer_g2, epoch + 1, it), filename=ckpt_name2)
                #save_checkpoint(checkpoint_state(dis, optimizer_d, epoch + 1, it), filename=ckpt_name3)
                #logger.info(f"Saved models at epoch {epoch + 1}, batch {bt_idx + 1}")

# In[12]:


if __name__ == '__main__':
    main()
