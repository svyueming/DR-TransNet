#!/usr/bin/python
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch
import numpy as np



class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True,freeze_layer=False):
        super(conv, self).__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) // 2) * dilation, bias=bias),
            nn.LeakyReLU(0.2, inplace=True))
        if  freeze_layer :
          for layer in self.modules():
             layer.eval()
    def forward(self, x):
        return self.stem(x)
class conv_tanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True,freeze_layer=False):
        super(conv_tanh, self).__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) // 2) * dilation, bias=bias)
                      ,nn.Tanh())
            # nn.LeakyReLU(0.2, inplace=True))
        
        if freeze_layer:
            for layer in self.modules():
                layer.eval()
    def forward(self, x):
        return self.stem(x)
def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True,freeze_layer=False):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias,freeze_layer=freeze_layer)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias,freeze_layer=False):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
        if freeze_layer:
            for layer in self.modules():
                layer.eval()
    def forward(self, x):
        out = self.stem(x) + x
        return out

class MSD_sub(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, dilation, bias):
        super(MSD_sub, self).__init__()
        self.conv1 =  conv(in_channels, out_channels, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  conv(in_channels, out_channels, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  conv(in_channels, out_channels, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  conv(in_channels, out_channels, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  nn.Conv2d(out_channels*4, out_channels, kernel_size=1, stride=1, padding=(1-1)//2, bias=bias)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat)
        return out
class MSDB(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias=True,freeze_layer=False):
        super(MSDB, self).__init__()
        self.conv = nn.Conv2d(in_channels , int(in_channels/4), kernel_size=kernel_size, stride=1,padding=(kernel_size - 1) // 2, bias=bias)
        self.MSD_sub1 =  MSD_sub( int(in_channels/4),int(in_channels/4), kernel_size, dilation, bias)
        self.MSD_sub2 =  MSD_sub( int(in_channels/2),int(in_channels/4), kernel_size, dilation, bias)
        self.MSD_sub3 =  MSD_sub( int((in_channels/4)*3), int(in_channels/4), kernel_size, dilation, bias)
        self.convi =  nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1,padding=(1-1)//2, bias=bias)
        if freeze_layer:
            for layer in self.modules():
                layer.eval()
    def forward(self, x):
        conv    = self.conv(x)#64
        MSD_sub1 = self.MSD_sub1(conv)#64
        cat1 = torch.cat([MSD_sub1, conv], 1)#128
        MSD_sub2 = self.MSD_sub2(cat1)#64
        cat2 = torch.cat([cat1, MSD_sub2], 1)#192
        MSD_sub3 = self.MSD_sub3(cat2)#64
        cat3 = torch.cat([cat2, MSD_sub3], 1)#256
        out = self.convi(cat3) + x
        return out
class connect(nn.Module):
    def __init__(self,feature,r,M=2,L=32,freeze_layer=False):
        super(connect, self).__init__()
        d= max(int(feature / r ),L)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d( feature,d, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.convs= nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(nn.Conv2d(d,feature,kernel_size=1,stride=1,padding=0)))
        self.softmax=nn.Softmax(dim=1)
        if freeze_layer:
            for layer in self.modules():
                layer.eval()
    def forward(self, x,y):
        x = x.unsqueeze_(dim=1)
        y = y.unsqueeze_(dim=1)
        feas  = torch.cat([x,y],1)
        fea_U = torch.sum(feas,dim = 1)
        fea_s = self.gap(fea_U)
        fea_z = self.conv(fea_s)
        fea_z = self.relu(fea_z)
        for i,conv in enumerate(self.convs):
            vector = conv(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors,vector],1)
        attention_vectors =self.softmax(attention_vectors)
        fea_v = (feas*attention_vectors).sum(dim=1)

        return fea_v
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

