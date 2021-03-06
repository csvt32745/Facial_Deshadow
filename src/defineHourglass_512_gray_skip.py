from email.mime import base
import torch
from torch.autograd import Variable
from torch.cuda import init
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time

# we define Hour Glass network based on the paper
# Stacked Hourglass Networks for Human Pose Estimation
#       Alejandro Newell, Kaiyu Yang, and Jia Deng
# the code is adapted from
# https://github.com/umich-vl/pose-hg-train/blob/master/src/models/hg.lua


def conv3X3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)
# define the network
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, batchNorm_type=0, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # batchNorm_type 0 means batchnormalization
        #                1 means instance normalization
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = conv3X3(inplanes, outplanes, 1)
        self.conv2 = conv3X3(outplanes, outplanes, 1)
        if batchNorm_type == 0:
            self.bn1 = nn.BatchNorm2d(outplanes)
            self.bn2 = nn.BatchNorm2d(outplanes)
        else:
            self.bn1 = nn.InstanceNorm2d(outplanes)
            self.bn2 = nn.InstanceNorm2d(outplanes)
        
        self.shortcuts = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.inplanes != self.outplanes:
            out += self.shortcuts(x)
        else:
            out += x
        
        out = F.relu(out)
        return out

class HourglassBlock(nn.Module):
    '''
        define a basic block for hourglass neetwork
            ^-------------------------upper conv-------------------
            |                                                      |
            |                                                      V
        input------>downsample-->low1-->middle-->low2-->upsample-->+-->output
        NOTE about output:
            Since we need the lighting from the inner most layer, 
            let's also output the results from middel layer
    '''
    def __init__(self, inplane, mid_plane, middleNet, skipLayer=True):
        super(HourglassBlock, self).__init__()
        # upper branch
        self.skipLayer = True
        self.upper = BasicBlock(inplane, inplane, batchNorm_type=1)
        
        # lower branch
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        self.low1 = BasicBlock(inplane, mid_plane)
        self.middle = middleNet
        self.low2 = BasicBlock(mid_plane, inplane, batchNorm_type=1)

    def forward(self, x, count, skip_count):
        # we use count to indicate wich layer we are in
        # max_count indicates the from which layer, we would use skip connections
        out_upper = self.upper(x)
        out_lower = self.downSample(x)
        out_lower = self.low1(out_lower)
        if self.middle:
            out_lower, out_bottleneck = self.middle(out_lower, count+1, skip_count)
        else:
            out_bottleneck = out_lower
        out_lower = self.low2(out_lower)
        out_lower = self.upSample(out_lower)

        if count >= skip_count and self.skipLayer:
            # withSkip is true, then we use skip layer
            # easy for analysis
            out = out_lower + out_upper
        else:
            out = out_lower
            #out = out_upper
        return out, out_bottleneck

    def get_bottleneck(self, x, count, skip_count):
        # out_upper = self.upper(x)
        out_lower = self.downSample(x)
        out_lower = self.low1(out_lower)
        return self.middle.get_bottleneck(out_lower, count+1, skip_count) if self.middle \
            else out_lower

class lightingNet(nn.Module):
    '''
        define lighting network
    '''
    def __init__(self, ncInput, ncOutput, ncMiddle):
        super(lightingNet, self).__init__()
        self.ncInput = ncInput
        self.ncOutput = ncOutput
        self.ncMiddle = ncMiddle

        # basic idea is to compute the average of the channel corresponding to lighting
        # using fully connected layers to get the lighting
        # then fully connected layers to get back to the output size

        self.predict_FC1 = nn.Conv2d(self.ncInput,  self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.predict_relu1 = nn.PReLU()
        self.predict_FC2 = nn.Conv2d(self.ncMiddle, self.ncOutput, kernel_size=1, stride=1, bias=False)

        self.post_FC1 = nn.Conv2d(self.ncOutput,  self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.post_relu1 = nn.PReLU()
        self.post_FC2 = nn.Conv2d(self.ncMiddle, self.ncInput, kernel_size=1, stride=1, bias=False)
        self.post_relu2 = nn.ReLU()  # to be consistance with the original feature

    def forward(self, innerFeat, target_light, count, skip_count):
        x = innerFeat[:,0:self.ncInput,:,:] # lighting feature
        _, _, row, col = x.shape

        # predict lighting
        feat = x.mean(dim=(2,3), keepdim=True)
        light = self.predict_relu1(self.predict_FC1(feat))
        light = self.predict_FC2(light)

        # get back the feature space
        upFeat = self.post_relu1(self.post_FC1(target_light))
        upFeat = self.post_relu2(self.post_FC2(upFeat))
        upFeat = upFeat.repeat((1,1,row, col))
        innerFeat[:,0:self.ncInput,:,:] = upFeat
        return innerFeat, light


class HourglassNet(nn.Module):
    '''
    	basic idea: low layers are shared, upper layers are different	
    	            lighting should be estimated from the inner most layer
        NOTE: we split the bottle neck layer into albedo, normal and lighting
    '''
    def __init__(self, ch_in=3, n_out_layers=2, baseFilter = 16, gray=True, ch_out = 0, is_out=True, is_double_feat=True):
        super(HourglassNet, self).__init__()
        if ch_out < 1:
            ch_out = ch_in
        self.ncLight = 27   # number of channels for input to lighting network
        self.baseFilter = baseFilter

        # number of channles for output of lighting network
        if gray:
            self.ncOutLight = 9  # gray: channel is 1
        else:
            self.ncOutLight = 27  # color: channel is 3

        self.ncPre = self.baseFilter  # number of channels for pre-convolution

        # number of channels 
        if is_double_feat:
            self.ncHG3 = self.baseFilter
            self.ncHG2 = 2*self.baseFilter
            self.ncHG1 = 4*self.baseFilter
            self.ncHG0 = 8*self.baseFilter
            # self.ncHG0 = 8*self.baseFilter + self.ncLight
        else:
            self.ncHG3 = self.baseFilter
            self.ncHG2 = self.baseFilter
            self.ncHG1 = self.baseFilter
            self.ncHG0 = self.baseFilter

        self.pre = nn.Sequential(
            nn.Conv2d(ch_in, self.ncPre, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ncPre)
        ) if is_out else nn.Identity()

        # self.light = lightingNet(self.ncLight, self.ncOutLight, 128)
        self.HG0 = HourglassBlock(self.ncHG1, self.ncHG0, None)
        self.HG1 = HourglassBlock(self.ncHG2, self.ncHG1, self.HG0)
        self.HG2 = HourglassBlock(self.ncHG3, self.ncHG2, self.HG1)
        self.HG3 = HourglassBlock(self.ncPre, self.ncHG3, self.HG2)

        self.out_conv = nn.Sequential(
            nn.Conv2d(self.ncPre, self.ncPre, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ncPre),
            nn.ReLU(True),
            *[
                nn.Sequential(
                    nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1),
                    nn.BatchNorm2d(self.ncPre),
                    nn.ReLU(True),
                )
                for i in range(n_out_layers)
            ]
        )

        self.output = nn.Conv2d(self.ncPre, ch_out, kernel_size=1, stride=1, padding=0) if is_out \
            else nn.Identity()
        self.ch_bottleneck = self.ncHG0

    def forward(self, x, skip_count=0):
        feat = F.relu(self.pre(x))
        feat, bottleneck = self.HG3(feat, 0, skip_count)
        
        feat = self.out_conv(feat)
        out_img = self.output(feat)
        # out_img = torch.tanh(out_img)
        out_img = torch.sigmoid(out_img)
        return out_img, bottleneck

    def get_hgfeat(self, x, skip_count=0):
        return self.HG3(x, 0, skip_count)
        # feat = self.out_conv(feat)
        # return feat, bottleneck

    def get_features(self, *args, **kwargs):
        return list(zip(self.forward(*args, **kwargs)))

    def get_bottlenecks(self, x, skip_count=0):
        feat = F.relu(self.pre(x))
        return [self.HG3.get_bottleneck(feat, 0, skip_count)]

class HourglassNetExquotient(HourglassNet):
    def __init__(self, ch_in=3, n_out_layers=2, baseFilter=16, gray=True):
        super().__init__(ch_in=ch_in, ch_out=ch_in*2, n_out_layers=n_out_layers, baseFilter=baseFilter, gray=gray)
        self.ch = ch_in
    
    def forward(self, x, skip_count=0):
        feat = F.relu(self.pre(x))

        feat, bottleneck = self.HG3(feat, 0, skip_count)
        
        feat = self.out_conv(feat)
        gain, bias = torch.split(self.output(feat), (self.ch, self.ch), dim=1)
        return torch.sigmoid(x*gain + bias), bottleneck

    def get_features(self, *args, **kwargs):
        return list(zip(self.forward(*args, **kwargs)))

    def get_bottlenecks(self, x, skip_count=0):
        feat = F.relu(self.pre(x))
        return [self.HG3.get_bottleneck(feat, 0, skip_count)]

class SimpleStackedHourglassNet(nn.Module):
    def __init__(self, n_stacks, ch_in=3, n_out_layers=0, baseFilter = 16, gray=True, net_class=HourglassNet):
        super().__init__()
        self.n_stacks = n_stacks
        self.net = nn.ModuleList([net_class(ch_in, n_out_layers, baseFilter, gray) for i in range(n_stacks)])
        self.ch_bottleneck = self.net[0].ch_bottleneck

    def forward(self, x, skip_count=0):
        out = x
        for n in self.net:
            out, bottleneck = n(out, skip_count)
        return out, bottleneck
    
    def get_features(self, x, skip_count=0):
        feat = x
        out = []
        bottlenecks = []
        for n in self.net:
            feat, bottleneck = n(feat, skip_count)
            out.append(feat)
            bottlenecks.append(bottleneck)
        return out, bottlenecks # n_stacks, B, C, H, W
    
    def get_bottlenecks(self, x, skip_count=0):
        feat = x
        bottlenecks = []
        for n in self.net[:-1]:
            feat, bottleneck = n(feat, skip_count)
            bottlenecks.append(bottleneck)
        bottlenecks += self.net[-1].get_bottlenecks(feat, skip_count)
        return bottlenecks # n_stacks, B, C, H, W

class RecursiveStackedHourglassNet(nn.Module):
    def __init__(self, n_stacks, ch_in=3, n_out_layers=0, baseFilter = 16, gray=True, net_class=HourglassNet):
        super().__init__()
        self.n_stacks = n_stacks
        self.net = net_class(ch_in, n_out_layers, baseFilter, gray)
        self.ch_bottleneck = self.net.ch_bottleneck

    def forward(self, x, skip_count=0):
        out = x
        for i in range(self.n_stacks):
            out, bottleneck = self.net(out, skip_count)
        return out, bottleneck
    
    def get_features(self, x, skip_count=0):
        feat = x
        out = [] 
        bottlenecks = []
        for i in range(self.n_stacks):
            feat, bottleneck = self.net(feat, skip_count)
            out.append(feat)
            bottlenecks.append(bottleneck)
        return out, bottlenecks # n_stacks, B, C, H, W

    def get_bottlenecks(self, x, skip_count=0):
        feat = x
        bottlenecks = []
        for i in range(self.n_stacks-1):
            feat, bottleneck = self.net(feat, skip_count)
            bottlenecks.append(bottleneck)
        bottlenecks += self.net.get_bottleneck(feat, skip_count)
        return bottlenecks # n_stacks, B, C, H, W


class StackedHourglassNet(nn.Module):
    def __init__(self, n_stacks, ch_in=3, baseFilter=64):
        super().__init__()
        self.n_stacks = n_stacks
        self.pre = nn.Sequential(
            nn.Conv2d(ch_in, baseFilter//2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(baseFilter//2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            BasicBlock(baseFilter//2, baseFilter, batchNorm_type=1),
        )

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.hgs = nn.ModuleList([
            HourglassNet(baseFilter, 0, baseFilter, is_out=False, is_double_feat=False) \
            for i in range(n_stacks)])
        
        self.sup2out = nn.ModuleList([
            nn.Sequential(
                self.upsample,
                nn.Conv2d(baseFilter, ch_in, 3, 1, 1),
                nn.Sigmoid()
            )for i in range(n_stacks-1)])

        self.out2feat = nn.ModuleList([nn.Conv2d(ch_in, baseFilter, 3, 2, 1) for i in range(n_stacks-1)])
        self.sup2feat = nn.ModuleList([nn.Conv2d(baseFilter, baseFilter, 1) for i in range(n_stacks-1)])
        self.final = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            BasicBlock(baseFilter, baseFilter//2),
            nn.Conv2d(baseFilter//2, ch_in, 1),
            nn.Sigmoid()
        )

        self.ch_bottleneck = self.hgs[0].ch_bottleneck

    def forward(self, x, skip_count=0):
        feat = self.pre(x)
        for i in range(self.n_stacks-1):
            feat, bottleneck = self.hgs[i].get_hgfeat(feat, skip_count)
            sup_out = self.sup2out[i](feat)
            feat = self.out2feat[i](sup_out) + self.sup2feat[i](feat)
        
        feat, bottleneck = self.hgs[-1].get_hgfeat(feat)
        return self.final(feat), bottleneck
    
    def get_features(self, x, skip_count=0):
        feat = self.pre(x)
        out = []
        bottlenecks = []
        for i in range(self.n_stacks-1):
            feat, bottleneck = self.hgs[i].get_hgfeat(feat, skip_count)
            sup_out = self.sup2out[i](feat)
            out.append(sup_out) # intermediate out
            feat = self.out2feat[i](sup_out) + self.sup2feat[i](feat)
            bottlenecks.append(bottleneck)
        
        feat, bottleneck = self.hgs[-1].get_hgfeat(feat)
        out.append(self.final(feat))
        bottlenecks.append(bottleneck)
        return out, bottlenecks # n_stacks, B, C, H, W
    
    def get_bottlenecks(self, x, skip_count=0):
        feat = self.pre(x)
        bottlenecks = []
        for i in range(self.n_stacks-1):
            feat, bottleneck = self.hgs[i].get_hgfeat(feat, skip_count)
            sup_out = self.sup2out[i](feat)
            feat = self.out2feat[i](sup_out) + self.sup2feat[i](feat)
            bottlenecks.append(bottleneck)
        
        feat, bottleneck = self.hgs[-1].get_hgfeat(feat)
        bottlenecks.append(bottleneck)
        return bottlenecks # n_stacks, B, C, H, W

if __name__ == '__main__':
    a = StackedHourglassNet(2)
    b = torch.ones((3, 3, 256, 256))
    c = a(b)
    print(c[0].shape, c[1].shape)
    c = a.get_bottlenecks(b)
    c = a.get_features(b)
