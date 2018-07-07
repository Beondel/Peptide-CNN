# -*- coding: utf-8 -*-
"""
Created on Tuesday Jun 25 13:34:42 2018

@author: lux32
"""

import torch.nn as nn

class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        # padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, bias=False)

    def forward(self, input):
        output = self.conv(input)
        return output

class CR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        # padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, bias=False)
        # self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        # output = self.bn(output)
        output = self.act(output)
        return output

class P(nn.Module):
    def __init__(self, kSize, stride=2):
        super().__init__()
        # padding = int((kSize - 1) / 2)
        self.pool = nn.MaxPool1d(kSize, stride=stride)

    def forward(self, input):
        output = self.pool(input)
        return output


class shallow_net(nn.Module):
    def __init__(self):
        super().__init__()   # 1 * 30 * 20
        self.level1 = CR(20, 512, 2)  # 1 * 29 * 512
        self.level1_0 = P(2)   # 1 * 14  * 512
        self.level2 = CR(512, 512, 3)  # 1 * 12 * 512
        self.level3_0 = nn.Linear(1 * 12 * 512, 1 * 12 * 512)
        self.level3_1 = nn.Linear(1 * 12 * 512, 400)
        self.level3_2= nn.Linear(400, 1)

    def forward(self, input):
        output = self.level1(input)
        output = self.level1_0(output)
        output = self.level2(output)
        output = self.level3_0(output)
        output = self.level3_1(output)
        output = self.level3_2(output)
        return output