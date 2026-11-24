import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

class Net(nn.Module):
    fcs=[]
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,5,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(5,7,5)
        self.conv3=nn.Conv2d(7,10,5)
        self.conv4=nn.Conv2d(10,13,5)
        self.conv5=nn.Conv2d(13,16,5)
        self.fcs.append(nn.Linear(16*128*196,500000))
        self.fcs.append(nn.Linear(500000,600000))
        self.fcs.append(nn.Linear(600000,650000))
        self.fcs.append(nn.Linear(650000,675000))
        self.fcs.append(nn.Linear(675000,700000))
        self.fcs.append(nn.Linear(700000,725000))
        self.fcs.append(nn.Linear(725000,750000))
        self.fcs.append(nn.Linear(750000,775000))
        self.fcs.append(nn.Linear(775000,800000))
        self.fcs.append(nn.Linear(800000,825000))
        self.fcs.append(nn.Linear(825000,850000))
        self.fcs.append(nn.Linear(850000,825000))
        self.fcs.append(nn.Linear(825000,800000))
        self.fcs.append(nn.Linear(800000,786432))
    def forward(self,x):
        x=self.pool(nn.functional.relu(self.conv1(x)))
        x=nn.functional.relu(self.conv2(x))
        x=self.pool(nn.functional.relu(self.conv3(x)))
        x=nn.functional.relu(self.conv4(x))
        x=self.pool(nn.functional.relu(self.conv5(x)))
        x=x.view(-1,16*128*196)
        for fc in self.fcs[:-1]:
            x=nn.functional.relu(fc(x))
        x=self.fcs[-1](x)
        return x
    