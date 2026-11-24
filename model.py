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
        self.fcs.append(nn.Linear(16*64*98,16*64*98))
        self.fcs.append(nn.Linear(16*64*98,14*64*98))
        self.fcs.append(nn.Linear(14*64*98,12*64*98))
        self.fcs.append(nn.Linear(12*64*98,10*64*98))
        self.fcs.append(nn.Linear(10*64*98,7*64*98))
        self.fcs.append(nn.Linear(7*64*98,5*64*98))
        self.fcs.append(nn.Linear(5*64*98,3*64*98))
        self.fcs.append(nn.Linear(3*64*98,2*64*98))
        self.fcs.append(nn.Linear(2*64*98,64*98))
    def forward(self,x):
        x=nn.functional.relu(self.conv1(x))
        x=self.pool(nn.functional.relu(self.conv2(x)))
        x=self.pool(nn.functional.relu(self.conv3(x)))
        x=self.pool(nn.functional.relu(self.conv4(x)))
        x=self.pool(nn.functional.relu(self.conv5(x)))
        x=x.view(-1,16*64*98)
        for fc in self.fcs[:-1]:
            x=nn.functional.relu(fc(x))
        x=self.fcs[-1](x)
        return x
net=Net()
criterion=nn.MSELoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
