import numpy as np
from torch import nn

from collections import OrderedDict

import torch

from copy import deepcopy

## Define LeNet model 
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Learner(nn.Module):
    def __init__(self, n_outputs, n_tasks, channels=[20, 50, 800, 500]):
        super(Learner, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]

        channel1, channel2, channel3, channel4 = channels
        
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, channel1, 5, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(3)        
        self.map.append(s)
        self.conv2 = nn.Conv2d(channel1, channel2, 5, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(channel1)        
        self.smid=s
        self.map.append(channel2*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(3,2,padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)
        self.lrn = torch.nn.LocalResponseNorm(4,0.001/9.0,0.75,1)

        self.fc1 = nn.Linear(channel2*self.smid*self.smid,channel3, bias=False)
        self.fc2 = nn.Linear(channel3,channel4, bias=False)
        self.map.extend([channel3])
        
        self.fc3 = torch.nn.Linear(channel4, n_outputs,bias=False)

        self.multi_head = True
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

        x=x.reshape(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(x))

        y = self.fc3(x)
            
        return y