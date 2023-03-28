import ipdb
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from copy import deepcopy

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Learner(nn.Module):
    def __init__(self,n_outputs, n_tasks, channels=[64, 128, 256, 2048], copy_model=None):
        super(Learner, self).__init__()
        self.act=OrderedDict()

        channel1, channel2, channel3, channel4 = channels

        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, channel1, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(channel1, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(channel1, channel2, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(channel2, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(channel1)
        self.map.append(s)
        self.conv3 = nn.Conv2d(channel2, channel3, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(channel3, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(channel2)
        self.map.append(channel3*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(channel3*self.smid*self.smid,channel4, bias=False)
        self.bn4 = nn.BatchNorm1d(channel4, track_running_stats=False)
        self.fc2 = nn.Linear(channel4,channel4, bias=False)
        self.bn5 = nn.BatchNorm1d(channel4, track_running_stats=False)
        self.map.extend([channel4])
        
        self.fc3=torch.nn.Linear(channel4,n_outputs,bias=False)

        if copy_model != None:
            self.add_unit(copy_model)

        self.multi_head = True
        self.n_tasks = n_tasks
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.contiguous().view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        try:
            x = self.drop2(self.relu(self.bn4(x)))
        except:
            ipdb.set_trace()

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        
        y=self.fc3(x)
            
        return y

    def add_unit(self, model):
        in_channel, out_channel, _, _ = model.conv1.weight.size()
        (self.conv1.weight[:, :out_channel]).data.copy_(model.conv1.weight.data)

        in_channel, out_channel, _, _ = model.conv2.weight.size()
        (self.conv2.weight[:in_channel, :out_channel]).data.copy_(model.conv2.weight.data)

        in_channel, out_channel, _, _ = model.conv3.weight.size()
        (self.conv3.weight[:in_channel, :out_channel]).data.copy_(model.conv3.weight.data)

        out_dim, in_dim = model.fc1.weight.size()
        (self.fc1.weight[:out_dim, :in_dim]).data.copy_(model.fc1.weight.data)

        out_dim, in_dim = model.fc1.weight.size()
        (self.fc1.weight[:out_dim, :in_dim]).data.copy_(model.fc1.weight.data)

        jj = 0
        for t,n in self.taskcla:
            out_dim, in_dim = model.fc3[jj].weight.size()
            (self.fc3[jj].weight[:, :in_dim]).data.copy_(model.fc3[jj].weight.data)
            jj += 1