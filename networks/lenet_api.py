import numpy as np
from torch import nn, Tensor

from collections import OrderedDict
from torch.nn import functional as F

import torch

from copy import deepcopy
from math import sqrt
import ipdb

class api_Conv2d(nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(api_Conv2d,self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias, dilation=dilation, groups=groups, padding_mode=padding_mode)

    def forward(self, input, t, fai=None):
        if fai == None:
            return F.conv2d(input, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            weight = self.weight[:,:input.shape[1]]
            if t:
                Fai = torch.cat(fai[:t],dim=1)
                # weight = weight + torch.mm(self.weight[:,input.shape[1]:input.shape[1]+Fai.shape[1]], Fai)
                if Fai.shape[1]:
                    try:
                        weight = weight + torch.matmul(self.weight[:,input.shape[1]:input.shape[1]+Fai.shape[1]].permute(0, 2, 3, 1), Fai.permute(1,0)).permute(0, 3, 1, 2)
                    except:
                        ipdb.set_trace()
            try:
                out = F.conv2d(input, weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            except:
                ipdb.set_trace()
            return out
        
class aip_Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
    
    def forward(self, input: Tensor, t, fai=None) -> Tensor:
        if fai == None:
            return super().forward(input)
        else:
            weight = self.weight[:,:input.shape[1]]
            if t:
                Fai = torch.cat(fai[:t],dim=1)
                if Fai.shape[1]: weight = weight + torch.mm(self.weight[:,input.shape[1]:input.shape[1]+Fai.shape[1]], Fai.permute(1,0))
            return F.linear(input, weight, bias=self.bias)

## Define LeNet model 
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Learner(nn.Module):
    def __init__(self, n_outputs, n_tasks, channels=[20, 50, 800, 500], expand_size = None, copy_model=None, device='cuda'):
        super(Learner, self).__init__()
        self.act=OrderedDict()

        if expand_size != None:
            assert copy_model != None
            self.expand = copy_model.expand
            expand1, expand2, expand3, expand4 = expand_size
            self.expand[0].append(expand1)
            self.expand[1].append(expand2)
            self.expand[2].append(expand3)
            self.expand[3].append(expand4)

            self.weight1, self.weight2, self.weight3, self.weight4 = [], [], [], []
            for i in range(len(copy_model.weight1)):
                self.weight1.append(torch.zeros(copy_model.weight1[i].size(),device=device).copy_(copy_model.weight1[i].data))
            for i in range(len(copy_model.weight2)):
                self.weight2.append(torch.zeros(copy_model.weight2[i].size(),device=device).copy_(copy_model.weight2[i].data))
            for i in range(len(copy_model.weight3)):
                self.weight3.append(torch.zeros(copy_model.weight3[i].size(),device=device).copy_(copy_model.weight3[i].data))
            for i in range(len(copy_model.weight4)):
                self.weight4.append(torch.zeros(copy_model.weight4[i].size(),device=device).copy_(copy_model.weight4[i].data))
            # self.weight1.append((weights[0]).to(device))
            # self.weight2.append((weights[1]).to(device))
            # self.weight3.append((weights[2]).to(device))
            # self.weight4.append((weights[3]).to(device))
            # self.weight5.append((weights[4]).to(device))
        else:
            self.expand = [[], [], [], []]
            self.weight1, self.weight2, self.weight3, self.weight4 = [], [], [], []


        self.map =[]
        self.ksize=[]
        self.in_channel =[]

        channel1, channel2, channel3, channel4 = channels
        self.channel1, self.channel2, self.channel3, self.channel4 = channels
        
        self.map.append(32)
        self.conv1 = api_Conv2d(3+sum(self.expand[0]), channel1, 5, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(3+sum(self.expand[0]))        
        self.map.append(s)
        self.conv2 = api_Conv2d(channel1+sum(self.expand[1]), channel2, 5, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(channel1+sum(self.expand[1]))        
        self.smid=s
        self.map.append(channel2*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(3,2,padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)
        self.bn1 = torch.nn.BatchNorm2d(channel1, track_running_stats=False)
        self.bn2 = torch.nn.BatchNorm2d(channel2, track_running_stats=False)

        self.fc1 = aip_Linear(channel2*self.smid*self.smid+sum(self.expand[2]),channel3, bias=False)
        self.fc2 = aip_Linear(channel3+sum(self.expand[3]),channel4, bias=False)
        self.map.extend([channel3])
        
        self.fc3 = torch.nn.Linear(channel4, n_outputs,bias=False)

        if copy_model != None:
            self.add_unit(copy_model, device)
        else:
            self.select1, self.select2, self.select3, self.select4 = [], [], [], []

        self.multi_head = True
        
    def forward(self, x, t=0, get_feat=False):
        if get_feat:
            bsz = deepcopy(x.size(0))

            if t != 0:
                x = torch.cat([x] + [torch.matmul(x.permute(0, 2, 3, 1), self.weight1[i]).permute(0, 3, 1, 2) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[0][t:]), x.size(2), x.size(3), device=x.device)], dim=1)
            else:
                x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[0]), x.size(2), x.size(3), device=x.device)], dim=1)
            self.act['conv1']=x

            x = self.conv1(x, t=t)
            x = self.maxpool(self.drop1(self.bn1(self.relu(x))))

            if t != 0:
                x = torch.cat([x] + [torch.matmul(x.permute(0, 2, 3, 1), self.weight2[i]).permute(0, 3, 1, 2) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[1][t:]), x.size(2), x.size(3), device=x.device)], dim=1)
            else:
                x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[1]), x.size(2), x.size(3), device=x.device)], dim=1)
            self.act['conv2']=x

            x = self.conv2(x, t=t)
            x = self.maxpool(self.drop1(self.bn2 (self.relu(x))))

            x=x.reshape(bsz,-1)

            if t != 0:
                x = torch.cat([x] + [torch.mm(x, self.weight3[i]) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[2][t:]), device=x.device)], dim=1)
            else:
                x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[2]), device=x.device)], dim=1)
            self.act['fc1']=x

            x = self.fc1(x, t=t)
            x = self.drop2(self.relu(x))

            if t != 0:
                x = torch.cat([x] + [torch.mm(x, self.weight4[i]) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[3][t:]), device=x.device)], dim=1)
            else:
                x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[3]), device=x.device)], dim=1)
            self.act['fc2']=x        

            x = self.fc2(x, t=t)
            x = self.drop2(self.relu(x))

            y = self.fc3(x)
        else:
            bsz = deepcopy(x.size(0))

            x = self.conv1(x, t=t, fai=self.weight1)
            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            x = self.conv2(x, t=t, fai=self.weight2)
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            x=x.reshape(bsz,-1)

            x = self.fc1(x, t=t, fai=self.weight3)
            x = self.drop2(self.relu(x))

            x = self.fc2(x, t=t, fai=self.weight4)
            x = self.drop2(self.relu(x))

            y = self.fc3(x)
            
        return y

    def add_unit(self, model, device='cuda'):
        # ipdb.set_trace()
        old_out_channel, old_in_channel, _, _ = model.conv1.weight.size()
        new_out_channel, new_in_channel, _, _ = self.conv1.weight.size()
        assert old_out_channel == new_out_channel
        (self.conv1.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv1.weight.data)
        self.conv1.weight[:,old_in_channel:].data.zero_()
        select_dim = torch.randperm(3, device=device)[:self.expand[0][-1]]
        self.select1 = model.select1
        self.select1.append(select_dim)

        old_out_channel, old_in_channel, _, _ = model.conv2.weight.size()
        new_out_channel, new_in_channel, _, _ = self.conv2.weight.size()
        assert old_out_channel == new_out_channel
        (self.conv2.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv2.weight.data)
        self.conv2.weight[:,old_in_channel:].data.zero_()
        select_dim = torch.randperm(self.channel1, device=device)[:self.expand[1][-1]]
        self.select2 = model.select2
        self.select2.append(select_dim)

        old_out_dim, old_in_dim = model.fc1.weight.size()
        new_out_dim, new_in_dim = self.fc1.weight.size()
        assert old_out_channel == new_out_channel
        (self.fc1.weight[:old_out_dim, :old_in_dim]).data.copy_(model.fc1.weight.data)
        self.fc1.weight[:,old_in_dim:].data.zero_()
        select_dim = torch.randperm(self.channel2*self.smid*self.smid, device=device)[:self.expand[2][-1]]
        self.select3 = model.select3
        self.select3.append(select_dim)

        old_out_dim, old_in_dim = model.fc2.weight.size()
        new_out_dim, new_in_dim = self.fc2.weight.size()
        assert old_out_dim == new_out_dim
        (self.fc2.weight[:old_out_dim, :old_in_dim]).data.copy_(model.fc2.weight.data)
        self.fc2.weight[:,old_in_dim:].data.zero_()
        select_dim = torch.randperm(self.channel3, device=device)[:self.expand[3][-1]]
        self.select4 = model.select4
        self.select4.append(select_dim)

        out_dim, in_dim = model.fc3.weight.size()
        (self.fc3.weight[:, :in_dim]).data.copy_(model.fc3.weight.data)

    def define_task_lr_params(self, weights,init=False): 
        # Setup learning parameters
        self.weights = nn.ParameterList([])
        self.weights.append(nn.Parameter(torch.zeros(weights[0].shape, requires_grad=True).data.uniform_(-sqrt(1./(weights[0].shape[0])),sqrt(1./(weights[0].shape[0])))))
        self.weights.append(nn.Parameter(torch.zeros(weights[1].shape, requires_grad=True).data.uniform_(-sqrt(1./(weights[1].shape[0])),sqrt(1./(weights[1].shape[0])))))
        self.weights.append(nn.Parameter(torch.zeros(weights[2].shape, requires_grad=True).data.uniform_(-sqrt(1./(weights[2].shape[0])),sqrt(1./(weights[2].shape[0])))))
        self.weights.append(nn.Parameter(torch.zeros(weights[3].shape, requires_grad=True).data.uniform_(-sqrt(1./(weights[3].shape[0])),sqrt(1./(weights[3].shape[0])))))

        if init:
            self.weights[0].data.copy_(weights[0])
            self.weights[1].data.copy_(weights[1])
            self.weights[2].data.copy_(weights[2])
            self.weights[3].data.copy_(weights[3])

        self.weight1.append(self.weights[0])
        self.weight2.append(self.weights[1])
        self.weight3.append(self.weights[2])
        self.weight4.append(self.weights[3])

def expand_network(n_outputs, n_tasks, model, expand_size, channels=[20, 50, 800, 500]):
    # get the expand size
    expand1, expand2, expand3, expand4 = expand_size

    # get the new model
    new_model = Learner(n_outputs, n_tasks, channels, copy_model=model, expand_size=[expand1, expand2, expand3, expand4])
    return new_model

def expand_feature(expand_size, task, feature_list, model):
    new_model = None
    new_feature_list = []

    feature0 = feature_list[0]
    if expand_size[0] > 0:
        out_channel, in_channel, ksize, _ = model.conv1.weight.size()
        new_feature0 = np.zeros((in_channel*ksize*ksize, feature0.shape[1]+in_channel*ksize*ksize-feature0.shape[0]), dtype=np.float32)
        new_feature0[:feature0.shape[0], :feature0.shape[1]] = feature0
        new_feature0[feature0.shape[0]:, feature0.shape[1]:] = np.eye(in_channel*ksize*ksize-feature0.shape[0])
    else:
        new_feature0 = feature0
    new_feature_list.append(new_feature0)

    feature1 = feature_list[1]
    if expand_size[1] > 0:
        out_channel, in_channel, ksize, _ = model.conv2.weight.size()
        new_feature1 = np.zeros((in_channel*ksize*ksize, feature1.shape[1]+in_channel*ksize*ksize-feature1.shape[0]), dtype=np.float32)
        new_feature1[:feature1.shape[0], :feature1.shape[1]] = feature1
        new_feature1[feature1.shape[0]:, feature1.shape[1]:] = np.eye(in_channel*ksize*ksize-feature1.shape[0])
    else:
        new_feature1 = feature1
    new_feature_list.append(new_feature1)

    feature2 = feature_list[2]
    if expand_size[2] > 0:
        out_channel, in_channel, ksize, _ = model.conv3.weight.size()
        new_feature2 = np.zeros((in_channel*ksize*ksize, feature2.shape[1]+in_channel*ksize*ksize-feature2.shape[0]), dtype=np.float32)
        new_feature2[:feature2.shape[0], :feature2.shape[1]] = feature2
        new_feature2[feature2.shape[0]:, feature2.shape[1]:] = np.eye(in_channel*ksize*ksize-feature2.shape[0])
    else:
        new_feature2 = feature2
    new_feature_list.append(new_feature2)

    feature3 = feature_list[3]
    if expand_size[3] > 0:
        _, in_dim = model.fc1.weight.size()
        new_feature3 = np.zeros((in_dim, feature3.shape[1]+in_dim-feature3.shape[0]), dtype=np.float32)
        new_feature3[:feature3.shape[0], :feature3.shape[1]] = feature3
        new_feature3[feature3.shape[0]:, feature3.shape[1]:] = np.eye(in_dim-feature3.shape[0])
    else:
        new_feature3 = feature3
    new_feature_list.append(new_feature3)

    feature4 = feature_list[4]
    if expand_size[4] > 0:
        _, in_dim = model.fc2.weight.size()
        new_feature4 = np.zeros((in_dim, feature4.shape[1]+in_dim-feature4.shape[0]), dtype=np.float32)
        new_feature4[:feature4.shape[0], :feature4.shape[1]] = feature4
        new_feature4[feature4.shape[0]:, feature4.shape[1]:] = np.eye(in_dim-feature4.shape[0])
    else:
        new_feature4 = feature4
    new_feature_list.append(new_feature4)

    return new_feature_list



