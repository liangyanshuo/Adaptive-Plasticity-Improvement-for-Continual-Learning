import numpy as np
from torch import nn

from collections import OrderedDict

import torch

from copy import deepcopy
import ipdb

## Define LeNet model 
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Learner(nn.Module):
    def __init__(self, n_outputs, n_tasks, channels=[20, 50, 800, 500], expand_size = None, copy_model=None, device='cuda'):
        super(Learner, self).__init__()
        self.act=OrderedDict()

        if expand_size != None:
            assert copy_model != None
            self.expand1, self.expand2, self.expand3, self.expand4 = copy_model.expand1, copy_model.expand2, copy_model.expand3, copy_model.expand4
            expand1, expand2, expand3, expand4 = expand_size
            self.expand1.append(expand1)
            self.expand2.append(expand2)
            self.expand3.append(expand3)
            self.expand4.append(expand4)

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
            self.expand1, self.expand2, self.expand3, self.expand4 = [], [], [], []
            self.weight1, self.weight2, self.weight3, self.weight4 = [], [], [], []


        self.map =[]
        self.ksize=[]
        self.in_channel =[]

        channel1, channel2, channel3, channel4 = channels
        self.channel1, self.channel2, self.channel3, self.channel4 = channels
        
        self.map.append(32)
        self.conv1 = nn.Conv2d(3+sum(self.expand1), channel1, 5, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(3)        
        self.map.append(s)
        self.conv2 = nn.Conv2d(channel1+sum(self.expand2), channel2, 5, bias=False, padding=2)
        
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

        self.fc1 = nn.Linear(channel2*self.smid*self.smid+sum(self.expand3),channel3, bias=False)
        self.fc2 = nn.Linear(channel3+sum(self.expand4),channel4, bias=False)
        self.map.extend([channel3])
        
        self.fc3 = torch.nn.Linear(channel4, n_outputs,bias=False)

        if copy_model != None:
            self.add_unit(copy_model, device)
        else:
            self.select1, self.select2, self.select3, self.select4 = [], [], [], []

        self.multi_head = True
        
    def forward(self, x, t=0):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x

        if t != 0:
            try:
                x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand1[:t-1]), x.size(2), x.size(3), device=x.device), torch.matmul(x.permute(0, 2, 3, 1), self.weight1[t-1]).permute(0, 3, 1, 2), torch.zeros(x.size(0),sum(self.expand1[t:]), x.size(2), x.size(3), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
                raise Exception('Wrong')
        else:
            x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand1), x.size(2), x.size(3), device=x.device)], dim=1)

        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.lrn(self.relu(x))))
        self.act['conv2']=x

        if t != 0:
            try:
                x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand2[:t-1]), x.size(2), x.size(3), device=x.device), torch.matmul(x.permute(0, 2, 3, 1), self.weight2[t-1]).permute(0, 3, 1, 2), torch.zeros(x.size(0),sum(self.expand2[t:]), x.size(2), x.size(3), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
        else:
            x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand2), x.size(2), x.size(3), device=x.device)], dim=1)

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.lrn (self.relu(x))))

        x=x.reshape(bsz,-1)
        self.act['fc1']=x

        if t != 0:
            try:
                x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand3[:t-1]), device=x.device), torch.mm(x, self.weight3[t-1]), torch.zeros(x.size(0),sum(self.expand3[t:]), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
        else:
            x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand3), device=x.device)], dim=1)

        x = self.fc1(x)
        x = self.drop2(self.relu(x))
        self.act['fc2']=x        

        if t != 0:
            try:
                x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand4[:t-1]), device=x.device), torch.mm(x, self.weight4[t-1]), torch.zeros(x.size(0),sum(self.expand4[t:]), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
        else:
            x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand4), device=x.device)], dim=1)

        x = self.fc2(x)
        x = self.drop2(self.relu(x))

        y = self.fc3(x)
            
        return y

    def add_unit(self, model, device='cuda'):
        # ipdb.set_trace()
        old_out_channel, old_in_channel, _, _ = model.conv1.weight.size()
        new_out_channel, new_in_channel, _, _ = self.conv1.weight.size()
        try:
            assert old_out_channel == new_out_channel
        except:
            print(old_out_channel, new_out_channel)
            raise Exception('Wrong')
        (self.conv1.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv1.weight.data)
        select_dim = torch.randperm(3, device=device)[:self.expand1[-1]]
        self.select1 = model.select1
        self.select1.append(select_dim)

        old_out_channel, old_in_channel, _, _ = model.conv2.weight.size()
        new_out_channel, new_in_channel, _, _ = self.conv2.weight.size()
        assert old_out_channel == new_out_channel
        (self.conv2.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv2.weight.data)
        select_dim = torch.randperm(self.channel1, device=device)[:self.expand2[-1]]
        self.select2 = model.select2
        self.select2.append(select_dim)

        old_out_dim, old_in_dim = model.fc1.weight.size()
        new_out_dim, new_in_dim = self.fc1.weight.size()
        assert old_out_channel == new_out_channel
        (self.fc1.weight[:old_out_dim, :old_in_dim]).data.copy_(model.fc1.weight.data)
        select_dim = torch.randperm(self.channel2*self.smid*self.smid, device=device)[:self.expand3[-1]]
        self.select3 = model.select3
        self.select3.append(select_dim)

        old_out_dim, old_in_dim = model.fc2.weight.size()
        new_out_dim, new_in_dim = self.fc2.weight.size()
        assert old_out_dim == new_out_dim
        (self.fc2.weight[:old_out_dim, :old_in_dim]).data.copy_(model.fc2.weight.data)
        select_dim = torch.randperm(self.channel3, device=device)[:self.expand4[-1]]
        self.select4 = model.select4
        self.select4.append(select_dim)

        out_dim, in_dim = model.fc3.weight.size()
        (self.fc3.weight[:, :in_dim]).data.copy_(model.fc3.weight.data)

    def define_task_lr_params(self, weights): 
        # Setup learning parameters
        self.weights = nn.ParameterList([])
        self.weights.append(nn.Parameter(torch.zeros(weights[0].shape, requires_grad=True).data.copy_(weights[0])))
        self.weights.append(nn.Parameter(torch.zeros(weights[1].shape, requires_grad=True).data.copy_(weights[1])))
        self.weights.append(nn.Parameter(torch.zeros(weights[2].shape, requires_grad=True).data.copy_(weights[2])))
        self.weights.append(nn.Parameter(torch.zeros(weights[3].shape, requires_grad=True).data.copy_(weights[3])))

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

