import numpy as np
from scipy.stats.stats import mode
import torch
from torch import nn
from collections import OrderedDict

from copy import deepcopy
from math import sqrt
import ipdb

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Learner(nn.Module):
    def __init__(self,n_outputs, n_tasks, channels=[64, 128, 256, 2048], expand_size = None, copy_model=None, device='cuda'):
        super(Learner, self).__init__()
        self.act=OrderedDict()

        if expand_size != None:
            assert copy_model != None
            self.expand = copy_model.expand 
            expand1, expand2, expand3, expand4, expand5 = expand_size
            self.expand[0].append(expand1)
            self.expand[1].append(expand2)
            self.expand[2].append(expand3)
            self.expand[3].append(expand4)
            self.expand[4].append(expand5)

            self.weight1, self.weight2, self.weight3, self.weight4, self.weight5 = [], [], [], [], []
            for i in range(len(copy_model.weight1)):
                self.weight1.append(torch.zeros(copy_model.weight1[i].size(),device=device).copy_(copy_model.weight1[i].data))
            for i in range(len(copy_model.weight2)):
                self.weight2.append(torch.zeros(copy_model.weight2[i].size(),device=device).copy_(copy_model.weight2[i].data))
            for i in range(len(copy_model.weight3)):
                self.weight3.append(torch.zeros(copy_model.weight3[i].size(),device=device).copy_(copy_model.weight3[i].data))
            for i in range(len(copy_model.weight4)):
                self.weight4.append(torch.zeros(copy_model.weight4[i].size(),device=device).copy_(copy_model.weight4[i].data))
            for i in range(len(copy_model.weight5)):
                self.weight5.append(torch.zeros(copy_model.weight5[i].size(),device=device).copy_(copy_model.weight5[i].data))
            # self.weight1.append((weights[0]).to(device))
            # self.weight2.append((weights[1]).to(device))
            # self.weight3.append((weights[2]).to(device))
            # self.weight4.append((weights[3]).to(device))
            # self.weight5.append((weights[4]).to(device))
        else:
            self.expand = [[], [], [], [], []]
            self.weight1, self.weight2, self.weight3, self.weight4, self.weight5 = [], [], [], [], []

        self.channel1, self.channel2, self.channel3, self.channel4 = channels
        # channel1, channel2, channel3, channel4 = channels
        # channel1 += sum(self.expand1)
        # channel2 += sum(self.expand2)
        # channel3 += sum(self.expand3)
        # channel4 += sum(self.expand4)

        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3+sum(self.expand[0]), self.channel1, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channel1, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3+sum(self.expand[0]))
        self.map.append(s)
        self.conv2 = nn.Conv2d(self.channel1+sum(self.expand[1]), self.channel2, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channel2, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(self.channel1+sum(self.expand[1]))
        self.map.append(s)
        self.conv3 = nn.Conv2d(self.channel2+sum(self.expand[2]), self.channel3, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channel3, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(self.channel2+sum(self.expand[2]))
        self.map.append(self.channel3*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.channel3*self.smid*self.smid+sum(self.expand[3]),self.channel4, bias=False)
        self.bn4 = nn.BatchNorm1d(self.channel4, track_running_stats=False)
        self.fc2 = nn.Linear(self.channel4+sum(self.expand[4]),self.channel4, bias=False)
        self.bn5 = nn.BatchNorm1d(self.channel4, track_running_stats=False)
        self.map.extend([self.channel4+sum(self.expand[4])])

        self.fc3 = torch.nn.Linear(self.channel4,n_outputs,bias=False)

        if copy_model != None:
            self.add_unit(copy_model, device)
        else:
            self.select1, self.select2, self.select3, self.select4, self.select5 = [], [], [], [], []
        
        self.multi_head = True
        self.n_tasks = n_tasks
        
    def forward(self, x, t=0):
        bsz = deepcopy(x.size(0))

        if t != 0:
            try:
                x = torch.cat([x] + [torch.matmul(x.permute(0, 2, 3, 1), self.weight1[i]).permute(0, 3, 1, 2) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[0][t:]), x.size(2), x.size(3), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
                raise Exception('Wrong')
        else:
            x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[0]), x.size(2), x.size(3), device=x.device)], dim=1)
        self.act['conv1']=x

        try:
            x = self.conv1(x)
        except:
            ipdb.set_trace()
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))
        
        if t != 0:
            try:
                x = torch.cat([x] + [torch.matmul(x.permute(0, 2, 3, 1), self.weight2[i]).permute(0, 3, 1, 2) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[1][t:]), x.size(2), x.size(3), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
        else:
            x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[1]), x.size(2), x.size(3), device=x.device)], dim=1)
        self.act['conv2']=x

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))
        
        if t != 0:
            try:
                x = torch.cat([x] + [torch.matmul(x.permute(0, 2, 3, 1), self.weight3[i]).permute(0, 3, 1, 2) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[2][t:]), x.size(2), x.size(3), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
        else:
            x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[2]), x.size(2), x.size(3), device=x.device)], dim=1)
        self.act['conv3']=x        

        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))
        
        x=x.view(bsz,-1)

        if t != 0:
            try:
                x = torch.cat([x] + [torch.mm(x, self.weight4[i]) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[3][t:]), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
                raise Exception('Wrong')
        else:
            x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[3]), device=x.device)], dim=1)
        self.act['fc1']=x

        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        if t != 0:
            try:
                x = torch.cat([x] + [torch.mm(x, self.weight5[i]) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[4][t:]), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
        else:
            x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[4]), device=x.device)], dim=1)
        self.act['fc2']=x        

        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))

        y = self.fc3(x)
            
        return y

    def add_unit(self, model, device='cuda'):
        # ipdb.set_trace()
        old_out_channel, old_in_channel, _, _ = model.conv1.weight.size()
        new_out_channel, new_in_channel, _, _ = self.conv1.weight.size()
        assert old_out_channel == new_out_channel
        (self.conv1.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv1.weight.data)
        select_dim = torch.randperm(3, device=device)[:self.expand[0][-1]]
        self.select1 = model.select1
        self.select1.append(select_dim)

        self.bn1.weight.data.copy_(model.bn1.weight.data)
        self.bn1.bias.data.copy_(model.bn1.bias.data)

        old_out_channel, old_in_channel, _, _ = model.conv2.weight.size()
        new_out_channel, new_in_channel, _, _ = self.conv2.weight.size()
        assert old_out_channel == new_out_channel
        (self.conv2.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv2.weight.data)
        select_dim = torch.randperm(self.channel1, device=device)[:self.expand[1][-1]]
        self.select2 = model.select2
        self.select2.append(select_dim)

        self.bn2.weight.data.copy_(model.bn2.weight.data)
        self.bn2.bias.data.copy_(model.bn2.bias.data)

        old_out_channel, old_in_channel, _, _ = model.conv3.weight.size()
        new_out_channel, new_in_channel, _, _ = self.conv3.weight.size()
        assert old_out_channel == new_out_channel
        (self.conv3.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv3.weight.data)
        select_dim = torch.randperm(self.channel2, device=device)[:self.expand[2][-1]]
        self.select3 = model.select3
        self.select3.append(select_dim)

        self.bn3.weight.data.copy_(model.bn3.weight.data)
        self.bn3.bias.data.copy_(model.bn3.bias.data)

        old_out_dim, old_in_dim = model.fc1.weight.size()
        new_out_dim, new_in_dim = self.fc1.weight.size()
        assert old_out_channel == new_out_channel
        (self.fc1.weight[:old_out_dim, :old_in_dim]).data.copy_(model.fc1.weight.data)
        select_dim = torch.randperm(self.channel3*self.smid*self.smid, device=device)[:self.expand[3][-1]]
        self.select4 = model.select4
        self.select4.append(select_dim)

        self.bn4.weight.data.copy_(model.bn4.weight.data)
        self.bn4.bias.data.copy_(model.bn4.bias.data)

        old_out_dim, old_in_dim = model.fc2.weight.size()
        new_out_dim, new_in_dim = self.fc2.weight.size()
        assert old_out_dim == new_out_dim
        (self.fc2.weight[:old_out_dim, :old_in_dim]).data.copy_(model.fc2.weight.data)
        select_dim = torch.randperm(self.channel4, device=device)[:self.expand[4][-1]]
        self.select5 = model.select5
        self.select5.append(select_dim)

        self.bn5.weight.data.copy_(model.bn5.weight.data)
        self.bn5.bias.data.copy_(model.bn5.bias.data)

        out_dim, in_dim = model.fc3.weight.size()
        (self.fc3.weight[:, :in_dim]).data.copy_(model.fc3.weight.data)

    def define_task_lr_params(self, weights,init=True): 
        # Setup learning parameters
        self.weights = nn.ParameterList([])
        self.weights.append(nn.Parameter(torch.zeros(weights[0].shape, requires_grad=True).data.copy_(weights[0])))
        self.weights.append(nn.Parameter(torch.zeros(weights[1].shape, requires_grad=True).data.copy_(weights[1])))
        self.weights.append(nn.Parameter(torch.zeros(weights[2].shape, requires_grad=True).data.copy_(weights[2])))
        self.weights.append(nn.Parameter(torch.zeros(weights[3].shape, requires_grad=True).data.copy_(weights[3])))
        self.weights.append(nn.Parameter(torch.zeros(weights[4].shape, requires_grad=True).data.copy_(weights[4])))

        self.weight1.append(self.weights[0])
        self.weight2.append(self.weights[1])
        self.weight3.append(self.weights[2])
        self.weight4.append(self.weights[3])
        self.weight5.append(self.weights[4])

def expand_network(n_outputs, n_tasks, model, expand_size, channels=[64, 128, 256, 2048]):
    # get the expand size
    expand1, expand2, expand3, expand4, expand5 = expand_size

    # get the new model
    new_model = Learner(n_outputs, n_tasks, channels, copy_model=model, expand_size=[expand1, expand2, expand3, expand4, expand5])
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
