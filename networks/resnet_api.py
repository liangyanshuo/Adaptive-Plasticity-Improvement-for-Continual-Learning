import numpy as np
from numpy.lib.function_base import select
import torch
from torch import nn
from torch.nn.functional import relu, avg_pool2d
from math import sqrt
import ipdb

from collections import OrderedDict

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, expand_size=None, copy_model=None):
        super(BasicBlock, self).__init__()
        if stride != 1 or in_planes != self.expansion * planes:
            length = 3
        else:
            length = 2

        if expand_size != None:
            assert copy_model != None
            self.expand = copy_model.expand
            for i in range(len(expand_size)):
                self.expand[i].append(expand_size[i])

            self.weight = []
            for i in range(len(expand_size)):
                self.weight.append([])
                for j in range(len(copy_model.weight[i])):
                    self.weight[i].append(torch.zeros(copy_model.weight[i][j].size(),device='cuda').copy_(copy_model.weight[i][j].data))
        else:
            self.expand = []
            for i in range(length):
                self.expand.append([])
            self.weight = []
            for i in range(length):
                self.weight.append([])

        self.conv1 = conv3x3(in_planes+sum(self.expand[0]), planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.channel1 = in_planes

        self.conv2 = conv3x3(planes+sum(self.expand[1]), planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.channel2 = planes

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv3 = nn.Conv2d(in_planes+sum(self.expand[2]), self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            self.channel3 = in_planes
        else:
            self.conv3, self.bn3 = None, None
        self.act = OrderedDict()
        self.count = 0

        if copy_model != None:
            self.add_unit(copy_model)
        else:
            self.select1, self.select2 = [], []
            if stride != 1 or in_planes != self.expansion * planes:
                self.select3 = []

    def forward(self, x, t):
        self.count = self.count % 2 

        if t != 0:
            try:
                exp_x = torch.cat([x] + [torch.matmul(x.permute(0, 2, 3, 1), self.weight[0][i]).permute(0, 3, 1, 2) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[0][t:]), x.size(2), x.size(3), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
        else:
            exp_x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[0]), x.size(2), x.size(3), device=x.device)], dim=1)
        self.act['conv_{}'.format(self.count)] = exp_x
        self.count +=1

        out = relu(self.bn1(self.conv1(exp_x)))
        self.count = self.count % 2 

        if t != 0:
            try:
                exp_out = torch.cat([out] + [torch.matmul(out.permute(0, 2, 3, 1), self.weight[1][i]).permute(0, 3, 1, 2) for i in range(t)] + [torch.zeros(out.size(0),sum(self.expand[1][t:]), out.size(2), out.size(3), device=out.device)], dim=1)
            except:
                ipdb.set_trace()
        else:
            exp_out = torch.cat([out, torch.zeros(out.size(0),sum(self.expand[1]), out.size(2), out.size(3), device=out.device)], dim=1)
        self.act['conv_{}'.format(self.count)] = exp_out
        self.count +=1

        out = self.bn2(self.conv2(exp_out))

        if self.conv3 != None:
            if t != 0:
                try:
                    exp_x = torch.cat([x] + [torch.matmul(x.permute(0, 2, 3, 1), self.weight[2][i]).permute(0, 3, 1, 2) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand[2][t:]), x.size(2), x.size(3), device=x.device)], dim=1)
                except:
                    ipdb.set_trace()
            else:
                exp_x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand[2]), x.size(2), x.size(3), device=x.device)], dim=1)
            self.act['conv_{}'.format(self.count)] = exp_x
            
            shortcut = self.bn3(self.conv3(exp_x))
        else:
            shortcut = x

        out += shortcut
        out = relu(out)
        return out
    
    def add_unit(self, model, device='cuda'):
        # ipdb.set_trace()
        old_out_channel, old_in_channel, _, _ = model.conv1.weight.size()
        new_out_channel, new_in_channel, _, _ = self.conv1.weight.size()
        assert old_out_channel == new_out_channel
        (self.conv1.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv1.weight.data)
        select_dim = torch.randperm(self.channel1, device=device)[:self.expand[0][-1]]
        self.select1 = model.select1
        self.select1.append(select_dim)

        self.bn1.weight.data.copy_(model.bn1.weight.data)
        self.bn1.bias.data.copy_(model.bn1.bias.data)

        old_out_channel, old_in_channel, _, _ = model.conv2.weight.size()
        new_out_channel, new_in_channel, _, _ = self.conv2.weight.size()
        assert old_out_channel == new_out_channel
        (self.conv2.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv2.weight.data)
        try:
            select_dim = torch.randperm(self.channel2, device=device)[:self.expand[1][-1]]
        except:
            print(self.expand)
            raise Exception('Wrong')
        
        self.select2 = model.select2
        self.select2.append(select_dim)

        self.bn2.weight.data.copy_(model.bn2.weight.data)
        self.bn2.bias.data.copy_(model.bn2.bias.data)

        if self.conv3!=None:
            old_out_channel, old_in_channel, _, _ = model.conv3.weight.size()
            new_out_channel, new_in_channel, _, _ = self.conv3.weight.size()
            assert old_out_channel == new_out_channel
            (self.conv3.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv3.weight.data)
            select_dim = torch.randperm(self.channel3, device=device)[:self.expand[2][-1]]
            self.select3 = model.select3
            self.select3.append(select_dim)

            self.bn3.weight.data.copy_(model.bn3.weight.data)
            self.bn3.bias.data.copy_(model.bn3.bias.data)

    def define_task_lr_params(self, weights,init=False): 
        # Setup learning parameters
        self.weights = nn.ParameterList([])
        self.weights.append(nn.Parameter(torch.zeros(weights[0].shape, requires_grad=True)))
        self.weights.append(nn.Parameter(torch.zeros(weights[1].shape, requires_grad=True)))
        self.weights[0].data.copy_(weights[0])
        self.weights[1].data.copy_(weights[1])

        self.weight[0].append(self.weights[0])
        self.weight[1].append(self.weights[1])

        if self.conv3!=None:
            self.weights.append(nn.Parameter(torch.zeros(weights[2].shape, requires_grad=True)))
            self.weights[2].data.copy_(weights[2])
            self.weight[2].append(self.weights[2])

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_outputs, n_tasks, nf, expand_size=None, copy_model=None, dataset='five_datasets'):
        super(ResNet, self).__init__()
        self.in_planes = nf

        if expand_size != None:
            assert copy_model != None
            self.expand1 = copy_model.expand1
            self.expand1.append(expand_size[0])

            self.weight1 = []
            for i in range(len(copy_model.weight1)):
                self.weight1.append(torch.zeros(copy_model.weight1[i].size(),device='cuda').copy_(copy_model.weight1[i].data))
        else:
            self.expand1 = []
            self.weight1 = []

        self.n_outputs, self.n_tasks = n_outputs, n_tasks
        if dataset == 'five_datasets':
            self.conv1 = conv3x3(3+sum(self.expand1), nf * 1, 1)
            self.imagesize = 32
        elif dataset == 'mini_imagenet':
            self.conv1 = conv3x3(3+sum(self.expand1), nf * 1, 2)
            self.imagesize = 84
        else:
            raise NotImplementedError

        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)

        if expand_size != None:
            self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, expand_size=[expand_size[1:3], expand_size[3:5]], copy_model=copy_model.layer1)
            self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, expand_size=[expand_size[5:8], expand_size[8:10]], copy_model=copy_model.layer2)
            self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, expand_size=[expand_size[10:13], expand_size[13:15]], copy_model=copy_model.layer3)
            self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, expand_size=[expand_size[15:18], expand_size[18:20]], copy_model=copy_model.layer4)
        else:
            self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        if 'five' in dataset:
            self.linear = nn.Linear(nf * 8 * block.expansion * 4, self.n_outputs, bias=False)
        elif 'imagenet' in dataset:
            self.linear = nn.Linear(1440, self.n_outputs, bias=False)
        else:
            raise NotImplementedError
        self.act = OrderedDict()

        if copy_model != None:
            self.add_unit(copy_model)

        self.multi_head = True

    def _make_layer(self, block, planes, num_blocks, stride, expand_size=None, copy_model=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        i = 0
        for stride in strides:
            if expand_size != None:
                layers.append(block(self.in_planes, planes, stride, expand_size[i], copy_model[i]))
            else:
                layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            i += 1
        return layers

    def forward(self, x, t=0):
        bsz = x.size(0)

        x = x.view(-1, 3, self.imagesize, self.imagesize)
        if t != 0:
            try:
                x = torch.cat([x] + [torch.matmul(x.permute(0, 2, 3, 1), self.weight1[i]).permute(0, 3, 1, 2) for i in range(t)] + [torch.zeros(x.size(0),sum(self.expand1[t:]), x.size(2), x.size(3), device=x.device)], dim=1)
            except:
                ipdb.set_trace()
        else:
            x = torch.cat([x, torch.zeros(x.size(0),sum(self.expand1), x.size(2), x.size(3), device=x.device)], dim=1)
        self.act['conv_in'] = x

        out = relu(self.bn1(self.conv1(x))) 
        out = self.layer1[0](out, t)
        out = self.layer1[1](out, t)

        out = self.layer2[0](out, t)
        out = self.layer2[1](out, t)

        out = self.layer3[0](out, t)
        out = self.layer3[1](out, t)

        out = self.layer4[0](out, t)
        out = self.layer4[1](out, t)

        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y=self.linear(out)
        return y

    def add_unit(self, model):
        old_out_channel, old_in_channel, _, _ = model.conv1.weight.size()
        new_out_channel, new_in_channel, _, _ = self.conv1.weight.size()
        assert old_out_channel == new_out_channel
        (self.conv1.weight[:old_out_channel, :old_in_channel]).data.copy_(model.conv1.weight.data)

        self.bn1.weight.data.copy_(model.bn1.weight.data)
        self.bn1.bias.data.copy_(model.bn1.bias.data)

        self.linear.weight.data.copy_(model.linear.weight.data)

    def define_task_lr_params(self, weights, init=False): 
        # Setup learning parameters
        self.weights = nn.ParameterList([])
        self.weights.append(nn.Parameter(torch.zeros(weights[0].shape, requires_grad=True)))
        self.weights[0].data.copy_(weights[0])
        self.weight1.append(self.weights[0])

        self.layer1[0].define_task_lr_params(weights[1:3],init)
        self.layer1[1].define_task_lr_params(weights[3:5],init)

        self.layer2[0].define_task_lr_params(weights[5:8],init)
        self.layer2[1].define_task_lr_params(weights[8:10],init)

        self.layer3[0].define_task_lr_params(weights[10:13],init)
        self.layer3[1].define_task_lr_params(weights[13:15],init)

        self.layer4[0].define_task_lr_params(weights[15:18],init)
        self.layer4[1].define_task_lr_params(weights[18:20],init)

def Learner(n_outputs, n_tasks, nf=32, copy_model=None, expand_size=None, dataset='five_datasets'):
    return ResNet(BasicBlock, [2, 2, 2, 2], n_outputs, n_tasks, nf, expand_size, copy_model, dataset)

# def expand_network(args, x, feature_list, project_type, taskcla, per_layer_retain, copy_model, device='cuda'):
#     channels  = [ 3, 20,20,20,20, 20,40,20,40,40, 40,80,40,80,80, 80,160,80,160,160] 
#     kernel_size = [3, 3,3,3,3, 3,3,1,3,3, 3,3,1,3,3, 3,3,1,3,3]
#     bounds = []
#     print(per_layer_retain)
#     expand_quant = [0.5, 0.4,0.4,0.4,0.4, 0.3,0.3,0.5,0.3,0.3, 0.25,0.25,0.5,0.25,0.25, 0.2,0.2,0.5,0.2,0.2 ]

#     alpha, beta, mu, c = 0.2, 0.3, 2e4, 8e-5
#     p=[540, 3600,3600,3600,3600, 7200,14400,800,14400,14400, 28800,57600,3200,57600,57600, 115200,230400,12800,230400,230400]
#     expand_quant = []
#     for i in range(len(p)):
#         expand_quant.append(alpha+beta/(1+math.exp(c*(p[i]-mu))))
#     print(expand_quant)

#     for i in range(len(feature_list)):
#         expand_dim = max((expand_quant[i] - per_layer_retain[i])*feature_list[i].shape[0], 0)
#         bounds.append(math.ceil(expand_dim/(kernel_size[i]*kernel_size[i])))
#         # if project_type[i] == 'retain':
#         #     expand_size.append(max(int(expand_quant*channels[i]) - feature_list[i].shape[1]//(kernel_size[i]*kernel_size[i]), 0))
#         # elif project_type[i] == 'remove':
#         #     expand_size.append(max(int(expand_quant*channels[i]) - (feature_list[i].shape[0]-feature_list[i].shape[1])//(kernel_size[i]*kernel_size[i]), 0))
#         # else:
#         #     raise Exception('Wrong')

#     mat_list = get_representation_matrix_ResNet18(copy_model, device, x)

#     for i in range(len(mat_list)):
#         sz = mat_list[i].shape[-1]
#         mat_list[i] = np.linalg.norm(mat_list[i].transpose().reshape(sz, channels[i], kernel_size[i], kernel_size[i]), ord=2, axis=(2,3)).transpose()

#     weights, expand_size = [], []
#     for i in range(len(mat_list)):
#         if bounds[i] > 0:
#             # ipdb.set_trace()
#             U, S, Vh = np.linalg.svd(mat_list[i])
#             sval_total = (S**2).sum()
#             sval_ratio = (S**2)/sval_total
#             # r = np.sum(np.cumsum(sval_ratio)<bounds[i])+1
#             r = bounds[i]
#             weights.append(torch.from_numpy(U[:,0:r]).float())
#             expand_size.append(r)
#         else:
#             weights.append(torch.zeros(mat_list[i].shape[0],0))
#             expand_size.append(0)

#     # get the new model
#     print(expand_size)
#     new_model = ResNet18(taskcla, nf=20, copy_model=copy_model, expand_size=expand_size)
#     new_model.define_task_lr_params(weights=weights)
#     new_model = new_model.to(device)
#     return new_model