import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import math
# from model.base import *
from torch import nn
import ipdb
from networks import alexnet_gpm, resnet_gpm, lenet_gpm

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.n_tasks = n_tasks
        self.nc_per_task = self.n_outputs//self.n_tasks
        self.loss_ce = torch.nn.CrossEntropyLoss()
        self.glances = args.glances
        self.args = args

        self.arch = args.model_arch
        self.size = args.model_size
        self.step = args.step
        if 'cifar' in args.dataset:
            if self.arch == 'alexnet':
                self.net = alexnet_gpm.Learner(n_outputs, n_tasks, channels=[int(64*self.size), int(128*self.size), int(256*self.size), int(2048*self.size)])
                self.Grad_Norm_Retain = {}
                for k in range(5):
                    self.Grad_Norm_Retain[k] = []
            elif self.arch == 'lenet':
                self.net = lenet_gpm.Learner(n_outputs, n_tasks, channels=[int(20*self.size), int(50*self.size), int(800*self.size), int(500*self.size)])
                self.net.apply(init_weights)
            # self.arch = 'lenet'
        elif 'five' in args.dataset or 'imagenet' in args.dataset:
            self.net = resnet_gpm.Learner(n_outputs, n_tasks, nf=20, dataset = args.dataset)
            self.arch = 'resnet'
        else:
            raise NotImplementedError
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr)
        self.lr = args.lr

        self.dataset = args.dataset

        self.threshold = None
        self.feature_list = []
        self.project_type = []
        self.feature_mat = None
        self.epoch = 0
        self.current_task = -1
        self.iter = 0
        self.n_epochs = args.n_epochs
        self.cuda = args.cuda

    def forward(self, x, t):
        output = self.net(x)

        if self.net.multi_head:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)

        return output

    def observe(self, x, y, t, evalue = False, writer=None):
        if t != self.current_task:
            if self.arch == 'alexnet' and self.current_task>0:
                if len(self.Grad_Norm_Retain[0]):
                    avg = []
                    for k in range(5):
                        avg.append(sum(self.Grad_Norm_Retain[k])/len(self.Grad_Norm_Retain[k]))
                    writer.add_scalar(f"5.AVG-Grad-Norm-Retain", sum(avg)/len(avg), self.current_task)

                    for KK in self.Grad_Norm_Retain.keys():
                        self.Grad_Norm_Retain[KK] = []

            self.current_task = t
            self.iter = 0

        for pass_itr in range(self.glances):
            self.iter += 1
            self.zero_grads()

            output = self.net(x)
            if self.net.multi_head:
                # make sure we predict classes within the current task
                offset1, offset2 = self.compute_offsets(t)
                loss = self.loss_ce(output[:,offset1:offset2], y-offset1)
            else:
                loss = self.loss_ce(output, y)
            loss.backward()

            if self.arch == 'mlp' and writer != None:
                per_layer_norm = []
                for k, (m, params) in enumerate(self.net.named_parameters()):
                    per_layer_norm.append(params.grad.norm(p=2))
            elif self.arch == 'alexnet' and writer != None:
                kk = 0
                per_layer_norm = []
                for k, (m,params) in enumerate(self.net.named_parameters()):
                    if k<15 and len(params.size())!=1:
                        per_layer_norm.append(params.grad.norm(p=2))
                        kk +=1
            elif self.arch == 'lenet' and writer != None:
                kk = 0 
                per_layer_norm = []
                # ipdb.set_trace()
                for k, (m,params) in enumerate(self.net.named_parameters()):
                    if k<4 and len(params.size())!=1:
                        per_layer_norm.append(params.grad.norm(p=2))
                        kk +=1

            if self.current_task != 0:
                self.grad_projection(t)

            if self.arch == 'mlp' and writer != None:
                for k, (m, params) in enumerate(self.net.named_parameters()):
                    writer.add_scalar(f"4.Grad-Norm/task-{t}/layer-{k}", params.grad.norm(p=2), self.iter)
                    writer.add_scalar(f"5.Grad-Norm-Retain/task-{t}/layer-{k}", params.grad.norm(p=2)/per_layer_norm[k], self.iter)
            elif self.arch == 'alexnet' and writer != None:
                kk = 0
                for k, (m,params) in enumerate(self.net.named_parameters()):
                    if k<15 and len(params.size())!=1:
                        writer.add_scalar(f"4.Grad-Norm/task-{t}/layer-{kk+1}", params.grad.norm(p=2), self.iter)
                        writer.add_scalar(f"5.Grad-Norm-Retain/task-{t}/layer-{kk+1}", params.grad.norm(p=2)/per_layer_norm[kk], self.iter)
                        if evalue:
                            self.Grad_Norm_Retain[kk].append((params.grad.norm(p=2)/per_layer_norm[kk]).item())
                        kk +=1
            elif self.arch == 'lenet' and writer != None:
                kk = 0 
                # ipdb.set_trace()
                for k, (m,params) in enumerate(self.net.named_parameters()):
                    if k<4 and len(params.size())!=1:
                        writer.add_scalar(f"4.Grad-Norm/task-{t}/layer-{kk+1}", params.grad.norm(p=2), self.iter)
                        writer.add_scalar(f"5.Grad-Norm-Retain/task-{t}/layer-{kk+1}", params.grad.norm(p=2)/per_layer_norm[kk], self.iter)
                        kk +=1

            if not evalue:
                self.optimizer.step()
            # for param in self.net.parameters():
            #     if param.grad != None:
            #         param.data -= self.lr*param.grad.data
            #     else:
            #         assert len(param.size()) == 1

        return loss

    def grad_projection(self, t):
        """
            get the projection of grads on the subspace spanned by GPM
            """
        if self.arch == 'mlp':
            for k, (m,params) in enumerate(self.net.named_parameters()):
                sz =  params.grad.data.size(0)
                if self.project_type[k] == 'retain':
                    try:
                        params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                    (torch.eye(self.feature_mat[k].size(0)).cuda() - self.feature_mat[k])).view(params.size())
                    except:
                        ipdb.set_trace()
                elif self.project_type[k] == 'remove':
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                self.feature_mat[k]).view(params.size())
        elif self.arch == 'alexnet':
            kk = 0 
            for k, (m,params) in enumerate(self.net.named_parameters()):
                if k<15 and len(params.size())!=1:
                    sz =  params.grad.data.size(0)
                    if self.project_type[kk] == 'retain':
                        # ipdb.set_trace()
                        params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                (torch.eye(self.feature_mat[kk].size(0)).cuda() - self.feature_mat[kk])).view(params.size())
                    elif self.project_type[kk] == 'remove':
                        try:
                            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                    self.feature_mat[kk]).view(params.size())
                        except:
                            ipdb.set_trace()
                    kk +=1
                elif (k<15 and len(params.size())==1) and t !=0:
                    params.grad.data.fill_(0)
        elif self.arch == 'resnet':
            kk = 0 
            for k, (m,params) in enumerate(self.net.named_parameters()):
                if len(params.size())==4:
                    sz =  params.grad.data.size(0)
                    try:
                        self.project_type[kk]
                    except:
                        ipdb.set_trace()
                    if self.project_type[kk] == 'retain':
                        # ipdb.set_trace()
                        params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                (torch.eye(self.feature_mat[kk].size(0)).cuda() - self.feature_mat[kk])).view(params.size())
                    elif self.project_type[kk] == 'remove':
                        try:
                            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                    self.feature_mat[kk]).view(params.size())
                        except:
                            ipdb.set_trace()
                    kk+=1
                elif len(params.size())==1 and t != 0:
                    params.grad.data.fill_(0)
        elif self.arch == 'lenet':
            kk = 0 
            # ipdb.set_trace()
            for k, (m,params) in enumerate(self.net.named_parameters()):
                if k<4 and len(params.size())!=1:
                    sz =  params.grad.data.size(0)
                    if self.project_type[kk] == 'retain':
                        params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                (torch.eye(self.feature_mat[kk].size(0)).cuda() - self.feature_mat[kk])).view(params.size())
                    elif self.project_type[kk] == 'remove':
                        try:
                            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                    self.feature_mat[kk]).view(params.size())
                        except:
                            ipdb.set_trace()
                    kk +=1
                elif (k<4 and len(params.size())==1) and t != 0 :
                    params.grad.data.fill_(0)
        else:
            raise NotImplementedError

    def zero_grads(self):
        self.optimizer.zero_grad()
        self.net.zero_grad()

    def get_representation_matrix_ResNet18 (self, x, y=None): 
        # Collect activations by forward pass
        self.net.eval()

        if 'imagenet' in self.args.dataset:
            example_data, sample_num = [], 0
            for i, (data, target) in enumerate(x):
                example_data.append(data)
                sample_num += len(target)
                if sample_num >= 100: break
            example_data = torch.cat(example_data)[:100]
        else:
            r=np.arange(x.size(0))
            np.random.shuffle(r)
            r=torch.LongTensor(r)
            if self.cuda: r = r.cuda()
            b=r[0:100] # ns=100 examples 
            example_data = x[b]
        if self.cuda: example_data = example_data.cuda()
        example_out  = self.net(example_data)
    
        act_list =[]
        act_list.extend([self.net.act['conv_in'], 
            self.net.layer1[0].act['conv_0'], self.net.layer1[0].act['conv_1'], self.net.layer1[1].act['conv_0'], self.net.layer1[1].act['conv_1'],
            self.net.layer2[0].act['conv_0'], self.net.layer2[0].act['conv_1'], self.net.layer2[1].act['conv_0'], self.net.layer2[1].act['conv_1'],
            self.net.layer3[0].act['conv_0'], self.net.layer3[0].act['conv_1'], self.net.layer3[1].act['conv_0'], self.net.layer3[1].act['conv_1'],
            self.net.layer4[0].act['conv_0'], self.net.layer4[0].act['conv_1'], self.net.layer4[1].act['conv_0'], self.net.layer4[1].act['conv_1']])

        if self.args.dataset == 'five_datasets':
            batch_list  = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100] #scaled
            # network arch 
            stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
            map_list    = [32, 32,32,32,32, 32,16,16,16, 16,8,8,8, 8,4,4,4] 
            in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160] 
        elif self.args.dataset == 'mini_imagenet':
            batch_list  = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100] #scaled
            # network arch 
            stride_list = [2, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
            map_list    = [84, 42,42,42,42, 42,21,21,21, 21,11,11,11, 11,6,6,6] 
            in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160] 
        else:
            raise NotImplementedError

        pad = 1
        sc_list=[5,9,13]
        p1d = (1, 1, 1, 1)
        mat_final=[] # list containing GPM Matrices 
        mat_list=[]
        mat_sc_list=[]
        for i in range(len(stride_list)):
            if i==0:
                ksz = 3
            else:
                ksz = 3 
            bsz=batch_list[i]
            st = stride_list[i]     
            k=0
            s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
            mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
            act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                        k +=1
            mat_list.append(mat)
            # For Shortcut Connection
            if i in sc_list:
                k=0
                s=compute_conv_output_size(map_list[i],1,stride_list[i])
                mat = np.zeros((1*1*in_channel[i],s*s*bsz))
                act = act_list[i].detach().cpu().numpy()
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                            k +=1
                mat_sc_list.append(mat) 

        ik=0
        for i in range (len(mat_list)):
            mat_final.append(mat_list[i])
            if i in [6,10,14]:
                mat_final.append(mat_sc_list[ik])
                ik+=1

        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        for i in range(len(mat_final)):
            print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
        print('-'*30)
        return mat_final    

    def get_representation_matrix_lenet (self, x, y=None): 
        # Collect activations by forward pass
        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)
        if self.cuda: r = r.cuda()
        b=r[0:125] # Take 125 random samples 
        example_data = x[b]
        if self.cuda: example_data = example_data.cuda()
        example_out  = self.net(example_data)
    
        batch_list=[2*12,100,125,125] 
        pad = 2
        p1d = (2, 2, 2, 2)
        mat_list=[]
        act_key=list(self.net.act.keys())
        # pdb.set_trace()
        for i in range(len(self.net.map)):
            bsz=batch_list[i]
            k=0
            if i<2:
                ksz= self.net.ksize[i]
                s=compute_conv_output_size(self.net.map[i],self.net.ksize[i],1,pad)
                mat = np.zeros((self.net.ksize[i]*self.net.ksize[i]*self.net.in_channel[i],s*s*bsz))
                act = F.pad(self.net.act[act_key[i]], p1d, "constant", 0).detach().cpu().numpy()
         
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) #?
                            k +=1
                mat_list.append(mat)
            else:
                act = self.net.act[act_key[i]].detach().cpu().numpy()
                activation = act[0:bsz].transpose()
                mat_list.append(activation)

        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        for i in range(len(mat_list)):
            print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
        print('-'*30)
        return mat_list    

    def get_representation_matrix_for_alexnet (self, x, y=None): 
        # # Collect activations by forward pass
        # select_x, num_samples = [], 0
        # for k, (x, y) in enumerate(data_loader):
        #     select_x.append(x)
        #     num_samples += len(x)
        #     if num_samples >= 125: break
        # select_x = torch.cat(select_x)
        # select_x=select_x[0:125] # Take 125 random samples 
        # example_data = select_x
        # if self.cuda: example_data = example_data.cuda()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)
        if self.cuda: r = r.cuda()
        b=r[0:125] # Take 125 random samples 
        example_data = x[b]
        if self.cuda: example_data = example_data.cuda()

        example_out  = self.net(example_data)
    
        batch_list=[2*12,100,100,125,125] 
        mat_list=[]
        act_key=list(self.net.act.keys())
        for i in range(len(self.net.map)):
            bsz=batch_list[i]
            k=0
            if i<3:
                ksz= self.net.ksize[i]
                s=compute_conv_output_size(self.net.map[i],self.net.ksize[i])
                mat = np.zeros((self.net.ksize[i]*self.net.ksize[i]*self.net.in_channel[i],s*s*bsz))
                act = self.net.act[act_key[i]].detach().cpu().numpy()
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) 
                            k +=1
                mat_list.append(mat)
            else:
                act = self.net.act[act_key[i]].detach().cpu().numpy()
                activation = act[0:bsz].transpose()
                mat_list.append(activation)

        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        for i in range(len(mat_list)):
            print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
        print('-'*30)
        return mat_list 

    def get_representation_matrix_for_mlp (self, x, y=None): 
        # # Collect activations by forward pass
        # select_x, num_samples = [], 0
        # for k, (x, y) in enumerate(data_loader):
        #     select_x.append(x)
        #     num_samples += len(x)
        #     if num_samples >= 300: break
        # select_x = torch.cat(select_x)
        # select_x=select_x[0:300] # Take 300 random samples 
        # example_data = select_x
        # if self.cuda: example_data = example_data.cuda()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)
        if self.cuda: r = r.cuda()
        b=r[0:300] # Take random training samples
        example_data = x[b].view(-1,28*28)
        if self.cuda: example_data = example_data.cuda()

        example_out  = self.net(example_data)
    
        batch_list=[300,300,300] 
        mat_list=[] # list contains representation matrix of each layer
        act_key=list(self.net.act.keys())

        for i in range(len(act_key)):
            bsz=batch_list[i]
            act = self.net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        for i in range(len(mat_list)):
            print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
        print('-'*30)
        return mat_list

    def update_GPM (self, mat_list, threshold):
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
                if r < (activation.shape[0]/2):
                    self.feature_list.append(U[:,0:r])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:,r:])
                    self.project_type.append('retain')
        else:
            for i in range(len(mat_list)):
                if self.project_type[i] == 'remove':
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
            
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold[i]:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                        continue
                    # update GPM
                    Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                    if Ui.shape[1] > Ui.shape[0] :
                        self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                    else:
                        self.feature_list[i]=Ui
                else:
                    assert self.project_type[i] == 'retain'
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = sval_hat/sval_total

                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval >= (1-threshold[i]):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                        continue

                    # update GPM by Projected Representation (Eq-8)
                    act_feature = self.feature_list[i] - np.dot(np.dot(U[:,0:r],U[:,0:r].transpose()),self.feature_list[i])
                    Ui, Si, Vi = np.linalg.svd(act_feature)
                    self.feature_list[i]=Ui[:,:self.feature_list[i].shape[1]-r]

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                feature = self.feature_list[i]
                # ipdb.set_trace()
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:,feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i]=='retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0]/2)
            print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-'*40)

    def update_threshold(self, task_id):
        if 'alexnet' == self.arch:
            self.threshold = np.array([0.97] * 5) + task_id*np.array([0.03/self.n_tasks] * 5)
        elif 'lenet' == self.arch:
            self.threshold = np.array([0.98] * 5) + task_id*np.array([0.001] * 5)
        elif 'resnet' == self.arch:
            if self.args.dataset == 'five_datasets':
                self.threshold = np.array([0.965] * 20)
            elif self.args.dataset == 'mini_imagenet':
                self.threshold = np.array([0.992] * 20) + task_id*np.array([0.0003]*20)
            else:
                raise NotImplementedError
        elif 'mlp' == self.arch:
            self.threshold = np.array([0.95,0.99,0.99]) 

    def update_optimizer(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_offsets(self, t):
        if self.net.multi_head:
            # mapping from classes to their idx within a task
            offset1 = t * self.nc_per_task
            offset2 = min((t + 1) * self.nc_per_task, self.n_outputs)
        else:
            offset1 = 0
            offset2 = self.nc_per_task

        return offset1, offset2
