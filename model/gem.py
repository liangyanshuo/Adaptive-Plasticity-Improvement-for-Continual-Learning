
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import optim

from model.base import *
from networks import alexnet_gpm, lenet_gpm
import numpy as np
import quadprog

# Auxiliary functions useful for GEM's inner optimization.
def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1

def store_param(pp, params, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    params[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            params[beg: en, tid].copy_(param.data.view(-1))
        cnt += 1

def get_param(pp, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    parameter = torch.Tensor(sum(grad_dims)).fill_(0.0).cuda()
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            parameter[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return parameter

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            try:
                this_grad = newgrad[beg: en].contiguous().view(
                    param.grad.data.size())
            except:
                print(newgrad[beg: en])
                print(grad_dims)
                print(cnt)
                print(param.size())
                raise Exception('Wrong')
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    # v=quadprog_solve_qp()[t-1](P,q,G,h)
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

class Net(BaseNet):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)
        self.arch = args.model_arch
        if 'cifar' in self.args.dataset:
            if self.arch == 'alexnet':
                self.net = alexnet_gpm.Learner(n_outputs, n_tasks)
            elif self.arch == 'lenet':
                self.net = lenet_gpm.Learner(n_outputs, n_tasks)
                self.net.apply(init_weights)
            self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum)
            if self.cuda: self.net = self.net.cuda()
        else:
            raise NotImplementedError
        self.iter = 0
        
        self.margin = args.memory_strength

        self.n_memories = args.memories // n_tasks
        self.gpu = args.cuda

        # allocate episodic memory
        if 'cifar' in args.dataset:
            input_size = [3, 32, 32]
        elif 'mnist' in args.dataset:
            input_size = [1, 28, 28]
        self.memory_data = torch.FloatTensor(
            n_tasks, self.n_memories, *input_size)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.mem_cnt = 0

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

    def observe(self, x, y, t, writer=None):
        # update memory
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t
        
        for iter in range(self.glances):
            # compute gradient on previous tasks
            if len(self.observed_tasks) > 1:
                for tt in range(len(self.observed_tasks) - 1):
                    self.zero_grad()
                    # fwd/bwd on the examples in the memory
                    past_task = self.observed_tasks[tt]
    
                    offset1, offset2 = self.compute_offsets(past_task)
                    ptloss = self.loss_ce(
                        self.forward(
                            self.memory_data[past_task],
                            past_task)[:, offset1: offset2],
                        self.memory_labs[past_task] - offset1)
                    ptloss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_clip_norm)
                    store_grad(self.parameters, self.grads, self.grad_dims,
                               past_task)
    
            # now compute the grad on the current minibatch
            self.zero_grad()
    
            offset1, offset2 = self.compute_offsets(t)
            loss = self.loss_ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_clip_norm)

            # check if gradient violates constraints
            if len(self.observed_tasks) > 1:
                # copy gradient
                store_grad(self.parameters, self.grads, self.grad_dims, t)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                    else torch.LongTensor(self.observed_tasks[:-1])
                dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                                self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    project2cone2(self.grads[:, t].unsqueeze(1),
                                  self.grads.index_select(1, indx), self.margin)
                    # copy gradients back
                    overwrite_grad(self.parameters, self.grads[:, t],
                                   self.grad_dims)
            self.optimizer.step()

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            (x.data[: effbsz]))
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        return loss
