import torch
from model.base import *
import ipdb
from networks import alexnet_gpm, lenet_gpm
import torch.optim as optim

class Net(BaseNet):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)
        self.reg = args.memory_strength

        if 'cifar' in self.args.dataset:
            if args.model_arch == 'alexnet':
                self.net = alexnet_gpm.Learner(n_outputs, n_tasks)
                self.arch = 'alexnet'
            elif args.model_arch == 'lenet':
                self.net = lenet_gpm.Learner(n_outputs, n_tasks)
                self.arch = 'lenet'
            if self.cuda: self.net = self.net.cuda()
        else:
            raise NotImplementedError
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum)

        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.memx = None
        self.memy = None

        self.n_memories = args.memories

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
        self.net.train()

        # next task
        if t != self.current_task:
            self.net.zero_grad()
            
            fisher = {}
            batches = 1 if self.arch == 'mlp' else 20
            iters = 0
            for i in range(0, len(self.memx), batches):
                be, en = i, min(len(self.memx), i+batches)
                self.net.zero_grad()
                if self.net.multi_head:
                    offset1, offset2 = self.compute_offsets(self.current_task)
                    self.loss_ce((self.net(self.memx[be:en])[:, offset1: offset2]),
                             self.memy[be:en] - offset1).backward()
                else:
                    self.loss_ce(self.net(self.memx[be:en]), self.memy[be:en]).backward()

                for j, p in enumerate(self.net.parameters()):
                    pg = p.grad.data.clone().pow(2)
                    if i == 0:
                        fisher[j] = pg
                    else:
                        fisher[j] += pg
                iters += 1

            for j, p in enumerate(self.net.parameters()):
                pd = p.data.clone()
                self.optpar[j] = pd
                fisher[j]/=iters

                if self.current_task == 0:
                    self.fisher[j] = fisher[j]
                else:
                    self.fisher[j] = (self.fisher[j]*self.current_task+fisher[j])/(self.current_task+1.)
            self.current_task = t
            self.memx = None
            self.memy = None

        if self.memx is None:
            self.memx = x.data.clone()
            self.memy = y.data.clone()
        else:
            if self.memx.size(0) < self.n_memories:
                self.memx = torch.cat((self.memx, x.data.clone()))
                self.memy = torch.cat((self.memy, y.data.clone()))
                if self.memx.size(0) > self.n_memories:
                    self.memx = self.memx[:self.n_memories]
                    self.memy = self.memy[:self.n_memories]

        self.net.zero_grad()
        if self.net.multi_head:
            offset1, offset2 = self.compute_offsets(t)
            loss = self.loss_ce((self.net(x)[:, offset1: offset2]),
                            y - offset1)
        else:
            loss = self.loss_ce(self(x, t), y)

        if torch.isnan(loss): ipdb.set_trace()
        if t:
            for i, p in enumerate(self.net.parameters()):
                try:
                    l = self.reg * self.fisher[i]
                    l = l * (p - self.optpar[i]).pow(2)
                except:
                    ipdb.set_trace()
                loss += l.sum()
                if torch.isnan(loss): ipdb.set_trace()
        loss.backward()
        if 'mnist' not in self.args.dataset:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
        self.optimizer.step()

        return loss

