# Reproduce for Experience Replay (ER) with reservoir sampling in MER(Algorithm 4) https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ipdb
import torch
import warnings
import math
from model.base import *
from networks import alexnet_gpm, lenet_gpm, resnet_gpm
import torch.optim as optim

warnings.filterwarnings("ignore")

class Net(BaseNet):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)
        if 'cifar' in self.args.dataset:
            if args.model_arch == 'alexnet':
                self.net = alexnet_gpm.Learner(n_outputs, n_tasks)
                self.arch = 'alexnet'
            if self.cuda: self.net = self.net.cuda()
        elif 'five' in args.dataset or 'imagenet' in args.dataset:
            if args.model_arch == 'resnet':
                self.net = resnet_gpm.Learner(n_outputs, n_tasks, nf=20, dataset = args.dataset)
                self.arch = 'resnet'
            if self.cuda: self.net = self.net.cuda()
        else:
            raise NotImplementedError
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum)

        if args.model_arch == 'alexnet_acl':
            self.optimizer = self.get_optimizer(0)
        self.iter = 0

    def forward(self, x, t=-1):
        if self.arch == 'alexnet_acl':
            output = self.net.forward(x,x,t)
        else:
            output = self.net.forward(x)

        if self.net.multi_head:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def get_optimizer(self, task_id):
        try:
            self.optimizer = torch.optim.SGD(list(self.net.shared.parameters())+
                                    list(self.net.private.task_out[2*task_id].parameters())+
                                    list(self.net.private.task_out[2*task_id+1].parameters()) +
                                    list(self.net.head[task_id].parameters()),
                                    momentum=self.args.momentum,
                                    weight_decay=0.01, lr=self.lr)
        except:
            ipdb.set_trace()

    def observe(self, x, y, t, writer=None):
        if t != self.current_task:
            self.current_task = t
            self.iter = 0
            if self.arch == 'alexnet_acl':
                self.get_optimizer(self.current_task)

        # ipdb.set_trace()
        for pass_itr in range(self.glances):

            self.zero_grads()

            if self.arch == 'mlp' and writer != None:
                loss = self.loss_ce(self.net(x), y)
                loss.backward()
                for k, (m, params) in enumerate(self.net.named_parameters()):
                    writer.add_scalar(f"4.Grad-Norm/task-{t}/layer-{k}", params.grad.norm(p=2), self.iter)
                self.zero_grads()

            # Draw batch from buffer:
            bx, by, bt = self.get_batch(x, y, t)

            self.iter += 1

            # FS-ER
            if self.args.sharpness:
                bsize = len(bx)
                inner_sz = math.ceil(bsize / self.args.inner_batches)
                meta_losses = torch.zeros(self.args.inner_batches).float()
                fast_weights = None
                k = 0

                # inner step
                for j in range(0, len(bx), inner_sz):

                    if j + inner_sz <= bsize:
                        batch_x = bx[j: j + inner_sz]
                        batch_y = by[j: j + inner_sz]
                        batch_t = bt[j: j + inner_sz]
                    else:
                        batch_x = bx[j:]
                        batch_y = by[j:]
                        batch_t = bt[j:]

                    # samples for sharpness are from the current task
                    fast_weights = self.update_weight(batch_x, batch_y, t, fast_weights)

                    # samples for tiny update are from the current task and old tasks
                    meta_losses[k] = self.meta_loss(bx, by, bt, fast_weights)
                    k += 1

                # Taking the tiny gradient step (will update the lambdas)
                self.zero_grads()
                loss = torch.mean(meta_losses)
            else:
                loss = self.meta_loss(bx, by, bt)
                self.zero_grads()

            loss.backward()
            if self.arch == 'mlp':
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

            # if self.arch == 'mlp' and writer != None:
            #     for k, (m, params) in enumerate(self.net.named_parameters()):
            #         writer.add_scalar(f"4.Grad-Norm/task-{t}/layer-{k}", params.grad.norm(p=2), self.iter)

            self.optimizer.step()

        if self.real_epoch == 0:
            self.push_to_mem(x, y, torch.tensor(t))

        return loss

    def zero_grads(self):
        self.optimizer.zero_grad()
        self.net.zero_grad()

    def update_weight(self, x, y, t, fast_weights):
        # Forward
        loss = self.take_loss(x, y, t, fast_weights)
        if fast_weights is None:
            fast_weights = self.net.get_params()

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.args.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required,
                                         allow_unused=True))
        for i in range(len(grads)):
            if grads[i] is not None:
                grads[i] = torch.clamp(grads[i], min=-self.args.grad_clip_norm, max=self.args.grad_clip_norm)

        fast_weights = list(
            map(lambda p: p[1] + p[0] * self.args.eta1 if p[0] is not None else p[1], zip(grads, fast_weights)))

        return fast_weights

    def meta_loss(self, x, y, tasks):
        """
            Get loss of multiple tasks tasks
            """
        if self.arch == 'alexnet_acl':
            outputs = self.net.forward(x,x,self.current_task)
        else:
            outputs = self.net.forward(x)
        loss = 0.0
        for task in np.unique(tasks.data.cpu().numpy()):
            task = int(task)
            idx = torch.nonzero(tasks == task).view(-1)

            if self.net.multi_head:
                offset1, offset2 = self.compute_offsets(task)
                loss += self.loss_ce(outputs[idx, offset1:offset2], y[idx] - offset1) * len(idx)
            else:
                loss += self.loss_ce(outputs[idx], y[idx]) * len(idx)

        return loss/len(y)

    def take_loss(self, x, y, t):
        """
            Get loss of a task t
            """
        outputs = self.net.forward(x)
        if self.net.multi_head:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            loss = self.loss_ce(outputs[:, offset1:offset2], y - offset1)
        else:
            loss = self.loss_ce(outputs, y)

        return loss
