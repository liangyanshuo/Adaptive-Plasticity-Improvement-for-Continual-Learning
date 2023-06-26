from socket import IPPROTO_FRAGMENT
import time
import importlib
import numpy as np
from numpy.lib.shape_base import expand_dims
from scipy.stats.stats import mode
import torch
import copy
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
import parser as file_parser
import utils.utils as utils
import utils.plot1D as plot
import os
import logging
import ipdb


def eval(model, x, y, t, args):
    model.net.eval()

    total_loss = 0
    total_acc = 0
    idx = np.arange(x.size(0))
    if args.model not in ['ccll', 'rkr', 'lrw']:
        np.random.shuffle(idx)
    idx = torch.LongTensor(idx)

    with torch.no_grad():
        # Loop batches
        for i in range(0, len(idx), args.test_batch_size):
            if i + args.test_batch_size <= len(idx):
                pos = idx[i: i + args.test_batch_size]
            else:
                pos = idx[i:]

            images = x[pos]
            targets = y[pos]
            if args.cuda:
                images = images.cuda()
                targets = targets.cuda()

            outputs = model(images, t)
            if model.net.multi_head:
                offset1, offset2 = model.compute_offsets(t)
                if 'five' in args.dataset:
                    loss = model.loss_ce(outputs[:, offset1:offset2], targets)
                    targets += offset1
                else:
                    loss = model.loss_ce(outputs[:, offset1:offset2], targets - offset1)
            else:
                loss = model.loss_ce(outputs, targets)

            _, p = torch.max(outputs.data.cpu(), 1, keepdim=False)
            total_loss += loss.detach() * len(pos)
            total_acc += (p == targets.cpu()).float().sum()

    return total_loss / len(x), total_acc / len(x)


def life_experience(model, data, ids, args, logger):
    time_start = time.time()

    # store accuravy & loss for all tasks
    acc = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)
    lss = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)
    tasks = np.arange(args.n_tasks, dtype=np.int32)

    # visual landscape
    if args.visual_landscape:
        steps = np.arange(args.step_min, args.step_max, args.step_size)
        visual_lss = np.zeros((args.n_tasks, args.n_tasks, args.dir_num, len(steps)), dtype=np.float32)
        visual_val_acc = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)
        visual_train_acc = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)

    # tensorboard & checkpoint
    args.log_dir, args.checkpoint_dir = utils.log_dir(args)
    writer = SummaryWriter(args.log_dir)

    # train/val/test order by ids
    # t: the real task id
    for i, t in enumerate(ids):
        # Get data
        xtrain = data[t]['train']['x']
        ytrain = data[t]['train']['y']
        xvalid = data[t]['valid']['x']
        yvalid = data[t]['valid']['y']
        task = t

        assert xtrain.shape[0] == ytrain.shape[0] and xvalid.shape[0] == yvalid.shape[0]

        if args.cuda:
            xtrain = xtrain.cuda()
            ytrain = ytrain.cuda()
            xvalid = xvalid.cuda()
            yvalid = yvalid.cuda()

        print('*' * 100)
        print('>>>Task {:2d}({:s}) | Train: {:5d}, Val: {:5d}, Test: {:5d}<<<'.format(i, data[t]['name'],
                                   ytrain.shape[0], yvalid.shape[0], data[t]['test']['y'].shape[0]))
        print('*' * 100)

        # Train
        clock0 = time.time()
        # bn's parameters are only learned for the first task
        if args.freeze_bn and i == 1:
            for m in model.net.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        # reset the learning rate
        lr = args.lr
        model.update_optimizer(lr)
        if args.model == 'fsdgpm':
            model.eta1 = args.eta1
            if len(model.M_vec) > 0 and args.method in ['dgpm', 'xdgpm']:
                # reset lambda
                model.eta2 = args.eta2
                model.define_lambda_params()
                model.update_opt_lambda(model.eta2)

        # if use early stop, then start training new tasks from the optimal model
        if args.earlystop:
            best_acc = 0.
            patience = args.lr_patience
            best_model = copy.deepcopy(model.net.state_dict())

        if args.model in ['api', 'api_i', 'api_i_1', 'api_i_2'] and i!=0:
            # train the model for a loop to count the gradient norm
            model.net.train()
            epochs = 1
            for ep in range(epochs):
                idx = np.arange(xtrain.size(0))
                np.random.shuffle(idx)
                idx = torch.LongTensor(idx)
                for bi in range(0, len(idx), args.batch_size):
                    if bi + args.batch_size <= len(idx):
                        pos = idx[bi: bi + args.batch_size]
                    else:
                        pos = idx[bi:]

                    v_x = xtrain[pos]
                    v_y = ytrain[pos]

                    loss = model.observe(v_x, v_y, t, evalue=True)
            if args.model in ['api', 'api_i', 'api_i_1', 'api_i_2']:
                model.expand_feature_space(args, xtrain, ytrain, writer=writer)
            else:
                model.expand_feature_space(args, t)
        
        # if args.model in ['gpm', 'dualgpm'] and i!=0:
        #     # train the model for a loop to count the gradient norm
        #     model.net.train()
        #     epochs = 1
        #     for ep in range(epochs):
        #         idx = np.arange(xtrain.size(0))
        #         np.random.shuffle(idx)
        #         idx = torch.LongTensor(idx)
        #         for bi in range(0, len(idx), args.batch_size):
        #             if bi + args.batch_size <= len(idx):
        #                 pos = idx[bi: bi + args.batch_size]
        #             else:
        #                 pos = idx[bi:]
        #             v_x = xtrain[pos]
        #             v_y = ytrain[pos]
        #             loss = model.observe(v_x, v_y, t, evalue=True, writer=writer)
            
        prog_bar = tqdm(range(args.n_epochs))
        for ep in prog_bar:
            # train
            model.epoch += 1
            model.real_epoch = ep

            model.net.train()
            idx = np.arange(xtrain.size(0))
            np.random.shuffle(idx)
            idx = torch.LongTensor(idx)
            train_loss = 0.0

            # Loop batches
            # clock0=time.time()
            for bi in range(0, len(idx), args.batch_size):
                if bi + args.batch_size <= len(idx):
                    pos = idx[bi: bi + args.batch_size]
                else:
                    pos = idx[bi:]
                v_x = xtrain[pos]
                v_y = ytrain[pos]

                # ipdb.set_trace()
                loss = model.observe(v_x, v_y, t, writer=writer, stage=1)
                train_loss += loss * len(v_x)

            train_loss = train_loss / len(xtrain)
            writer.add_scalar(f"1.Train-LOSS/{data[t]['name']}", round(train_loss.item(), 5), model.epoch)
            # ipdb.set_trace()
            
            # if use early stop, we need to adapt lr and store the best model
            if args.earlystop:
                # Valid
                valid_loss, valid_acc = eval(model, xvalid, yvalid, t, args)
                # train_loss, train_acc = eval(model, xtrain, ytrain, t, args)
                # print(train_loss, train_acc, valid_loss, valid_acc)
                writer.add_scalar(f"2.Val-LOSS/{data[t]['name']}", round(valid_loss.item(), 5), model.epoch)
                writer.add_scalar(f"2.Val-ACC/{data[t]['name']}", 100 * valid_acc, model.epoch)

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = copy.deepcopy(model.net.state_dict())
                    patience = args.lr_patience
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= args.lr_factor
                        print(' lr={:.1e} |'.format(lr), end='')
                        if lr < args.lr_min:
                            break
                        patience = args.lr_patience
                        model.update_optimizer(lr)
                        if args.model == 'fsdgpm':
                            model.eta1 = model.eta1 / args.lr_factor
                            if len(model.M_vec) > 0 and args.method in ['dgpm', 'xdgpm']:
                                model.eta2 = model.eta2 / args.lr_factor
                                model.update_opt_lambda(model.eta2)

                    prog_bar.set_description(
                        "Task: {} | Epoch: {}/{} | time={:2.2f}s | Train: loss={:.3f} | Valid: loss={:.3f}, acc={:5.1f}% |".format(
                            i, ep + 1, model.n_epochs, time.time() - clock0, round(train_loss.item(), 5),
                            round(valid_loss.item(), 5), 100 * valid_acc)
                    )
            else:
                prog_bar.set_description("Task: {} | Epoch: {}/{} | time={:2.2f}s | Train: loss={:.3f} |".format(
                        i, ep + 1, model.n_epochs, time.time() - clock0, round(train_loss.item(), 5))
                    )
            

        if args.earlystop:
            model.net.load_state_dict(copy.deepcopy(best_model))

        # ipdb.set_trace()

        print('-' * 60)
        print('Total Epoch: {}/{} | Training Time: {:.2f} min | Last Lr: {}'.format(ep + 1, model.n_epochs,
                                                                                    (time.time() - clock0) / 60, lr))
        print('-' * 60)

        # Test
        clock1 = time.time()
        inference_time = []
        for u in range(i + 1):
            xtest = data[ids[u]]['test']['x']
            ytest = data[ids[u]]['test']['y']

            if args.cuda:
                xtest = xtest.cuda()
                ytest = ytest.cuda()

            if args.dataset == 'cifar100_20' and args.model == 'expand_gpm2_4':
                test_loss, test_acc = 0, 0
                for _ in range(10):
                    test_loss_, test_acc_ = eval(model, xtest, ytest, ids[u], args)
                    test_loss += test_loss_
                    test_acc += test_acc_
                test_loss /= 10
                test_acc /= 10
            else:
                test_start = time.time()
                test_loss, test_acc = eval(model, xtest, ytest, ids[u], args)
                test_end = time.time()
                inference_time.append(test_end-test_start)

            # ipdb.set_trace()
            acc[i, u] = test_acc
            lss[i, u] = test_loss

            writer.add_scalar(f"0.Test-LOSS/{data[ids[u]]['name']}", test_loss, i)
            writer.add_scalar(f"0.Test-ACC/{data[ids[u]]['name']}", 100 * test_acc, i)
            writer.add_scalar(f"0.Test-BWT/{data[ids[u]]['name']}", 100 * (test_acc - acc[u, u]), i)

        avg_acc = sum(acc[i]) / (i + 1)
        bwt = np.mean((acc[i]-np.diag(acc)))

        writer.add_scalar(f"0.Test/Avg-ACC", 100 * avg_acc, i)
        writer.add_scalar(f"0.Test/Avg-BWT", 100 * bwt, i)

        print('-' * 60)
        print('Accuracies =')
        # ipdb.set_trace()
        for i_a in range(i + 1):
            print('\t',end='')
            for j_a in range(len(acc[i_a])):
                print('{:5.1f}% '.format(acc[i_a, j_a]*100),end='')
            print()
        print('-' * 60)

        memory_space = args.memories
        # Update Memory of Feature Space
        if args.model in ['fsdgpm']:
            clock2 = time.time()

            # Get Thres
            thres_value = min(args.thres + i * args.thres_add, args.thres_last)
            thres = np.array([thres_value] * model.net.n_rep)
            print('-' * 60)
            print('Threshold: ', thres)

            # Update basis of Feature Space
            model.set_gpm_by_svd(thres)

            # Get the info of mem
            for p in range(len(model.M_vec)):
                writer.add_scalar(f"3.MEM-Total/Layer_{p}", model.M_vec[p].shape[1], i)

            print('Spend Time = {:.2f} s'.format(time.time() - clock2))
            print('-' * 60)

        elif args.model in ['gpm']:
            # Threshold Update
            model.update_threshold(i)

            # Memory Update
            if model.arch == 'alexnet':
                mat_list = model.get_representation_matrix_for_alexnet (xtrain, ytrain)
            elif model.arch == 'mlp':
                mat_list = model.get_representation_matrix_for_mlp (xtrain, ytrain)
            elif model.arch == 'resnet':
                mat_list = model.get_representation_matrix_ResNet18 (xtrain, ytrain)
            elif model.arch == 'lenet':
                mat_list = model.get_representation_matrix_lenet (xtrain, ytrain)
            else:
                raise NotImplementedError

            model.update_GPM (mat_list, model.threshold)
            # Projection Matrix Precomputation
            model.feature_mat = []
            for p in range(len(model.feature_list)):
                Uf=torch.Tensor(np.dot(model.feature_list[p],model.feature_list[p].transpose()))
                if model.cuda: Uf = Uf.cuda()
                print('Layer {} - Projection Matrix shape: {}'.format(p+1,Uf.shape))
                model.feature_mat.append(Uf)

            # Get the info of mem
            for p in range(len(model.feature_list)):
                writer.add_scalar(f"3.MEM-Total/Layer_{p}", model.feature_list[p].shape[1], i)

            # Get the info of mem
            memory_space = 0
            for p in range(len(model.feature_list)):
                memory_space += model.feature_list[p].shape[1] * model.feature_list[p].shape[0]
            if 'mnist' in args.dataset:
                memory_space = memory_space/784.
            elif 'cifar' in args.dataset:
                memory_space = memory_space/3072.
            elif 'five' in args.dataset:
                memory_space = memory_space/3072.
            elif 'imagenet' in args.dataset:
                memory_space = memory_space/7056.
            else:
                raise NotImplementedError
            writer.add_scalar(f"3.MEM-Total/Memory", memory_space, i)

            if args.model in ['trgp']:
                scale_space = 0
                for ii in range(t+1):
                    # select the regime 2, which need to learn scale
                    if ii > 0:
                        for i in range(len(model.space1)):
                            for k, task_sel in enumerate(model.memory[ii][str(i)]['selected_task']):
                                # print(memory[task_name]['regime'][task_sel])
                                if model.memory[ii][str(i)]['regime'][task_sel] == '2':
                                    scale_space += model.memory[task_sel][str(i)]['space_list'].shape[1]*model.memory[task_sel][str(i)]['space_list'].shape[1]
                writer.add_scalar(f"3.MEM-Total/Scale", scale_space, i)
        elif args.model in ['dualgpm', 'api', 'api_i', 'api_i_1', 'api_i_2']:
            # Threshold Update
            model.update_threshold(i)

            # Memory Update  
            if model.arch == 'alexnet' or model.arch == 'alexnet2':
                mat_list = model.get_representation_matrix_for_alexnet (xtrain, ytrain)
            elif model.arch == 'mlp':
                mat_list = model.get_representation_matrix_for_mlp (xtrain, ytrain)
            elif model.arch == 'lenet':
                mat_list = model.get_representation_matrix_lenet (xtrain, ytrain)
            elif model.arch == 'resnet':
                mat_list = model.get_representation_matrix_ResNet18(xtrain, ytrain)
            else:
                raise NotImplementedError

            model.update_GPM (mat_list, model.threshold)

            # Get the info of mem
            base_memory = 0
            for p in range(len(model.feature_list)):
                base_memory += model.feature_list[p].shape[1] * model.feature_list[p].shape[0]
            if 'mnist' in args.dataset:
                base_memory = base_memory/784.
            elif 'cifar' in args.dataset:
                base_memory = base_memory/3072.
            elif 'five' in args.dataset:
                base_memory = base_memory/3072.
            else:
                raise NotImplementedError
            writer.add_scalar(f"3.MEM-Total/base_Memory", base_memory, i)

            param_memory = 0
            if args.model_arch == 'mlp':
                param_memory += (model.net.lin1.weight.size(1) - 784)*100
                param_memory += (model.net.lin2.weight.size(1) - 100)*100
                param_memory += (model.net.fc1.weight.size(1) - 100)*10
            elif args.model_arch == 'alexnet':
                param_memory += (model.net.conv1.weight.size(1) - 3)*64*16
                param_memory += (model.net.conv2.weight.size(1) - 64)*128*9
                param_memory += (model.net.conv3.weight.size(1) - 128)*256*4
                param_memory += (model.net.fc1.weight.size(1) - 1024)*2048
                param_memory += (model.net.fc2.weight.size(1) - 2048)*2048
            elif args.model_arch == 'lenet':
                param_memory += (model.net.conv1.weight.size(1) - 3)*20*25
                param_memory += (model.net.conv2.weight.size(1) - 20)*50*25
                param_memory += (model.net.fc1.weight.size(1) - 3200)*800
                param_memory += (model.net.fc2.weight.size(1) - 800)*500
            elif args.model_arch == 'resnet':
                param_memory += (model.net.conv1.weight.size(1)-3)*20*9

                param_memory += (model.net.layer1[0].conv1.weight.size(1)-20)*20*9
                param_memory += (model.net.layer1[0].conv2.weight.size(1)-20)*20*9
                param_memory += (model.net.layer1[1].conv1.weight.size(1)-20)*20*9
                param_memory += (model.net.layer1[1].conv2.weight.size(1)-20)*20*9

                param_memory += (model.net.layer2[0].conv1.weight.size(1)-20)*40*9
                param_memory += (model.net.layer2[0].conv2.weight.size(1)-40)*40*9
                param_memory += (model.net.layer2[0].conv3.weight.size(1)-20)*40*1
                param_memory += (model.net.layer2[1].conv1.weight.size(1)-40)*40*9
                param_memory += (model.net.layer2[1].conv2.weight.size(1)-40)*40*9

                param_memory += (model.net.layer3[0].conv1.weight.size(1)-40)*80*9
                param_memory += (model.net.layer3[0].conv2.weight.size(1)-80)*80*9
                param_memory += (model.net.layer3[0].conv3.weight.size(1)-40)*80*1
                param_memory += (model.net.layer3[1].conv1.weight.size(1)-80)*80*9
                param_memory += (model.net.layer3[1].conv2.weight.size(1)-80)*80*9

                param_memory += (model.net.layer4[0].conv1.weight.size(1)-80)*160*9
                param_memory += (model.net.layer4[0].conv2.weight.size(1)-160)*160*9
                param_memory += (model.net.layer4[0].conv3.weight.size(1)-80)*160*1
                param_memory += (model.net.layer4[1].conv1.weight.size(1)-160)*160*9
                param_memory += (model.net.layer4[1].conv2.weight.size(1)-160)*160*9
        
            # ipdb.set_trace()
            
            if args.model in ['api', 'api_i', 'api_i_1', 'api_i_2']:
                if args.model_arch == 'mlp':
                    param_memory += (model.net.lin1.weight.size(1) - 784)*784
                    param_memory += (model.net.lin2.weight.size(1) - 100)*100
                    param_memory += (model.net.fc1.weight.size(1) - 100)*100
                elif args.model_arch == 'alexnet':
                    param_memory += (model.net.conv1.weight.size(1) - 3)*3
                    param_memory += (model.net.conv2.weight.size(1) - 64)*64
                    param_memory += (model.net.conv3.weight.size(1) - 128)*128
                    param_memory += (model.net.fc1.weight.size(1) - 1024)*1024
                    param_memory += (model.net.fc2.weight.size(1) - 2048)*2048
                elif args.model_arch == 'lenet':
                    param_memory += (model.net.conv1.weight.size(1) - 3)*3
                    param_memory += (model.net.conv2.weight.size(1) - 20)*20
                    param_memory += (model.net.fc1.weight.size(1) - 3200)*3200
                    param_memory += (model.net.fc2.weight.size(1) - 800)*800
                elif args.model_arch == 'resnet':
                    param_memory += (model.net.conv1.weight.size(1)-3)*3

                    param_memory += (model.net.layer1[0].conv1.weight.size(1)-20)*20
                    param_memory += (model.net.layer1[0].conv2.weight.size(1)-20)*20
                    param_memory += (model.net.layer1[1].conv1.weight.size(1)-20)*20
                    param_memory += (model.net.layer1[1].conv2.weight.size(1)-20)*20

                    param_memory += (model.net.layer2[0].conv1.weight.size(1)-20)*20
                    param_memory += (model.net.layer2[0].conv2.weight.size(1)-40)*40
                    param_memory += (model.net.layer2[0].conv3.weight.size(1)-20)*20
                    param_memory += (model.net.layer2[1].conv1.weight.size(1)-40)*40
                    param_memory += (model.net.layer2[1].conv2.weight.size(1)-40)*40

                    param_memory += (model.net.layer3[0].conv1.weight.size(1)-40)*40
                    param_memory += (model.net.layer3[0].conv2.weight.size(1)-80)*80
                    param_memory += (model.net.layer3[0].conv3.weight.size(1)-40)*40
                    param_memory += (model.net.layer3[1].conv1.weight.size(1)-80)*80
                    param_memory += (model.net.layer3[1].conv2.weight.size(1)-80)*80

                    param_memory += (model.net.layer4[0].conv1.weight.size(1)-80)*80
                    param_memory += (model.net.layer4[0].conv2.weight.size(1)-160)*160
                    param_memory += (model.net.layer4[0].conv3.weight.size(1)-80)*80
                    param_memory += (model.net.layer4[1].conv1.weight.size(1)-160)*160
                    param_memory += (model.net.layer4[1].conv2.weight.size(1)-160)*160

            # ipdb.set_trace()
            if 'mnist' in args.dataset:
                param_memory = param_memory/784.
            elif 'cifar' in args.dataset:
                param_memory = param_memory/3072.
            elif 'five' in args.dataset:
                param_memory = param_memory/3072.
            else:
                raise NotImplementedError
            writer.add_scalar(f"3.MEM-Total/param_Memory", param_memory, i)

            writer.add_scalar(f"3.MEM-Total/Total_Memory", param_memory+base_memory, i)

            if args.model in ['dualgpm', 'api', 'api_i', 'api_i_1', 'api_i_2']:
                model.feature_mat = []
                # Projection Matrix Precomputation
                for p in range(len(model.feature_list)):
                    if model.feature_list[p].shape[1]>0:
                        Uf=torch.Tensor(np.dot(model.feature_list[p],model.feature_list[p].transpose()))
                    else:
                        assert model.project_type[p] == 'retain'
                        Uf=torch.zeros(model.feature_list[p].shape[0], model.feature_list[p].shape[0])
                    print('Layer {} - Projection Matrix shape: {}'.format(p+1,Uf.shape))
                    if model.cuda: Uf = Uf.cuda()
                    model.feature_mat.append(Uf)

        if args.visual_landscape:
            visual_lss[i], visual_val_acc[i], visual_train_acc[i] = plot.calculate_loss(model, data, ids, i, steps, args)

    time_end = time.time()
    time_spent = time_end - time_start

    print('*' * 100)
    print('>>> Final Test Result: ACC={:5.3f}%, BWT={:5.3f}%, Total time = {:.2f} min<<<'.format(
        100 * avg_acc, 100 * bwt, time_spent / 60))
    print('*' * 100)

    memory_space, param_space = args.memories, 0
    if args.model in ['gpm', 'dualgpm', 'api', 'api_i', 'api_i_1', 'api_i_2']:
        param_space = 0
        # ipdb.set_trace()
        for p in range(len(model.feature_list)):
            param_space += model.feature_list[p].shape[1] * model.feature_list[p].shape[0]

        memory_space = 0
        if args.model in ['api', 'api_i', 'api_i_1', 'api_i_2']:
            if args.model_arch == 'mlp':
                memory_space += (model.net.lin1.weight.size(1) - 784)*100
                memory_space += (model.net.lin2.weight.size(1) - 100)*100
                memory_space += (model.net.fc1.weight.size(1) - 100)*10
            elif args.model_arch == 'alexnet':
                memory_space += (model.net.conv1.weight.size(1) - 3)*64*16
                memory_space += (model.net.conv2.weight.size(1) - 64)*128*9
                memory_space += (model.net.conv3.weight.size(1) - 128)*256*4
                memory_space += (model.net.fc1.weight.size(1) - 1024)*2048
                memory_space += (model.net.fc2.weight.size(1) - 2048)*2048
            elif args.model_arch == 'lenet':
                memory_space += (model.net.conv1.weight.size(1) - 3)*20*25
                memory_space += (model.net.conv2.weight.size(1) - 20)*50*25
                memory_space += (model.net.fc1.weight.size(1) - 3200)*800
                memory_space += (model.net.fc2.weight.size(1) - 800)*500
            elif args.model_arch == 'resnet':
                memory_space += (model.net.conv1.weight.size(1)-3)*20*9

                memory_space += (model.net.layer1[0].conv1.weight.size(1)-20)*20*9
                memory_space += (model.net.layer1[0].conv2.weight.size(1)-20)*20*9
                memory_space += (model.net.layer1[1].conv1.weight.size(1)-20)*20*9
                memory_space += (model.net.layer1[1].conv2.weight.size(1)-20)*20*9 

                memory_space += (model.net.layer2[0].conv1.weight.size(1)-20)*40*9
                memory_space += (model.net.layer2[0].conv2.weight.size(1)-40)*40*9
                memory_space += (model.net.layer2[0].conv3.weight.size(1)-20)*40*1
                memory_space += (model.net.layer2[1].conv1.weight.size(1)-40)*40*9
                memory_space += (model.net.layer2[1].conv2.weight.size(1)-40)*40*9

                memory_space += (model.net.layer3[0].conv1.weight.size(1)-40)*80*9
                memory_space += (model.net.layer3[0].conv2.weight.size(1)-80)*80*9
                memory_space += (model.net.layer3[0].conv3.weight.size(1)-40)*80*1
                memory_space += (model.net.layer3[1].conv1.weight.size(1)-80)*80*9
                memory_space += (model.net.layer3[1].conv2.weight.size(1)-80)*80*9

                memory_space += (model.net.layer4[0].conv1.weight.size(1)-80)*160*9
                memory_space += (model.net.layer4[0].conv2.weight.size(1)-160)*160*9
                memory_space += (model.net.layer4[0].conv3.weight.size(1)-80)*160*1
                memory_space += (model.net.layer4[1].conv1.weight.size(1)-160)*160*9
                memory_space += (model.net.layer4[1].conv2.weight.size(1)-160)*160*9


        if args.model in ['api', 'api_i', 'api_i_1', 'api_i_2']:
            if args.model_arch == 'mlp':
                memory_space += (model.net.lin1.weight.size(1) - 784)*784
                memory_space += (model.net.lin2.weight.size(1) - 100)*100
                memory_space += (model.net.fc1.weight.size(1) - 100)*100
            elif args.model_arch == 'alexnet' or args.model_arch == 'alexnet2':
                memory_space += (model.net.conv1.weight.size(1) - 3)*3
                memory_space += (model.net.conv2.weight.size(1) - 64)*64
                memory_space += (model.net.conv3.weight.size(1) - 128)*128
                memory_space += (model.net.fc1.weight.size(1) - 1024)*1024
                memory_space += (model.net.fc2.weight.size(1) - 2048)*2048
            elif args.model_arch == 'lenet':
                memory_space += (model.net.conv1.weight.size(1) - 3)*3
                memory_space += (model.net.conv2.weight.size(1) - 20)*20
                memory_space += (model.net.fc1.weight.size(1) - 3200)*3200
                memory_space += (model.net.fc2.weight.size(1) - 800)*800
            elif args.model_arch == 'resnet':
                memory_space += (model.net.conv1.weight.size(1)-3)*3

                memory_space += (model.net.layer1[0].conv1.weight.size(1)-20)*20
                memory_space += (model.net.layer1[0].conv2.weight.size(1)-20)*20
                memory_space += (model.net.layer1[1].conv1.weight.size(1)-20)*20
                memory_space += (model.net.layer1[1].conv2.weight.size(1)-20)*20 

                memory_space += (model.net.layer2[0].conv1.weight.size(1)-20)*20
                memory_space += (model.net.layer2[0].conv2.weight.size(1)-40)*40
                memory_space += (model.net.layer2[0].conv3.weight.size(1)-20)*20
                memory_space += (model.net.layer2[1].conv1.weight.size(1)-40)*40
                memory_space += (model.net.layer2[1].conv2.weight.size(1)-40)*40

                memory_space += (model.net.layer3[0].conv1.weight.size(1)-40)*40
                memory_space += (model.net.layer3[0].conv2.weight.size(1)-80)*80
                memory_space += (model.net.layer3[0].conv3.weight.size(1)-40)*40
                memory_space += (model.net.layer3[1].conv1.weight.size(1)-80)*80
                memory_space += (model.net.layer3[1].conv2.weight.size(1)-80)*80

                memory_space += (model.net.layer4[0].conv1.weight.size(1)-80)*80
                memory_space += (model.net.layer4[0].conv2.weight.size(1)-160)*160
                memory_space += (model.net.layer4[0].conv3.weight.size(1)-80)*80
                memory_space += (model.net.layer4[1].conv1.weight.size(1)-160)*160
                memory_space += (model.net.layer4[1].conv2.weight.size(1)-160)*160


        memory_space += param_space
        if 'mnist' in args.dataset:
            memory_space = memory_space/784.
            logger.info('run:memory_space:{}, param_space:{}'.format(memory_space, memory_space-(param_space/784.)))
        elif 'cifar' in args.dataset or 'five' in args.dataset:
            memory_space = memory_space/3072.
            logger.info('run:memory_space:{}, param_space:{}'.format(memory_space, memory_space-(param_space/3072.)))
        else:
            raise NotImplementedError
        
    elif args.model in ['trgp']:
        base_space = 0
        # ipdb.set_trace()
        for p in range(len(model.feature_list)):
            base_space += model.feature_list[p].shape[1] * model.feature_list[p].shape[0]

        scale_space = 0
        for ii in range(len(ids)):
            # select the regime 2, which need to learn scale
            if ii > 0:
                for i in range(len(model.space1)):
                    for k, task_sel in enumerate(model.memory[ii][str(i)]['selected_task']):
                        # print(memory[task_name]['regime'][task_sel])
                        if model.memory[ii][str(i)]['regime'][task_sel] == '2':
                            scale_space += model.memory[task_sel][str(i)]['space_list'].shape[1]*model.memory[task_sel][str(i)]['space_list'].shape[1]

        memory_space = base_space + scale_space
        if 'mnist' in args.dataset:
            base_space =base_space/784.
            scale_space = scale_space/784.
            memory_space = base_space + scale_space
            logger.info('run:memory_space:{}, base_space:{}, param_space:{}'.format(memory_space, base_space, scale_space))
        elif 'cifar' in args.dataset or 'five' in args.dataset:
            base_space =base_space/3072.
            scale_space = scale_space/3072.
            memory_space = base_space + scale_space
            logger.info('run:memory_space:{}, base_space:{}, param_space:{}'.format(memory_space, base_space, scale_space))
        else:
            raise NotImplementedError
    elif args.model in ['ER']:
        logger.info('run:memory_space:{}'.format(args.memories))


    # plot & save
    if args.visual_landscape:
        timestamp = utils.get_date_time()
        file_name = '%s_ep_%d_task_%d_%s' % (args.model, args.n_epochs, model.n_tasks, timestamp)
        plot.plot_1d_loss_all(visual_lss, steps, file_name, show=True)
        plot.save_visual_results(visual_val_acc, visual_train_acc, acc, file_name)
    logger.info('inference time:{}'.format(inference_time))

    return torch.from_numpy(tasks), torch.from_numpy(acc), time_spent


def eval_class_tasks(model, task_loader, args, t):
    model.eval()

    result_acc = []
    result_lss = []

    lss = 0.0
    acc = 0.0

    for (i, (x, y)) in enumerate(task_loader):
        if args.cuda:
            x = x.cuda()
            y = y.cuda()

        outputs = model(x, t)

        offset1, offset2 = model.compute_offsets(t)
        loss = model.loss_ce(outputs, y)

        _, p = torch.max(outputs.data.cpu(), 1, keepdim=False)

        lss += loss.detach() * len(x)
        acc += (p == y.cpu()).float().sum()
    return lss / len(task_loader.dataset), acc / len(task_loader.dataset)


def eval_loader(model, test_loader, t, args):
    model.net.eval()

    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        # Loop batches
        for k, (data, target) in enumerate(test_loader):
            images = data.cuda() if args.cuda else data
            targets = target.cuda() if args.cuda else target

            outputs = model(images, t)
            if model.net.multi_head:
                offset1, offset2 = model.compute_offsets(t)
                if 'five' in args.dataset:
                    loss = model.loss_ce(outputs[:, offset1:offset2], targets)
                    targets += offset1
                else:
                    loss = model.loss_ce(outputs[:, offset1:offset2], targets - offset1)
            else:
                loss = model.loss_ce(outputs, targets)

            _, p = torch.max(outputs.data.cpu(), 1, keepdim=False)
            total_loss += loss.detach() * len(images)
            total_acc += (p == targets.cpu()).float().sum()
    return total_loss / len(test_loader.dataset), total_acc / len(test_loader.dataset)



def life_experience_loader(model, data, ids, args, logger):
    time_start = time.time()

    # store accuravy & loss for all tasks
    acc = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)
    lss = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)
    tasks = np.arange(args.n_tasks, dtype=np.int32)

    # visual landscape
    if args.visual_landscape:
        steps = np.arange(args.step_min, args.step_max, args.step_size)
        visual_lss = np.zeros((args.n_tasks, args.n_tasks, args.dir_num, len(steps)), dtype=np.float32)
        visual_val_acc = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)
        visual_train_acc = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)

    # tensorboard & checkpoint
    args.log_dir, args.checkpoint_dir = utils.log_dir(args)
    writer = SummaryWriter(args.log_dir)

    # train/val/test order by ids
    # t: the real task id
    test_loaders = [torch.utils.data.DataLoader(data[t]['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4) for t in ids]
    for i, t in enumerate(ids):
        # Get data
        task = t

        train_loader = torch.utils.data.DataLoader(data[t]['train'], batch_size=args.batch_size, shuffle=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(data[t]['valid'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)

        print('*' * 100)
        print('>>>Task {:2d}({:s}) | Train: {:5d}, Val: {:5d}, Test: {:5d}<<<'.format(i, data[t]['name'],
                                   len(data[t]['train']), len(data[t]['valid']), len(data[t]['test'])))
        print('*' * 100)

        # if i != 0 and args.model in ['expand_gpm', 'expand_gpm1', 'expand_gpm2', 'expand_gpm_1', 'expand_gpm1_1', 'expand_gpm2_1']:
        #     model.expand_feature_space(args, t)
        # elif i != 0 and args.model in ['expand_gpm3']:
        #     model.expand_feature_space(args, t, xtrain)

        # Train
        clock0 = time.time()
        # bn's parameters are only learned for the first task
        if args.freeze_bn and i == 1:
            for m in model.net.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        # reset the learning rate
        lr = args.lr
        model.update_optimizer(lr)
        if args.model == 'fsdgpm':
            model.eta1 = args.eta1
            if len(model.M_vec) > 0 and args.method in ['dgpm', 'xdgpm']:
                # reset lambda
                model.eta2 = args.eta2
                model.define_lambda_params()
                model.update_opt_lambda(model.eta2)

        # if use early stop, then start training new tasks from the optimal model
        if args.earlystop:
            best_loss = np.inf
            patience = args.lr_patience
            best_model = copy.deepcopy(model.net.state_dict())

        if args.model in ['api', 'api_i', 'api_i_1', 'api_i_2'] and i!=0:
            # train the model for a loop to count the gradient norm
            model.net.train()
            idx = np.arange(len(data[t]['train']))
            np.random.shuffle(idx)
            idx = torch.LongTensor(idx)
            for k, (image, target) in enumerate(train_loader):
                v_x = image.cuda() if args.cuda else image
                v_y = target.cuda() if args.cuda else target

                loss = model.observe(v_x, v_y, t, evalue=True)
            if args.model in ['api', 'api_i', 'api_i_1', 'api_i_2']:
                model.expand_feature_space(args, train_loader,writer=writer)
            else:
                model.expand_feature_space(args, t)
            # print(eval(model, data[ids[0]]['test']['x'], data[ids[u]]['test']['y'], ids[0], args))
        # elif args.model in ['alci'] and i != 0:
        #     model.expand_feature_space(args, train_loader)



        prog_bar = tqdm(range(args.n_epochs))
        for ep in prog_bar:
            # train
            model.epoch += 1
            model.real_epoch = ep

            model.net.train()
            train_loss = 0.0

            # Loop batches
            # clock0=time.time()
            for k, (image, target) in enumerate(train_loader):
                v_x = image.cuda() if args.cuda else image
                v_y = target.cuda() if args.cuda else target
                if args.model != 'hat':
                    loss = model.observe(v_x, v_y, t, writer, stage=1)
                else:
                    loss = model.observe(v_x, v_y, t, float(k)/len(train_loader.dataset), writer)
                train_loss += loss * len(v_x)

            train_loss = train_loss / len(data[t]['train'])
            writer.add_scalar(f"1.Train-LOSS/{data[t]['name']}", round(train_loss.item(), 5), model.epoch)
            # ipdb.set_trace()
            
            # if use early stop, we need to adapt lr and store the best model
            if args.earlystop:
                # Valid
                valid_loss, valid_acc = eval_loader(model, valid_loader, t, args)
                writer.add_scalar(f"2.Val-LOSS/{data[t]['name']}", round(valid_loss.item(), 5), model.epoch)
                writer.add_scalar(f"2.Val-ACC/{data[t]['name']}", 100 * valid_acc, model.epoch)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = copy.deepcopy(model.net.state_dict())
                    patience = args.lr_patience
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= args.lr_factor
                        print(' lr={:.1e} |'.format(lr), end='')
                        if lr < args.lr_min:
                            break
                        patience = args.lr_patience
                        model.update_optimizer(lr)
                        if args.model == 'fsdgpm':
                            model.eta1 = model.eta1 / args.lr_factor
                            if len(model.M_vec) > 0 and args.method in ['dgpm', 'xdgpm']:
                                model.eta2 = model.eta2 / args.lr_factor
                                model.update_opt_lambda(model.eta2)

                    prog_bar.set_description(
                        "Task: {} | Epoch: {}/{} | time={:2.2f}s | Train: loss={:.3f} | Valid: loss={:.3f}, acc={:5.1f}% |".format(
                            i, ep + 1, model.n_epochs, time.time() - clock0, round(train_loss.item(), 5),
                            round(valid_loss.item(), 5), 100 * valid_acc)
                    )
            else:
                prog_bar.set_description("Task: {} | Epoch: {}/{} | time={:2.2f}s | Train: loss={:.3f} |".format(
                        i, ep + 1, model.n_epochs, time.time() - clock0, round(train_loss.item(), 5))
                    )
            

        if args.earlystop:
            model.net.load_state_dict(copy.deepcopy(best_model))

        # ipdb.set_trace()

        print('-' * 60)
        print('Total Epoch: {}/{} | Training Time: {:.2f} min | Last Lr: {}'.format(ep + 1, model.n_epochs,
                                                                                    (time.time() - clock0) / 60, lr))
        print('-' * 60)

        # if args.model in ['alci']:
        #     model.memory_scale[task] = []
        #     for k, (name, param) in enumerate(model.net.named_parameters()):
        #         if 'scale' in name:
        #             model.memory_scale[task].append(copy.deepcopy(param.data))

        # Test
        clock1 = time.time()
        inference_time = []
        for u in range(i + 1):

            if args.model in ['trgp']:
                if model.arch == 'lenet':
                    space1, space2 = [None]*4, [None]*4
                elif model.arch == 'alexnet':
                    space1, space2 = [None]*5, [None]*5
                elif model.arch == 'mlp':
                    space1, space2 = [None]*3, [None]*3
                elif model.arch == 'resnet':
                    space1, space2 = [None]*20, [None]*20
                else:
                    raise NotImplementedError
                
                model.space1, model.space2 = space1, space2

                if u > 0:
                    for i in range(len(space1)):
                        for k, task_sel in enumerate(model.memory[u][str(i)]['selected_task']):
                            # print(memory[task_name]['regime'][task_sel])
                            if model.memory[u][str(i)]['regime'][task_sel] == '2':
                                if k == 0:
                   
                                    space1[i] = torch.FloatTensor(model.memory[task_sel][str(i)]['space_list']).to('cuda')
                                    idx = 0
                                    for m,params in model.named_parameters():
                                        if 'scale1' in m:
                                            params.data = model.memory[u][str(idx)]['scale1'].to('cuda')
                                            idx += 1
                                else:
                                    space2[i] = torch.FloatTensor(model.memory[task_sel][str(i)]['space_list']).to('cuda')
                                    idx = 0
                                    for m,params in model.named_parameters():                               
                                        if 'scale2' in m:
                                            params.data = model.memory[u][str(idx)]['scale2'].to('cuda')
                                            idx += 1  
            
            test_start = time.time()
            test_loss, test_acc = eval_loader(model, test_loaders[u], ids[u], args)
            test_end = time.time()
            inference_time.append(test_end-test_start)
            
            # test_loss, test_acc = 0, 0
            # for _ in range(10):
            #     test_loss_, test_acc_ = eval_loader(model, test_loaders[u], ids[u], args)
            #     test_loss += test_loss_
            #     test_acc += test_acc_
            # test_loss /= 10
            # test_acc /= 10

            # ipdb.set_trace()
            acc[i, u] = test_acc
            lss[i, u] = test_loss

            writer.add_scalar(f"0.Test-LOSS/{data[ids[u]]['name']}", test_loss, i)
            writer.add_scalar(f"0.Test-ACC/{data[ids[u]]['name']}", 100 * test_acc, i)
            writer.add_scalar(f"0.Test-BWT/{data[ids[u]]['name']}", 100 * (test_acc - acc[u, u]), i)

        avg_acc = sum(acc[i]) / (i + 1)
        bwt = np.mean((acc[i]-np.diag(acc)))

        writer.add_scalar(f"0.Test/Avg-ACC", 100 * avg_acc, i)
        writer.add_scalar(f"0.Test/Avg-BWT", 100 * bwt, i)

        print('-' * 60)
        print('Accuracies =')
        # ipdb.set_trace()
        for i_a in range(i + 1):
            print('\t',end='')
            for j_a in range(len(acc[i_a])):
                print('{:5.1f}% '.format(acc[i_a, j_a]*100),end='')
            print()
        print('-' * 60)

        memory_space = args.memories
        # Update Memory of Feature Space
        if args.model in ['gpm', 'dul_gpm']:
            # Threshold Update
            model.update_threshold(i)

            # Memory Update
            if model.arch == 'resnet':
                mat_list = model.get_representation_matrix_ResNet18 (train_loader)
            else:
                raise NotImplementedError

            model.update_GPM (mat_list, model.threshold)
            # Projection Matrix Precomputation
            model.feature_mat = []
            for p in range(len(model.feature_list)):
                Uf=torch.Tensor(np.dot(model.feature_list[p],model.feature_list[p].transpose()))
                if model.cuda: Uf = Uf.cuda()
                print('Layer {} - Projection Matrix shape: {}'.format(p+1,Uf.shape))
                model.feature_mat.append(Uf)

            # Get the info of mem
            for p in range(len(model.feature_list)):
                writer.add_scalar(f"3.MEM-Total/Layer_{p}", model.feature_list[p].shape[1], i)

            # Get the info of mem
            memory_space = 0
            for p in range(len(model.feature_list)):
                memory_space += model.feature_list[p].shape[1] * model.feature_list[p].shape[0]
            if 'imagenet' in args.dataset:
                memory_space = memory_space/7056.
            else:
                raise NotImplementedError
            writer.add_scalar(f"3.MEM-Total/Memory", memory_space, i)
        elif args.model in ['dualgpm', 'api', 'api_i', 'api_i_1', 'api_i_2']:
            # Threshold Update
            model.update_threshold(i)

            # Memory Update  
            if model.arch == 'resnet':
                mat_list = model.get_representation_matrix_ResNet18 (train_loader)
            else:
                raise NotImplementedError

            model.update_GPM (mat_list, model.threshold)

            # Get the info of mem
            for p in range(len(model.feature_list)):
                writer.add_scalar(f"3.MEM-Total/Layer_{p}", model.feature_list[p].shape[1], i)

            # Get the info of mem
            base_memory = 0
            for p in range(len(model.feature_list)):
                base_memory += model.feature_list[p].shape[1] * model.feature_list[p].shape[0]
            
            if 'imagenet' in args.dataset:
                base_memory = base_memory/7056.
            else:
                raise NotImplementedError
            writer.add_scalar(f"3.MEM-Total/base_Memory", base_memory, i)

            param_memory = 0
            param_memory += (model.net.conv1.weight.size(1)-3)*20*9
            param_memory += (model.net.layer1[0].conv1.weight.size(1)-20)*20*9
            param_memory += (model.net.layer1[0].conv2.weight.size(1)-20)*20*9
            param_memory += (model.net.layer1[1].conv1.weight.size(1)-20)*20*9
            param_memory += (model.net.layer1[1].conv2.weight.size(1)-20)*20*9 

            param_memory += (model.net.layer2[0].conv1.weight.size(1)-20)*40*9
            param_memory += (model.net.layer2[0].conv2.weight.size(1)-40)*40*9
            param_memory += (model.net.layer2[0].conv3.weight.size(1)-20)*40*1
            param_memory += (model.net.layer2[1].conv1.weight.size(1)-40)*40*9
            param_memory += (model.net.layer2[1].conv2.weight.size(1)-40)*40*9

            param_memory += (model.net.layer3[0].conv1.weight.size(1)-40)*80*9
            param_memory += (model.net.layer3[0].conv2.weight.size(1)-80)*80*9
            param_memory += (model.net.layer3[0].conv3.weight.size(1)-40)*80*1
            param_memory += (model.net.layer3[1].conv1.weight.size(1)-80)*80*9
            param_memory += (model.net.layer3[1].conv2.weight.size(1)-80)*80*9

            param_memory += (model.net.layer4[0].conv1.weight.size(1)-80)*160*9
            param_memory += (model.net.layer4[0].conv2.weight.size(1)-160)*160*9
            param_memory += (model.net.layer4[0].conv3.weight.size(1)-80)*160*1
            param_memory += (model.net.layer4[1].conv1.weight.size(1)-160)*160*9
            param_memory += (model.net.layer4[1].conv2.weight.size(1)-160)*160*9

            param_memory += (model.net.conv1.weight.size(1)-3)*3
            param_memory += (model.net.layer1[0].conv1.weight.size(1)-20)*20
            param_memory += (model.net.layer1[0].conv2.weight.size(1)-20)*20
            param_memory += (model.net.layer1[1].conv1.weight.size(1)-20)*20
            param_memory += (model.net.layer1[1].conv2.weight.size(1)-20)*20 

            param_memory += (model.net.layer2[0].conv1.weight.size(1)-20)*20
            param_memory += (model.net.layer2[0].conv2.weight.size(1)-40)*40
            param_memory += (model.net.layer2[0].conv3.weight.size(1)-20)*20
            param_memory += (model.net.layer2[1].conv1.weight.size(1)-40)*40
            param_memory += (model.net.layer2[1].conv2.weight.size(1)-40)*40

            param_memory += (model.net.layer3[0].conv1.weight.size(1)-40)*40
            param_memory += (model.net.layer3[0].conv2.weight.size(1)-80)*80
            param_memory += (model.net.layer3[0].conv3.weight.size(1)-40)*40
            param_memory += (model.net.layer3[1].conv1.weight.size(1)-80)*80
            param_memory += (model.net.layer3[1].conv2.weight.size(1)-80)*80

            param_memory += (model.net.layer4[0].conv1.weight.size(1)-80)*80
            param_memory += (model.net.layer4[0].conv2.weight.size(1)-160)*160
            param_memory += (model.net.layer4[0].conv3.weight.size(1)-80)*80
            param_memory += (model.net.layer4[1].conv1.weight.size(1)-160)*160
            param_memory += (model.net.layer4[1].conv2.weight.size(1)-160)*160

            if 'imagenet' in args.dataset:
                param_memory = param_memory/7056.
            else:
                raise NotImplementedError
            writer.add_scalar(f"3.MEM-Total/param_Memory", param_memory, i)

            writer.add_scalar(f"3.MEM-Total/Total_Memory", param_memory+base_memory, i)

            if args.model in ['dualgpm', 'api', 'api_i', 'api_i_1', 'api_i_2']:
                model.feature_mat = []
                # Projection Matrix Precomputation
                for p in range(len(model.feature_list)):
                    if model.feature_list[p].shape[1]>0:
                        Uf=torch.Tensor(np.dot(model.feature_list[p],model.feature_list[p].transpose()))
                    else:
                        assert model.project_type[p] == 'retain'
                        Uf=torch.zeros(model.feature_list[p].shape[0], model.feature_list[p].shape[0])
                    print('Layer {} - Projection Matrix shape: {}'.format(p+1,Uf.shape))
                    if model.cuda: Uf = Uf.cuda()
                    model.feature_mat.append(Uf)

    time_end = time.time()
    time_spent = time_end - time_start

    print('*' * 100)
    print('>>> Final Test Result: ACC={:5.3f}%, BWT={:5.3f}%, Total time = {:.2f} min<<<'.format(
        100 * avg_acc, 100 * bwt, time_spent / 60))
    print('*' * 100)

    memory_space, param_space = args.memories, 0
    if args.model in ['gpm', 'dul_gpm']:
        memory_space = 0
        # ipdb.set_trace()
        for p in range(len(model.feature_list)):
            memory_space += model.feature_list[p].shape[1] * model.feature_list[p].shape[0]

        if 'imagenet' in args.dataset:
            memory_space = memory_space/7056.
        else:
            raise NotImplementedError  
        logger.info('run:memory_space:{}, param_space:{}'.format(memory_space, memory_space-(param_space/7056.)))
    elif args.model in ['gpm', 'dualgpm', 'api', 'api_i', 'api_i_1', 'api_i_2']:
        param_space = 0
        # ipdb.set_trace()
        for p in range(len(model.feature_list)):
            param_space += model.feature_list[p].shape[1] * model.feature_list[p].shape[0]

        memory_space = 0

        memory_space += (model.net.conv1.weight.size(1)-3)*20*9
        memory_space += (model.net.layer1[0].conv1.weight.size(1)-20)*20*9
        memory_space += (model.net.layer1[0].conv2.weight.size(1)-20)*20*9
        memory_space += (model.net.layer1[1].conv1.weight.size(1)-20)*20*9
        memory_space += (model.net.layer1[1].conv2.weight.size(1)-20)*20*9 

        memory_space += (model.net.layer2[0].conv1.weight.size(1)-20)*40*9
        memory_space += (model.net.layer2[0].conv2.weight.size(1)-40)*40*9
        memory_space += (model.net.layer2[0].conv3.weight.size(1)-20)*40*1
        memory_space += (model.net.layer2[1].conv1.weight.size(1)-40)*40*9
        memory_space += (model.net.layer2[1].conv2.weight.size(1)-40)*40*9

        memory_space += (model.net.layer3[0].conv1.weight.size(1)-40)*80*9
        memory_space += (model.net.layer3[0].conv2.weight.size(1)-80)*80*9
        memory_space += (model.net.layer3[0].conv3.weight.size(1)-40)*80*1
        memory_space += (model.net.layer3[1].conv1.weight.size(1)-80)*80*9
        memory_space += (model.net.layer3[1].conv2.weight.size(1)-80)*80*9

        memory_space += (model.net.layer4[0].conv1.weight.size(1)-80)*160*9
        memory_space += (model.net.layer4[0].conv2.weight.size(1)-160)*160*9
        memory_space += (model.net.layer4[0].conv3.weight.size(1)-80)*160*1
        memory_space += (model.net.layer4[1].conv1.weight.size(1)-160)*160*9
        memory_space += (model.net.layer4[1].conv2.weight.size(1)-160)*160*9

        memory_space += (model.net.conv1.weight.size(1)-3)*3
        memory_space += (model.net.layer1[0].conv1.weight.size(1)-20)*20
        memory_space += (model.net.layer1[0].conv2.weight.size(1)-20)*20
        memory_space += (model.net.layer1[1].conv1.weight.size(1)-20)*20
        memory_space += (model.net.layer1[1].conv2.weight.size(1)-20)*20 

        memory_space += (model.net.layer2[0].conv1.weight.size(1)-20)*20
        memory_space += (model.net.layer2[0].conv2.weight.size(1)-40)*40
        memory_space += (model.net.layer2[0].conv3.weight.size(1)-20)*20
        memory_space += (model.net.layer2[1].conv1.weight.size(1)-40)*40
        memory_space += (model.net.layer2[1].conv2.weight.size(1)-40)*40

        memory_space += (model.net.layer3[0].conv1.weight.size(1)-40)*40
        memory_space += (model.net.layer3[0].conv2.weight.size(1)-80)*80
        memory_space += (model.net.layer3[0].conv3.weight.size(1)-40)*40
        memory_space += (model.net.layer3[1].conv1.weight.size(1)-80)*80
        memory_space += (model.net.layer3[1].conv2.weight.size(1)-80)*80

        memory_space += (model.net.layer4[0].conv1.weight.size(1)-80)*80
        memory_space += (model.net.layer4[0].conv2.weight.size(1)-160)*160
        memory_space += (model.net.layer4[0].conv3.weight.size(1)-80)*80
        memory_space += (model.net.layer4[1].conv1.weight.size(1)-160)*160
        memory_space += (model.net.layer4[1].conv2.weight.size(1)-160)*160

        memory_space += param_space
        if 'imagenet' in args.dataset:
            memory_space = memory_space/7056.
        else:
            raise NotImplementedError
        
        logger.info('run:memory_space:{}, param_space:{}'.format(memory_space, memory_space-(param_space/7056.)))
    elif args.model in ['ER']:
        logger.info('run:memory_space:{}'.format(memory_space))
    # plot & save
    if args.visual_landscape:
        timestamp = utils.get_date_time()
        file_name = '%s_ep_%d_task_%d_%s' % (args.model, args.n_epochs, model.n_tasks, timestamp)
        plot.plot_1d_loss_all(visual_lss, steps, file_name, show=True)
        plot.save_visual_results(visual_val_acc, visual_train_acc, acc, file_name)
    logger.info('inference time:{}'.format(inference_time))
    return torch.from_numpy(tasks), torch.from_numpy(acc), time_spent





def main(args, logger):
    utils.print_arguments(args)
    print("Starting at :", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # initialize seeds
    utils.init_seed(args)

    # Setup DataLoader
    print('Load data...')
    print("Dataset: ", args.dataset, args.data_path)

    if args.dataset in ['tinyimagenet', 'mnist_permutations']:
        Loader = importlib.import_module('dataloaders.' + args.loader)
        loader = Loader.IncrementalLoader(args, seed=args.seed)
        n_inputs, n_outputs, n_tasks, input_size = loader.get_dataset_info()

        # input_size: ch * size * size = n_inputs
        print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
        print('-' * 100)
    else:
        dataloader = importlib.import_module('dataloaders.' + args.dataset)
        if args.dataset == 'cifar100_superclass':
            task_order = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                          np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
                          np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
                          np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
                          np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])]

            ids = task_order[args.t_order]
            data, output_info, input_size, n_tasks, n_outputs = dataloader.get(data_path=args.data_path, task_order=ids,
                                                                               seed=args.seed, pc_valid=args.pc_valid)
            args.n_tasks = n_tasks
            args.samples_per_task = int(data[0]['train']['y'].shape[0] / (1.0 - args.pc_valid))
        else:
            data, output_info, input_size, n_tasks, n_outputs = dataloader.get(data_path=args.data_path, args=args,
                                                                               seed=args.seed, pc_valid=args.pc_valid,
                                                                               samples_per_task=args.samples_per_task)
            if args.dataset != 'mini_imagenet':
                args.samples_per_task = int(data[0]['train']['y'].shape[0] / (1.0 - args.pc_valid))
            else:
                args.samples_per_task = int(len(data[0]['train']) / (1.0 - args.pc_valid))
            args.n_tasks = n_tasks
            # Shuffle tasks
            if args.shuffle_task:
                ids = list(shuffle(np.arange(args.n_tasks), random_state=args.seed))
            else:
                ids = list(np.arange(args.n_tasks))

        print('Task info =', output_info)
        print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
        print('Task order =', ids)
        print('-' * 100)

    # Setup Model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(input_size, n_outputs, n_tasks, args)
    print("Model:", model.net)
    if args.cuda:
        model.net.cuda()

    # Train & Test
    try:
        if args.dataset in ['mini_imagenet']:
            result_test_t, result_test_a, spent_time = life_experience_loader(model, data, ids, args, logger)
        else:
            result_test_t, result_test_a, spent_time = life_experience(model, data, ids, args, logger)

        # save results in checkpoint_dir
        utils.save_results(args, result_test_t, result_test_a, model, spent_time)

    except KeyboardInterrupt:
        print()

    return sum(result_test_a[-1])/len(result_test_a[-1])

if __name__ == "__main__":
    parser = file_parser.get_parser()
    args = parser.parse_args()

    if args.model_arch == 'mlp':
        assert args.dataset == 'pmnist'
    elif args.model_arch == 'lenet' or args.model_arch == 'alexnet':
        assert 'cifar' in args.dataset
    elif args.model_arch == 'pc_cnn':
        assert args.dataset == 'tinyimagenet'
    elif args.model_arch == 'resnet':
        assert args.dataset == 'five_datasets' or args.dataset == 'mini_imagenet' or 'cifar' in args.dataset

    if args.model in ['api', 'api_i', 'api_i_1', 'api_i_2']:
        path = os.path.join('Results', 'enabled_{}/dataset_{}/arch_{}/method_{}/step_{}_alpha_{}/size_{}/'.format(args.cuda_enabled, args.dataset, args.model_arch, args.model, args.step, args.alpha, args.model_size))
    else:
        raise NotImplementedError
    args.path = path

    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger('experiment_log')
    logger.setLevel(logging.INFO)
    rf_handler = logging.StreamHandler()
    rf_handler.setLevel(logging.INFO)
    if args.model in ['api', 'api_i', 'api_i_1', 'api_i_2']: 
        save_path = os.path.join(path, 'lr_{}-n_runs_{}.log'.format(args.lr, args.n_runs)) 
    else:
        raise Exception('Wrong method!')
    f_handler = logging.FileHandler(filename=save_path)
    f_handler.setLevel(logging.INFO)
    logger.addHandler(rf_handler)
    logger.addHandler(f_handler)

    all_avg_accs = []
    training_time = []
    for seed in range(args.n_runs):
        args.seed = seed
        start = time.time()
        avg_acc = main(args, logger)
        end = time.time()
        training_time.append(end-start)
        all_avg_accs.append(avg_acc.item())
    logger.info(all_avg_accs)
    logger.info("average acc:{}".format(sum(all_avg_accs)/len(all_avg_accs)))
    logger.info(training_time)
    logger.info("average time:{}".format(sum(training_time)/len(training_time)))
