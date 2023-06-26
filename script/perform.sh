#!/bin/bash

n_runs=1

CIFAR20='--dataset cifar100_20 --n_tasks 20 --pc_valid 0.05 --batch_size 64 --test_batch_size 64 --n_epochs 200 --cuda --earlystop'
CSUPER='--dataset cifar100_superclass --pc_valid 0.05 --batch_size 64 --test_batch_size 64 --n_epochs 50 --cuda --earlystop'
FIVED='--dataset five_datasets --samples_per_task -1 --n_tasks 5 --pc_valid 0.05 --batch_size 64 --test_batch_size 64 --n_epochs 100 --cuda --earlystop --lr_min 1e-3 --lr_patience 5 --lr_factor 3'

# 20-Split CIFAR100
# CUDA_VISIBLE_DEVICES=$2 python main_api.py $CIFAR20 --n_runs $n_runs --data_path $1 --model api --lr 0.01 --model_arch alexnet --model_size 1.0 --expt_name api --cuda_enabled 1 --alpha 10 --step 0.5 
CUDA_VISIBLE_DEVICES=$2 python main_gpm.py $CIFAR20 --n_runs $n_runs --data_path $1 --model dualgpm --lr 0.01 --model_arch alexnet --model_size 1.0 --expt_name dulgpm --cuda_enabled 1 

# # CIFAR100-sup
# CUDA_VISIBLE_DEVICES=$2 python main_api_cifarsup.py $CSUPER --n_runs $n_runs --data_path $1 --model api --lr 0.01 --model_arch lenet --model_size 1.0 --expt_name api --cuda_enabled 1 --alpha 10 --step 0.5 
# CUDA_VISIBLE_DEVICES=$2 python main_gpm.py $CSUPER --n_runs $n_runs --data_path $1 --model dualgpm --lr 0.01 --model_arch lenet --model_size 1.0 --expt_name dulgpm --cuda_enabled 1 

# five datasets
# CUDA_VISIBLE_DEVICES=$2 python main_api_five.py $FIVED --n_runs $n_runs --data_path $1 --model api --lr 0.1 --model_arch resnet --model_size 1.0 --expt_name api --cuda_enabled 1 --alpha 10 --step 0.5
# # CUDA_VISIBLE_DEVICES=$2 python main_gpm.py $FIVED --n_runs $n_runs --data_path $1 --model dualgpm --lr 0.1 --model_arch resnet --model_size 1.0 --expt_name dualgpm --cuda_enabled 1 


