import datetime
import json
import os
import random
import numpy as np
import torch
import argparse
from utils.metrics import confusion_matrix


def init_seed(args):
    print("Set seed", args.seed)
    random.seed(args.seed)
    torch.cuda.cudnn_enabled = (args.cuda_enabled == 1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.enabled = (args.cuda_enabled == 1)


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")


def get_date_time():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_arguments(args):
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

def log_dir(args, timestamp=None):
    if timestamp is None:
        timestamp = get_date_time()

    stat = '/%s-%s' % (args.expt_name, timestamp)

    log_dir = args.path + '/%s' % (args.dataset) + stat
    checkpoint = args.checkpoint_dir + '%s' % (args.dataset) + stat

    mkdir(log_dir)

    return log_dir, checkpoint


def save_results(args, result_test_t, result_test_a, model, spent_time, stronger_result_a=None):

    #mkdir(args.checkpoint_dir)
    with open(args.log_dir + '/training_parameters.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    fname = os.path.join(args.log_dir, 'results')

    # save confusion matrix and print one line of stats

    test_stats = confusion_matrix(result_test_t, result_test_a, args.log_dir, 'accuracy.txt',stronger_result_a)

    one_liner = ' # test: ' + ' '.join(["%.3f" % stat for stat in test_stats])

    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_test_t, result_test_a, model.state_dict(),
                test_stats, one_liner, args), fname + '.pt')
    return test_stats


# https://stackoverflow.com/a/43357954/6365092
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
