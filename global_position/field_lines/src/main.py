import argparse
import os
import os.path as osp
from glob import glob
import importlib
import random
import numpy as np
import torch
import torch.nn as nn

import _init_paths
from opt import Params

def main():
    parser = argparse.ArgumentParser(description='Court Line')
    parser.add_argument('--default_cfg_path', default = None, help='default config file')
    parser.add_argument('--cfg_paths', type=str, nargs="+", default = None, help='List of updated config file')
    args = parser.parse_args()
    # setup opt
    if args.default_cfg_path is None:
        raise ValueError('default config path not found, should define one')
    opt = Params(args.default_cfg_path)
    if args.cfg_paths is not None:
        for cfg_path in args.cfg_paths:
            opt.update(cfg_path)
    # setup random or deterministic training
    if opt.seed is None:
        torch.backends.cudnn.benchmark = True
    else:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.vis_gpu
    Trainer = importlib.import_module('trainers.train_' + opt.trainer_name)
    Trainer.createAndRunTrainer(opt.gpu_id, opt)

if __name__ == '__main__':
    main()
