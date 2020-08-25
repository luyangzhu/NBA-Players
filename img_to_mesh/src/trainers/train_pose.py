import os
import os.path as osp
from glob import glob
import shutil
import numpy as np
import random
import importlib
import pickle
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.training_utils import *
from utils.vis_utils import *
from utils.pose_inference import *
from utils.pose_data_utils import flip_back
import constants

class TrainPose(object):
    def __init__(self, opt):
        super(TrainPose, self).__init__()
        self.opt = opt
        # set device
        self.loc = 'cuda:{}'.format(self.opt.gpu_id)
        self.device = torch.device(self.loc)
        # setup joints data
        self.setup_joint_data()
        # setup model
        self.setup_model()
        if self.opt.exp_type == 'demo':
            return

    def setup_joint_data(self):
        # joints meta data
        self.opt.dataset.nParts = constants.JOINTS_NPARTS[self.opt.dataset.joints_type]
        self.parent_ids = constants.JOINTS_PARENT_IDS[self.opt.dataset.joints_type]
        self.flip_pairs = constants.JOINTS_FLIP_PAIRS[self.opt.dataset.joints_type]
        self.joint_index = constants.JOINTS_INDEX[self.opt.dataset.joints_type]
        self.right_joint_list = constants.RIGHT_JOINTS[self.opt.dataset.joints_type]
        self.lsp_idx = constants.LSP_INDEX

    def setup_model(self):
        print('===> Building models')
        model_lib = importlib.import_module('models.{}'.format(self.opt.model.lib))
        # update head params
        if 'hm' in self.opt.model.heads.dict.keys():
            self.opt.model.heads.hm = self.opt.dataset.nParts
        if 'j3d' in self.opt.model.heads.dict.keys():
            self.opt.model.heads.j3d = self.opt.dataset.nParts*3
        # build model
        self.model = model_lib.get_model(num_layers=self.opt.model.num_layers, 
                                        heads=self.opt.model.heads.dict, 
                                        num_classes=self.opt.model.nClasses
                                )
        self.model.to(self.device)
        # load pretrained checkpoint
        if self.opt.checkpoint_path is not None and osp.isfile(self.opt.checkpoint_path):
            checkpoint = torch.load(self.opt.checkpoint_path, map_location=self.loc)
            restore(self.model, checkpoint['model'])
            print("=> Successfully loaded checkpoint at '{}'".format(self.opt.checkpoint_path))



    def demo(self, img_dir, out_dir):
        # create folder
        os.makedirs(osp.join(out_dir, '2d'), exist_ok=True)
        os.makedirs(osp.join(out_dir, '3d_overlay'), exist_ok=True)
        os.makedirs(osp.join(out_dir, 'npy'), exist_ok=True)
        
        img_paths = sorted(glob(osp.join(img_dir,'*.png')),key=osp.getmtime)
        print(len(img_paths))
        normalize = transforms.Normalize(mean=constants.IMG_MEAN,
                                    std=constants.IMG_NORM)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        with torch.no_grad():
            self.model.eval()
            pred_j2d_list,pred_j3d_list,pred_jump_cls_list,pred_jump_dist_list = [],[],[],[]
            img_path_list = []
            for idx, img_path in enumerate(img_paths):
                print(img_path)
                img_path_list.append(img_path)
                # load img
                rgb_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # ndarray, (h,w,3), uint8, (0,255)
                rgb_img_tensor = img_transform(rgb_img).unsqueeze(0).to(self.device)
                # forward
                output = self.model(rgb_img_tensor)
                pred_heatmaps = output[-1]['hm']
                pred_j3dmaps = output[-1]['j3d']
                pred_jump_score = output[-1]['jump_cls']
                pred_jump_dist = output[-1]['jump_dist']
                # save jump info
                pred_jump_cls = torch.argmax(pred_jump_score,1)
                pred_jump_cls_list.append(pred_jump_cls.item())
                pred_jump_dist_list.append(pred_jump_dist.reshape(-1).item())
                # flip test
                input_flipped = np.flip(rgb_img_tensor.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).to(self.device)
                output_flipped = self.model(input_flipped)
                pred_heatmaps_flipped = output_flipped[-1]['hm']
                pred_heatmaps_flipped = flip_back(pred_heatmaps_flipped.cpu().numpy(),
                                           self.flip_pairs)
                pred_heatmaps_flipped = torch.from_numpy(pred_heatmaps_flipped.copy()).to(self.device)
                pred_heatmaps_flipped[:, :, :, 1:] = pred_heatmaps_flipped.clone()[:, :, :, 0:-1]
                pred_heatmaps = (pred_heatmaps + pred_heatmaps_flipped) * 0.5
                # get pred_j3d
                pred_j2d_np = get_hm_j2d(pred_heatmaps.detach().cpu().numpy())
                pred_j2d = torch.from_numpy(pred_j2d_np).to(self.device)
                pred_j3d = get_j3d_pred(j3dmaps=pred_j3dmaps, j2d=pred_j2d, 
                            nParts=self.opt.dataset.nParts, out_res=self.opt.dataset.hm_res)
                # save j2d results
                pred_j2d_np, _ = get_final_preds(pred_heatmaps.cpu().numpy(), 
                                        np.array([[self.opt.dataset.crop_res[1]/2, self.opt.dataset.crop_res[0]/2]]), 
                                        np.array([1.]),
                                        self.opt.dataset.crop_res)
                pred_j2d_list.append(pred_j2d_np)
                # save j3d results
                pred_j3d_np = pred_j3d.cpu().numpy()
                pred_j3d_list.append(pred_j3d_np)
        
        if len(img_paths) > 0:        
            all_jump_cls_np = np.array(pred_jump_cls_list)
            all_jump_dist_np = np.array(pred_jump_dist_list)
            all_j2d_preds_np = np.concatenate(pred_j2d_list,0)
            all_j3d_preds_np = np.concatenate(pred_j3d_list,0)
            assert all_jump_cls_np.shape[0] == all_jump_dist_np.shape[0] == \
                    all_j2d_preds_np.shape[0] == all_j3d_preds_np.shape[0] == len(img_path_list)
            np.save(osp.join(out_dir,'npy','all_jump_dist.npy'), all_jump_dist_np)
            np.save(osp.join(out_dir,'npy','all_jump_cls.npy'), all_jump_cls_np)
            np.save(osp.join(out_dir,'npy','j2d.npy'), all_j2d_preds_np)
            np.save(osp.join(out_dir,'npy','j3d.npy'), all_j3d_preds_np)
            pickle.dump(img_path_list, open(osp.join(out_dir,'npy','img_paths.pkl'),'wb'))
        
        # visualization
        for idx, img_path in enumerate(img_paths):
            print(idx)
            rgb_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            vis_rgb = cv2.imread(img_path)
            vis_j2d = vis_skeleton(vis_rgb.copy(), all_j2d_preds_np[idx], self.parent_ids, self.right_joint_list)
            vis_concat = np.concatenate((vis_rgb, vis_j2d),1)
            cv2.imwrite(osp.join(out_dir,'2d','{:04d}.png'.format(idx)), vis_concat)
            plot_3d_skeleton_demo(all_j3d_preds_np[idx], rgb_img, self.parent_ids, 
                                osp.join(out_dir,'3d_overlay','{:04d}.png'.format(idx)),
                                right_joint_list=self.right_joint_list)

def createAndRunTrainer(gpu_id, opt):
    opt.gpu_id = gpu_id
    trainer = TrainPose(opt)
    # Demo
    if opt.exp_type == 'demo':
        print("=> Running Demo")
        trainer.demo(opt.demo.img_dir, opt.demo.out_dir)
    else:
        raise ValueError('Experiment Type not supported.')
