import os
import os.path as osp
import shutil
from glob import glob
import pickle
import numpy as np
from tqdm import tqdm
import importlib
import cv2
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from utils.training_utils import *

class TrainPix2pix(object):
    def __init__(self, opt):
        super(TrainPix2pix, self).__init__()
        self.opt = opt
        # set device
        self.loc = 'cuda:{}'.format(self.opt.gpu_id)
        self.device = torch.device(self.loc)
        # setup model
        self.setup_model()
        if self.opt.exp_type == 'demo':
            return

    def setup_model(self):
        print('===> Building models')
        model_lib = importlib.import_module('models.{}'.format(self.opt.model.lib))
        # build model
        self.model = getattr(model_lib, self.opt.model.name)(input_nc=3, output_nc=2)
        self.model.apply(model_lib.weights_init)
        self.model.to(self.device)
        # load pretrained checkpoint
        if self.opt.checkpoint_path is not None and osp.isfile(self.opt.checkpoint_path):
            checkpoint = torch.load(self.opt.checkpoint_path, map_location=self.loc)
            restore(self.model, checkpoint['model'])
            print("=> Successfully loaded checkpoint at '{}'".format(self.opt.checkpoint_path))


    def demo(self, img_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        img_paths = sorted(glob(osp.join(img_dir,'*.png')), key=osp.getmtime)
        to_tensor = transforms.ToTensor()
        resize = iaa.Resize({"height": self.opt.dataset.rgb_res[0], "width": self.opt.dataset.rgb_res[1]})
        with torch.no_grad():
            self.model.eval()
            for idx, img_path in enumerate(img_paths):
                print(img_path)
                # prepare input
                rgb_img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                resize_to_orig = iaa.Resize({"height": rgb_img.shape[0], "width": rgb_img.shape[1]})
                rgb_img = resize(image=rgb_img)
                rgb_img_ts = to_tensor(rgb_img)
                rgb_img_ts = rgb_img_ts.unsqueeze(0).to(self.device)
                # forward
                pred_mask = self.model(rgb_img_ts)
                # transform ouput
                pred_mask = pred_mask[0].cpu() # (2,256,256), float
                pred_mask = torch.argmax(pred_mask,0).numpy().astype('uint8')
                pred_segmap = SegmentationMapsOnImage(pred_mask, shape=rgb_img.shape)
                rgb_img, pred_segmap = resize_to_orig(image=rgb_img, segmentation_maps=pred_segmap)
                # write results
                pred_mask = pred_segmap.get_arr().astype('uint8')
                pred_mask = (pred_mask*255).astype('uint8')
                cv2.imwrite(osp.join(out_dir, img_path.split('/')[-1]), pred_mask)
                cells = []
                cells.append(rgb_img)
                cells.append(pred_segmap.draw_on_image(rgb_img)[0])
                cells.append(pred_segmap.draw(size=rgb_img.shape[:2])[0])
                grid_image = ia.draw_grid(cells, rows=1, cols=3)
                grid_image_path = osp.join(out_dir, img_path.split('/')[-1].replace('.png','_grid.png'))
                cv2.imwrite(grid_image_path, grid_image[:,:,::-1])



def createAndRunTrainer(gpu_id, opt):
    opt.gpu_id = gpu_id
    trainer = TrainPix2pix(opt)
    # Demo
    if opt.exp_type == 'demo':
        print("=> Running Demo")
        trainer.demo(opt.demo.img_dir, opt.demo.out_dir)
    else:
        raise ValueError('Experiment Type not supported.')
