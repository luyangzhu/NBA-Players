import os
import os.path as osp
from glob import glob
import shutil
import numpy as np
import json
import random
import pickle
import importlib
from tqdm import tqdm
from pdb import set_trace as set_t
import time
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

try:
    import psbody.mesh
    found = True
except ImportError:
    found = False
if found:
    from psbody.mesh import Mesh
import trimesh

from utils.training_utils import *
from utils.mesh_utils import *
import constants

class TrainMesh(object):
    def __init__(self, opt):
        super(TrainMesh, self).__init__()
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
        # setup data
        self.setup_data_loader()
        # setup misc
        self.setup_misc()


    def setup_joint_data(self):
        # joints meta data
        self.opt.dataset.nParts = constants.JOINTS_NPARTS[self.opt.dataset.joints_type]
        self.parent_ids = constants.JOINTS_PARENT_IDS[self.opt.dataset.joints_type]
        self.flip_pairs = constants.JOINTS_FLIP_PAIRS[self.opt.dataset.joints_type]
        self.joint_index = constants.JOINTS_INDEX[self.opt.dataset.joints_type]
        self.right_joint_list = constants.RIGHT_JOINTS[self.opt.dataset.joints_type]

    def setup_model(self):
        print('===> Building models')
        identity_lib = importlib.import_module('models.{}'.format(self.opt.model.iden.lib))
        self.identity_model = getattr(identity_lib, self.opt.model.iden.name)(num_points = constants.NVERTS['human'])
        self.identity_model = self.identity_model.to(self.device)
        # load pretrained checkpoint for identityNet
        if self.opt.checkpoint_dir is not None:
            iden_checkpoint_path = osp.join(self.opt.checkpoint_dir, '{}_identity_network.pth'.format(self.opt.iden_ckpt_prefix))
            if iden_checkpoint_path is not None and osp.isfile(iden_checkpoint_path):
                checkpoint = torch.load(iden_checkpoint_path, map_location=self.loc)
                restore(self.identity_model, checkpoint['model'])
                print("=> Successfully loaded checkpoint at '{}'".format(iden_checkpoint_path))

        skinnning_lib = importlib.import_module('models.{}'.format(self.opt.model.skin.lib))
        self.skinning_model = getattr(skinnning_lib, self.opt.model.skin.name)(opt=self.opt, loc=self.loc,
                    sub_model_name=self.opt.model.skin.sub_name)
        self.skinning_model = self.skinning_model.to(self.device)
        self.skinning_model.load_checkpoint(self.opt.checkpoint_dir, self.opt.skin_ckpt_prefix)
        # loss
        self.criterionL1 = nn.L1Loss()
        # optimizer
        if self.opt.exp_type == 'train':
            self.optimizer_dict = {}
            for body_part in self.opt.body_part_list:
                body_part_model = getattr(self.skinning_model, '{}_net'.format(body_part))
                self.optimizer_dict[body_part] = optim.Adam(body_part_model.parameters(), 
                                            lr=self.opt.training.skin_lr, weight_decay=self.opt.training.weight_decay)
            
            self.optimizer_dict['identity_model'] = optim.Adam(self.identity_model.parameters(), 
                                lr=self.opt.training.iden_lr, weight_decay=self.opt.training.weight_decay)

            self.scheduler_dict = {}
            if self.opt.training.scheduler:
                for body_part in self.opt.body_part_list:
                    self.scheduler_dict[body_part] = optim.lr_scheduler.StepLR(self.optimizer_dict[body_part],
                                            self.opt.training.decay_steps, gamma=self.opt.training.decay_rate)
        if self.opt.exp_type == 'test':
            # load pose model
            pose_lib = importlib.import_module('models.{}'.format(self.opt.model.pose.lib))
            # update head params
            if 'hm' in self.opt.model.pose.heads.dict.keys():
                self.opt.model.pose.heads.hm = self.opt.dataset.nParts
            if 'j3d' in self.opt.model.pose.heads.dict.keys():
                self.opt.model.pose.heads.j3d = self.opt.dataset.nParts*3
            # build model
            self.pose_model = pose_lib.get_model(num_layers=self.opt.model.pose.num_layers, 
                                            heads=self.opt.model.pose.heads.dict, 
                                            num_classes=self.opt.model.pose.nClasses
                                    )
            self.pose_model.to(self.device)
            # load checkpoint
            if self.opt.pose_checkpoint_path is not None and osp.isfile(self.opt.pose_checkpoint_path):
                checkpoint = torch.load(self.opt.pose_checkpoint_path, map_location=self.loc)
                restore(self.pose_model, checkpoint['model'])
                print("=> Successfully loaded checkpoint at '{}'".format(self.opt.pose_checkpoint_path))


    def setup_data_loader(self):
        print('===> Setup data')
        dataset_lib = importlib.import_module('datasets.{}'.format(self.opt.dataset.lib))

        if self.opt.exp_type == 'train':
            train_dataset = dataset_lib.get_dataset(self.opt, self.opt.data_root_dir, 'train', 
                            dummy_node=self.opt.dataset.dummy_node, dir_types=self.opt.dataset.dir_types)
            self.train_data_loader = DataLoader(train_dataset, 
                                        batch_size= self.opt.training.batch_size,
                                        shuffle = True,
                                        pin_memory=False,
                                        drop_last=True,
                                        num_workers = self.opt.training.num_workers
                                    )
            valid_dataset = dataset_lib.get_dataset(self.opt, self.opt.data_root_dir, 'valid', 
                            dummy_node=self.opt.dataset.dummy_node, dir_types=self.opt.dataset.dir_types)
            self.valid_data_loader = DataLoader(valid_dataset, 
                                        batch_size= self.opt.training.valid_batch_size,
                                        shuffle = False,
                                        pin_memory=False,
                                        drop_last=True,
                                        num_workers = self.opt.training.num_workers
                                    )
        elif self.opt.exp_type == 'test':
            test_dataset = dataset_lib.get_dataset(self.opt, self.opt.data_root_dir, 'test', 
                            dummy_node=self.opt.dataset.dummy_node, dir_types=self.opt.dataset.dir_types)
            # batch_size=1 for easy testing
            self.test_data_loader = DataLoader(test_dataset, 
                                        batch_size= 1,
                                        shuffle = False,
                                        pin_memory=False,
                                        drop_last=True,
                                        num_workers = self.opt.training.num_workers
                                    )

    def setup_misc(self):
        print('===> Preparing Summaries and visualization')
        # set log name
        if self.opt.log_name is None:
            prefix = '{}-{}-{}-{}-slr{}-ilr{}-bn{}'.format(
                    self.opt.trainer_name,
                    self.opt.dataset.joints_type,
                    self.opt.model.skin.sub_name,
                    self.opt.model.iden.name,
                    self.opt.training.skin_lr,
                    self.opt.training.iden_lr,
                    self.opt.training.batch_size,
                )
            self.opt.log_name = prefix
            if self.opt.custom_postfix != '':
                self.opt.log_name += '-' + self.opt.custom_postfix

        self.log_dir = osp.join(self.opt.base_log_dir, self.opt.log_name)
        self.ckpt_dir = create_dir(osp.join(self.log_dir, 'ckpt'))
        vis_dir = create_dir(osp.join(self.log_dir, '{}_vis'.format(self.opt.exp_type)))
        setattr(self, '{}_vis_dir'.format(self.opt.exp_type), vis_dir)
        log_path = osp.join(self.log_dir, '{}_log.txt'.format(self.opt.exp_type))
        setattr(self, 'log_path', log_path)

        if self.opt.exp_type == 'train':
            self.valid_vis_dir = create_dir(osp.join(self.log_dir, 'valid_vis'))
            all_keys = self.opt.get_all_keys(all_keys={}, dic=self.opt.dict, parent_key='')
            with open(osp.join(self.log_dir, 'config.txt'), 'w') as f:
                for k,v in all_keys.items():
                    f.write('{}: {}\n'.format(k,v))

        # shape net summaries
        self.identity_mpvpe = AverageValueMeter()
        self.identity_min_error = 1e5
        self.identity_is_best = False
        
        # body part skinningNet summaries
        self.min_error_dict,self.is_best_dict = {},{}
        self.train_loss_dict,self.val_loss_dict = {},{}
        self.mpvpe_v_dict,self.mpvpe_j_dict = {},{}
        for body_part in self.opt.body_part_list:
            self.min_error_dict[body_part] = 1e5
            self.is_best_dict[body_part] = False
            self.train_loss_dict[body_part] = AverageValueMeter()
            self.val_loss_dict[body_part] = AverageValueMeter()
            self.mpvpe_v_dict[body_part] = AverageValueMeter()
            self.mpvpe_j_dict[body_part] = AverageValueMeter()

        # create shapedata, for visualize
        self.shapedata = ShapeData(
                        train_file=osp.join(self.opt.processed_verts_dir, 'human/preprocessed/train_verts.npy'), 
                        val_file=osp.join(self.opt.processed_verts_dir, 'human/preprocessed/val_verts.npy'), 
                        test_file=osp.join(self.opt.processed_verts_dir, 'human/preprocessed/test_verts.npy'), 
                        reference_mesh_file=constants.REF_MESH['human'],
                        normalization=self.opt.dataset.normalization,
                        meshpackage=self.opt.dataset.meshpackage, load_flag = True)


    def train_epoch(self):
        for epoch in range(0, self.opt.training.nepochs):
            # train
            self.train(self.train_data_loader, 'train', epoch)
            # valid
            self.validate(self.valid_data_loader, 'valid', epoch)
            # scheduler
            if self.opt.training.scheduler:
                for body_part in self.opt.body_part_list:
                    self.scheduler_dict[body_part].step()
            # save checkpoint and log
            self.save_ckpt_and_log(epoch)
            

    def save_ckpt_and_log(self, epoch):
        # save identity net ckpt
        if self.identity_mpvpe.avg < self.identity_min_error:
            self.identity_min_error = self.identity_mpvpe.avg
            self.identity_is_best = True
        ckpt_dict = {
                    'model': self.identity_model.state_dict(),
                    'optimizer': self.optimizer_dict['identity_model'].state_dict(),
                    'min_iden_error': self.identity_min_error,
                    'epoch': epoch,
                }
        torch.save(ckpt_dict, osp.join(self.ckpt_dir, 'latest_identity_network.pth'))
        if epoch % self.opt.training.nepoch_ckpt == 0:
            torch.save(ckpt_dict, osp.join(self.ckpt_dir, 'epoch{:03d}_identity_network.pth'.format(epoch)))
        if self.identity_is_best:
            torch.save(ckpt_dict, osp.join(self.ckpt_dir, 'best_identity_network.pth'))
            self.identity_is_best = False

        log_table = {
            "epoch" : epoch,
            "identity_mpvpe": self.identity_mpvpe.avg,
            "identity_min_error": self.identity_min_error
        }
        with open(self.log_path, 'a') as file:
            file.write(json.dumps(log_table))
            file.write('\n')

        for body_part in self.opt.body_part_list:
            if self.mpvpe_j_dict[body_part].avg < self.min_error_dict[body_part]:
                self.min_error_dict[body_part] = self.mpvpe_j_dict[body_part].avg
                self.is_best_dict[body_part] = True
            body_part_model = getattr(self.skinning_model, '{}_net'.format(body_part))
            ckpt_dict = {
                        'model': body_part_model.state_dict(),
                        'optimizer': self.optimizer_dict[body_part].state_dict(),
                        'min_error': self.min_error_dict[body_part],
                        'epoch': epoch,
                    }
            torch.save(ckpt_dict, osp.join(self.ckpt_dir, 'latest_{}_network.pth'.format(body_part)))
            if epoch % self.opt.training.nepoch_ckpt == 0:
                torch.save(ckpt_dict, osp.join(self.ckpt_dir, 'epoch{:03d}_{}_network.pth'.format(epoch, body_part)))
            if self.is_best_dict[body_part]:
                torch.save(ckpt_dict, osp.join(self.ckpt_dir, 'best_{}_network.pth'.format(body_part)))
                self.is_best_dict[body_part] = False

            # dump stats in log file
            log_table = {
                "epoch" : epoch,
                "body_part": body_part,
                "train_loss" : self.train_loss_dict[body_part].avg,
                "val_loss" : self.val_loss_dict[body_part].avg,
                "mpvpe_v": self.mpvpe_v_dict[body_part].avg,
                "mpvpe_j": self.mpvpe_j_dict[body_part].avg,
                "min_error": self.min_error_dict[body_part]
            }
            with open(self.log_path, 'a') as file:
                file.write(json.dumps(log_table))
                file.write('\n')


    def run_iteration(self, epoch, iteration, iter_len, batch, exp_type, vis_dir):
        inp_j3d = batch['inp_j3d'].to(self.device)
        inp_img = batch['inp_img'].to(self.device)
        inp_rest_verts = batch['inp_rest_verts'].to(self.device)
        target_rest_verts = batch['tgt_rest_verts'].to(self.device)
        # identity net forward
        pred_rest_verts = self.identity_model(inp_img, inp_rest_verts)
        batch['human_rest_verts'] = pred_rest_verts.detach()
        # get predicted body part rest verts
        for body_part in self.opt.body_part_list:
            body_part_idx = torch.from_numpy(constants.BODY_PART_INDEX[body_part]).long().to(self.device)
            batch[body_part+'_rest_verts'] = batch['human_rest_verts'][:, body_part_idx]
        # add dummy dimension
        if self.opt.dataset.dummy_node:
            for body_part in self.opt.body_part_list:
                cur_verts = batch[body_part+'_rest_verts']
                zeros = torch.zeros((cur_verts.shape[0],1,cur_verts.shape[2])).float().to(self.device)
                batch[body_part+'_rest_verts'] = torch.cat((cur_verts,zeros),1)

        # skinningNet forward
        pred_dict = self.skinning_model.forward(batch, inp_j3d)
        # compute loss
        loss_dict = {}
        loss_dict['identity_loss'] = self.criterionL1(pred_rest_verts, target_rest_verts)
        for body_part in self.opt.body_part_list:
            target_verts = batch[body_part+'_verts'].to(self.device)
            z_v = pred_dict[body_part+'_z_v']
            pred_verts_v = pred_dict[body_part+'_verts_v']
            z_j = pred_dict[body_part+'_z_j']
            pred_verts_j = pred_dict[body_part+'_verts_j']
            l1_loss_v = self.criterionL1(pred_verts_v, target_verts)
            l1_loss_j = self.criterionL1(pred_verts_j, target_verts)
            l1_loss_z = self.criterionL1(z_j, z_v.detach())
            loss_net = self.opt.loss.w_l1_v*l1_loss_v + self.opt.loss.w_l1_z*l1_loss_z
            loss_dict[body_part+'_loss_v'] = l1_loss_v
            loss_dict[body_part+'_loss_j'] = l1_loss_j
            loss_dict[body_part+'_loss_z'] = l1_loss_z
            loss_dict[body_part+'_loss_net'] = loss_net

        if exp_type == 'train':
            # identity net backward
            self.optimizer_dict['identity_model'].zero_grad()
            loss_dict['identity_loss'].backward()
            self.optimizer_dict['identity_model'].step()
            # skinningNet backward
            for body_part in self.opt.body_part_list:
                self.train_loss_dict[body_part].update(loss_dict[body_part+'_loss_net'].item())
                # backward
                self.optimizer_dict[body_part].zero_grad()
                loss_dict[body_part+'_loss_net'].backward()
                self.optimizer_dict[body_part].step()
        else:
            for body_part in self.opt.body_part_list:
                self.val_loss_dict[body_part].update(loss_dict[body_part+'_loss_net'].item())

        if (exp_type == 'train' and iteration % self.opt.training.log_interval == 0) or (exp_type != 'train'):
            log_info = "===> Epoch[{}]({}/{}):".format(epoch, iteration, iter_len)
            print(log_info)
            # logging identity net
            error = torch.mean(torch.sqrt(torch.sum((pred_rest_verts-target_rest_verts)**2,-1)),-1)
            mean_iden_error = torch.mean(error)
            if exp_type != 'train':
                self.identity_mpvpe.update(mean_iden_error.item())
            log_info = "        identity: Iden_Loss: {:.5f}, Iden_MPVPE: {:.5f}".format(
                                loss_dict['identity_loss'].item(),
                                mean_iden_error.item()
                        )
            print(log_info)

            for body_part in self.opt.body_part_list:
                pred_verts_v = pred_dict[body_part+'_verts_v']
                pred_verts_j = pred_dict[body_part+'_verts_j']
                target_verts = batch[body_part+'_verts'].to(self.device)
                # mean per vertex position error
                if self.opt.dataset.dummy_node:
                    error_v = torch.mean(torch.sqrt(torch.sum((pred_verts_v[:,:-1]-target_verts[:,:-1])**2,-1)),-1)
                    error_j = torch.mean(torch.sqrt(torch.sum((pred_verts_j[:,:-1]-target_verts[:,:-1])**2,-1)),-1)
                else:
                    error_v = torch.mean(torch.sqrt(torch.sum((pred_verts_v-target_verts)**2,-1)),-1)
                    error_j = torch.mean(torch.sqrt(torch.sum((pred_verts_j-target_verts)**2,-1)),-1)
                mean_error_v = torch.mean(error_v)
                mean_error_j = torch.mean(error_j)
                if exp_type != 'train':
                    self.mpvpe_v_dict[body_part].update(mean_error_v.item())
                    self.mpvpe_j_dict[body_part].update(mean_error_j.item())
                log_info = "        {}: Loss: {:.5f}, L1_V: {:.5f}, L1_J: {:.5f}, L1_Z: {:.5f}, MPVPE_V: {:.5f}, MPVPE_J: {:.5f}".format(
                            body_part,
                            loss_dict[body_part+'_loss_net'].item(),
                            loss_dict[body_part+'_loss_v'].item(),
                            loss_dict[body_part+'_loss_j'].item(),
                            loss_dict[body_part+'_loss_z'].item(),
                            mean_error_v.item(),
                            mean_error_j.item())
                print(log_info)

        vis_cond1 = (exp_type == 'train') and (iteration % self.opt.training.train_vis_iter == 0)
        vis_cond2 = (exp_type == 'valid') and (iteration % self.opt.training.val_vis_iter == 0)
        if vis_cond1 or vis_cond2:
            target_verts = batch['human_verts']
            bid = [random.randint(0,target_verts.shape[0]-1)]
            
            pred_mesh_v = pred_dict['human_verts_v'][bid[0]:bid[0]+1].detach().cpu().numpy()
            dst_path = osp.join(vis_dir,'epo{:03d}_iter{:05d}_pred_v'.format(epoch,iteration))
            self.shapedata.save_meshes(dst_path, pred_mesh_v, bid)
            pred_mesh_j = pred_dict['human_verts_j'][bid[0]:bid[0]+1].detach().cpu().numpy()
            dst_path = osp.join(vis_dir,'epo{:03d}_iter{:05d}_pred_j'.format(epoch,iteration))
            self.shapedata.save_meshes(dst_path, pred_mesh_j, bid)
            target_mesh = target_verts[bid[0]:bid[0]+1].detach().cpu().numpy()
            dst_path = osp.join(vis_dir,'epo{:03d}_iter{:05d}_target'.format(epoch,iteration))
            self.shapedata.save_meshes(dst_path, target_mesh, bid)


    def train(self, data_loader, exp_type, epoch):
        for body_part in self.opt.body_part_list:
            self.train_loss_dict[body_part].reset()
        self.identity_model.train()
        self.skinning_model.train()
        for iteration, batch in enumerate(data_loader):
            self.run_iteration(epoch, iteration, len(data_loader), batch, exp_type, self.train_vis_dir)

    def validate(self, data_loader, exp_type, epoch):
        with torch.no_grad():
            self.identity_mpvpe.reset()
            for body_part in self.opt.body_part_list:
                self.val_loss_dict[body_part].reset()
                self.mpvpe_v_dict[body_part].reset()
                self.mpvpe_j_dict[body_part].reset()
            self.identity_model.eval()
            self.skinning_model.eval()
            for iteration, batch in enumerate(data_loader):
                self.run_iteration(epoch, iteration, len(data_loader), batch, exp_type, self.valid_vis_dir)

    def test(self, data_loader, exp_type, epoch=0):
        from utils.pose_evaluation import compute_similarity_transform
        from utils.mesh_evaluation import perform_registration
        from utils.pose_inference import get_hm_j2d, get_j3d_pred
        from utils.pose_data_utils import flip_back
        import utils.dist_chamfer as chamfer
        import utils.dist_emd as emd
        distChamfer = chamfer.chamferDist()
        distEmd = emd.emdDist()
        with torch.no_grad():
            self.pose_model.eval()
            self.identity_model.eval()
            self.skinning_model.eval()
            mpvpe_list,mpvpe_pa_list = [],[]
            emd_list, cd_list = [],[]
            for iteration, batch in enumerate(tqdm(data_loader)):
                inp_img = batch['inp_img'].to(self.device)
                inp_rest_verts = batch['inp_rest_verts'].to(self.device)
                # pose net forward
                output = self.pose_model(inp_img)
                pred_heatmaps = output[-1]['hm']
                pred_j3dmaps = output[-1]['j3d']
                # flip test
                input_flipped = np.flip(inp_img.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).to(self.device)
                output_flipped = self.pose_model(input_flipped)
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
                # norm j3d direction
                pred_j3d_np = pred_j3d.squeeze(0).cpu().numpy()
                rot_mat, inv_rot_mat = self.get_rot_mat(pred_j3d_np)
                j3d_norm = np.dot(rot_mat, pred_j3d_np.T).T
                inp_j3d = torch.from_numpy(j3d_norm.copy()).float().unsqueeze(0).to(self.device)
                # identity net forward
                pred_rest_verts = self.identity_model(inp_img, inp_rest_verts)
                rest_verts_dict = {}
                rest_verts_dict['human_rest_verts'] = pred_rest_verts
                # get predicted body part rest verts
                for body_part in self.opt.body_part_list:
                    body_part_idx = torch.from_numpy(constants.BODY_PART_INDEX[body_part]).long().to(self.device)
                    rest_verts_dict[body_part+'_rest_verts'] = rest_verts_dict['human_rest_verts'][:, body_part_idx]
                if self.opt.dataset.dummy_node:
                    # add dummy dimension
                    for body_part in self.opt.body_part_list:
                        cur_verts = rest_verts_dict[body_part+'_rest_verts']
                        zeros = torch.zeros((cur_verts.shape[0],1,cur_verts.shape[2])).float().to(self.device)
                        rest_verts_dict[body_part+'_rest_verts'] = torch.cat((cur_verts,zeros),1)
                # skinningNet forward
                pred_dict = self.skinning_model.test(inp_j3d, rest_verts_dict)
                pred_verts = pred_dict['human_verts_j'].squeeze(0).detach().cpu().numpy()
                target_verts = batch['human_verts'].squeeze(0).cpu().numpy()
                # compute mpvpe and mpvpe_pa
                pred_verts_pa = compute_similarity_transform(pred_verts.copy(), target_verts.copy())
                cur_mpvpe = np.mean(np.sqrt(np.sum((pred_verts-target_verts)**2,axis=-1)))
                cur_mpvpe_pa = np.mean(np.sqrt(np.sum((pred_verts_pa-target_verts)**2,axis=-1)))
                mpvpe_list.append(cur_mpvpe)
                mpvpe_pa_list.append(cur_mpvpe_pa)
                # compute cd and emd
                pred_verts_icp = perform_registration(pred_verts.copy(), target_verts.copy())
                pred_verts_icp_tensor = torch.from_numpy(pred_verts_icp).to(self.device).float().unsqueeze(0)
                target_verts_tensor = torch.from_numpy(target_verts).to(self.device).float().unsqueeze(0)
                emd_loss = distEmd(pred_verts_icp_tensor, target_verts_tensor)
                cf1_loss, cf2_loss = distChamfer(pred_verts_icp_tensor, target_verts_tensor)
                emd_list.append(torch.mean(emd_loss).item() / pred_verts_icp_tensor.shape[1])
                cd_list.append((torch.mean(cf1_loss).item() + torch.mean(cf2_loss).item()) / 2.)
                # visualization
                if iteration % self.opt.training.test_vis_iter == 0:
                    target_verts = batch['human_verts']
                    bid = [0]
                    pred_mesh_j = pred_dict['human_verts_j'][bid[0]:bid[0]+1].detach().cpu().numpy()
                    dst_path = osp.join(self.test_vis_dir,'epo{:03d}_iter{:05d}_pred_j'.format(epoch,iteration))
                    self.shapedata.save_meshes(dst_path, pred_mesh_j, bid)
                    target_mesh = target_verts[bid[0]:bid[0]+1].detach().cpu().numpy()
                    dst_path = osp.join(self.test_vis_dir,'epo{:03d}_iter{:05d}_target'.format(epoch,iteration))
                    self.shapedata.save_meshes(dst_path, target_mesh, bid)

            mean_mpvpe = np.mean(mpvpe_list)
            mean_mpvpe_pa = np.mean(mpvpe_pa_list)
            mean_emd = np.mean(emd_list)
            mean_cd = np.mean(cd_list)
            log_info = 'MPVPE: {:.5f}, MPVPE-PA: {:.5f}, EMD: {:.5f}, CD: {:.5f}\n'.format(
                mean_mpvpe*1000, mean_mpvpe_pa*1000,
                mean_emd, mean_cd*1000)
            print(log_info)
            with open(self.log_path, 'a') as f:
                f.write(log_info)


    def get_rot_mat(self, j3d, a_id=0, b_id=10, c_id=13):
        # a: pelvis, b: lshoulder, c: rshoulder
        v_norm = np.cross(j3d[b_id]-j3d[a_id], j3d[c_id]-j3d[a_id])
        v_norm = np.array([v_norm[0],0,v_norm[2]])
        gt_norm = np.array([1.,0,0])
        v_norm = v_norm / np.linalg.norm(v_norm)
        gt_norm = gt_norm / np.linalg.norm(gt_norm)
        angle = np.arccos(np.dot(v_norm, gt_norm))
        if v_norm[2] < 0:
            angle = 2 * np.pi - angle
        axis = np.array([0,1.,0])
        axis_angle = axis * angle
        rot_mat = R_axis_angle(axis, angle)
        inv_rot_mat = rot_mat.T
        return rot_mat, inv_rot_mat

    def demo(self, inp_dir, out_dir):
        obj_dir = osp.join(out_dir, 'objs')
        os.makedirs(obj_dir, exist_ok=True)
        
        # prepare j3d
        all_j3d = np.load(osp.join(inp_dir,'npy/j3d.npy'))
        # prepare img_paths
        all_img_paths = pickle.load(open(osp.join(inp_dir,'npy/img_paths.pkl'),'rb'))

        assert len(all_img_paths) == all_j3d.shape[0]
        
        # img preprocessing operator
        normalize = transforms.Normalize(mean=constants.IMG_MEAN,
                                    std=constants.IMG_NORM)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        # prepare rest pose
        pkl_path = osp.join(self.opt.data_root_dir, 'template', 'template.pkl')
        self.rest_verts_dict = pickle.load(open(pkl_path,'rb'))
        inp_rest_verts = self.rest_verts_dict['mean']
        inp_rest_verts = torch.tensor(inp_rest_verts.copy()).unsqueeze(0).float().to(self.device)
        
        # demo
        with torch.no_grad():
            self.identity_model.eval()
            self.skinning_model.eval()
            pred_verts_list = []
            for i in tqdm(range(all_j3d.shape[0])):
                inp_img = cv2.imread(all_img_paths[i])
                inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
                inp_img = img_transform(inp_img).unsqueeze(0).to(self.device)
                # identity net forward
                pred_rest_verts = self.identity_model(inp_img, inp_rest_verts)
                rest_verts_dict = {}
                rest_verts_dict['human_rest_verts'] = pred_rest_verts
                # get predicted body part rest verts
                for body_part in self.opt.body_part_list:
                    body_part_idx = torch.from_numpy(constants.BODY_PART_INDEX[body_part]).long().to(self.device)
                    rest_verts_dict[body_part+'_rest_verts'] = rest_verts_dict['human_rest_verts'][:, body_part_idx]
                if self.opt.dataset.dummy_node:
                    # add dummy dimension
                    for body_part in self.opt.body_part_list:
                        cur_verts = rest_verts_dict[body_part+'_rest_verts']
                        zeros = torch.zeros((cur_verts.shape[0],1,cur_verts.shape[2])).float().to(self.device)
                        rest_verts_dict[body_part+'_rest_verts'] = torch.cat((cur_verts,zeros),1)
                # norm j3d direction
                j3d = all_j3d[i]
                rot_mat, inv_rot_mat = self.get_rot_mat(j3d)
                j3d_norm = np.dot(rot_mat, j3d.T).T
                inp_j3d = torch.from_numpy(j3d_norm.copy()).float().unsqueeze(0).to(self.device)
                # skinningNet forward
                pred_dict = self.skinning_model.test(inp_j3d, rest_verts_dict)
                pred_verts = pred_dict['human_verts_j']
                pred_verts_np = pred_verts[0].detach().cpu().numpy()
                pred_verts_np = np.dot(inv_rot_mat, pred_verts_np.T).T
                pred_verts_np = move_position(pred_verts_np)
                dst_path = osp.join(obj_dir, all_img_paths[i].split('/')[-1].replace('.png','.obj') )
                writeObj(pred_verts_np, constants.REF_MESH['human'], dst_path)
                pred_verts_list.append(pred_verts_np)
            all_pred_verts = np.array(pred_verts_list)
            np.save(osp.join(out_dir,'all_pred_verts.npy'), all_pred_verts)


def move_position(cur_verts):
    seg_map_dict = {
        0: 'head',
        1: 'larm',
        2: 'rarm',
        3: 'lshoes',
        4: 'rshoes',
        5: 'shirt',
        6: 'pant',
        7: 'lleg',
        8: 'rleg'
    }
    verts_segm = np.load('../data/mesh/seg_files/verts_segm_lr.npy')
    seg_part_dict = {}
    for part_id, part_name in seg_map_dict.items():
        v_id = np.where(verts_segm == part_id)[0]
        seg_part_dict[part_name] = v_id

    # pant leg relative position
    rleg_idx = [208,203,186,234,224,189,215,187]
    rleg_idx = [4343+i for i in rleg_idx]
    rpant_idx = [42,29,11,63,53,17,42,9]
    rpant_idx = [4175+i for i in rpant_idx]
    offset = np.mean(cur_verts[rpant_idx],0) - np.mean(cur_verts[rleg_idx],0)
    cur_verts[seg_part_dict['rleg']] += offset
    cur_verts[seg_part_dict['rshoes']] += offset
    
    lleg_idx = [111,117,118,115,123,109,116,126]
    lleg_idx = [4343+i for i in lleg_idx]
    lpant_idx = [163,95,118,139,108,150,145,101]
    lpant_idx = [4175+i for i in lpant_idx]
    offset = np.mean(cur_verts[lpant_idx],0) - np.mean(cur_verts[lleg_idx],0)
    cur_verts[seg_part_dict['lleg']] += offset
    cur_verts[seg_part_dict['lshoes']] += offset
    
    # leg shoes relative position
    offset = cur_verts[4683] - cur_verts[1999]
    cur_verts[seg_part_dict['rshoes']] += offset
    offset = cur_verts[4501] - cur_verts[1478]
    cur_verts[seg_part_dict['lshoes']] += offset
    
    return cur_verts

def createAndRunTrainer(gpu_id, opt):
    opt.gpu_id = gpu_id
    trainer = TrainMesh(opt)
    
    # Training
    if opt.exp_type == 'train':
        print("=> Start Training")
        trainer.train_epoch()
    elif opt.exp_type == 'test':
        trainer.test(trainer.test_data_loader, 'test')
    elif opt.exp_type == 'demo':
        print("=> Run Demo ")
        print(opt.demo.inp_dir, opt.demo.out_dir)
        trainer.demo(opt.demo.inp_dir, opt.demo.out_dir)