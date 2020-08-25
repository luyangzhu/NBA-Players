import os
import os.path as osp
from glob import glob
import pickle
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.mesh_utils import R_axis_angle, readObjV2
import constants


class MeshJ3DGeneraldataset(Dataset):
    def __init__(self, opt, root_dir, exp_type, player_list, dummy_node = True, dir_types=['2ku', 'normal']):
        self.opt = opt
        self.root_dir = root_dir
        self.exp_type = exp_type
        self.player_list = player_list
        self.dummy_node = dummy_node

        # collect data
        j3d_list, verts_list, img_path_list = [],[],[]
        for cur_player in tqdm(self.player_list):
            for dir_type in dir_types:
                pkl_path = osp.join(self.root_dir, 'release', cur_player, 'release_{}_{}.pkl'.format(cur_player, dir_type))
                meta_data = pickle.load(open(pkl_path, 'rb'))
                j3d_list += meta_data['j3d']
                verts_list += meta_data['human_verts']
                for nba_dir, person_id in zip(meta_data['nba_dir'], meta_data['person_id']):
                    img_path = osp.join(self.root_dir, 'release', cur_player, 'images', dir_type, '{}_{:02d}.png'.format(nba_dir, person_id))
                    img_path_list.append(img_path)

        self.j3d = np.array(j3d_list)
        try:
            self.flip_pairs = constants.JOINTS_FLIP_PAIRS[self.opt.dataset.joints_type]
            self.parent_ids = constants.JOINTS_PARENT_IDS[self.opt.dataset.joints_type]
            self.joint_index = constants.JOINTS_INDEX[self.opt.dataset.joints_type]
            self.right_joint_list = constants.RIGHT_JOINTS[self.opt.dataset.joints_type]
            self.foot_joint_index = constants.FOOT_JOINTS[self.opt.dataset.joints_type]
        except:
            raise ValueError('joints type not supported')

        self.j3d = self.j3d[:,self.joint_index] # select joints according to joints type
        self.verts = np.array(verts_list)
        self.img_paths = np.array(img_path_list)
        assert self.j3d.shape[0] == self.verts.shape[0] == self.img_paths.shape[0]
        print(self.j3d.shape, self.verts.shape, self.img_paths.shape)

        # prepare rest verts dict
        pkl_path = osp.join(self.root_dir, 'template', 'template.pkl')
        self.rest_verts_dict = pickle.load(open(pkl_path,'rb'))

        # img preprocessing
        normalize = transforms.Normalize(mean=constants.IMG_MEAN,
                                    std=constants.IMG_NORM)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


    def rotate(self, j3d, verts, begin_rad=-np.pi/6, end_rad=np.pi/6):
        axis = np.array([0,1.,0 ])
        azim = np.random.random() * (end_rad-begin_rad) + begin_rad
        rot_mat = R_axis_angle(axis, azim)
        rot_j3d = np.dot(rot_mat, j3d.T).T
        rot_verts = np.dot(rot_mat, verts.T).T
        
        return rot_j3d, rot_verts


    def j3d_preprocessing(self, j3d):
        nParts = j3d.shape[0]
        j3d = j3d.T
        noise_x = np.random.normal(0,self.opt.dataset.j3d_factor,nParts)
        noise_y = np.random.normal(0,self.opt.dataset.j3d_factor,nParts)
        noise_z = np.random.normal(0,self.opt.dataset.j3d_factor,nParts)
        j3d[0] = j3d[0] + noise_x
        j3d[1] = j3d[1] + noise_y
        j3d[2] = j3d[2] + noise_z
        j3d = j3d.T
        return j3d

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        cur_player_name = img_path.split('/')[-4]
        inp_img = cv2.imread(img_path)
        # BGR -> RGB
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
        inp_img = self.img_transform(inp_img)
        
        cur_verts = self.verts[idx]
        inp_j3d = self.j3d[idx]
        inp_rest_verts = self.rest_verts_dict['mean']
        tgt_rest_verts = self.rest_verts_dict[cur_player_name]
        # add noise for training
        if self.exp_type == 'train' and self.opt.dataset.is_augmentation:
            if np.random.random() > 0.7:
                inp_j3d, cur_verts = self.rotate(inp_j3d, cur_verts)
            inp_j3d = self.j3d_preprocessing(inp_j3d)
        
        item = {}
        item['inp_img'] = inp_img
        item['inp_j3d'] = torch.tensor(inp_j3d.copy()).float()
        item['human_verts'] = torch.tensor(cur_verts.copy()).float()
        item['tgt_rest_verts'] = torch.tensor(tgt_rest_verts.copy()).float()
        item['inp_rest_verts'] = torch.tensor(inp_rest_verts.copy()).float()
        
        for body_part in self.opt.body_part_list:
            body_part_idx = constants.BODY_PART_INDEX[body_part]
            item[body_part+'_verts'] = cur_verts[body_part_idx]
        
        # process dummy node
        if self.dummy_node:
            for body_part in self.opt.body_part_list:
                cur_verts = item[body_part+'_verts']
                verts_dummy = np.zeros((cur_verts.shape[0]+1,cur_verts.shape[1]),dtype=np.float32)
                verts_dummy[:-1,:] = cur_verts
                item[body_part+'_verts'] = torch.tensor(verts_dummy.copy()).float()
                
        else:
            for body_part in self.opt.body_part_list:
                item[body_part+'_verts'] = torch.tensor(item[body_part+'_verts'].copy()).float()


        return item

    def __len__(self):
        return self.j3d.shape[0]


def get_dataset(opt, root_dir, exp_type, dummy_node, dir_types):
    if exp_type == 'train':
        player_list = constants.TRAIN_PLAYERS
    elif exp_type == 'valid':
        player_list = constants.VALID_PLAYERS
    elif exp_type == 'test':
        player_list = constants.TEST_PLAYERS
    dataset = MeshJ3DGeneraldataset(opt, root_dir, exp_type, player_list, 
                dummy_node=dummy_node, dir_types=dir_types)
    return dataset