import torch
import torch.nn as nn
import models.mesh_net as mesh_net
from opt import *
from utils.mesh_utils import *
from utils.training_utils import *
import constants

__all__ = ['HumanNetV2']


class HumanNetV2(nn.Module):
    def __init__(self, opt, loc, sub_model_name='SpiralTLNetV2'):
        super(HumanNetV2, self).__init__()
        self.opt = opt
        self.loc = loc
        self.device = torch.device(self.loc)
        # init body part model
        for body_part in self.opt.body_part_list:
            print('init {} net'.format(body_part))
            model = self.init_model(body_part, sub_model_name)
            setattr(self, '{}_net'.format(body_part), model)

    def init_model(self, body_part, sub_model_name):
        # process mesh
        shapedata, tspirals, spiral_sizes, \
        vnum, tD, tU = process_mesh(self.opt, body_part, self.device)
        # config models
        model = getattr(mesh_net, sub_model_name)(
                            filters_enc=constants.FILTER_SIZES_ENC[body_part],   
                            filters_dec=constants.FILTER_SIZES_DEC[body_part],
                            latent_size=constants.NZ[body_part],
                            sizes=vnum,
                            spiral_sizes=spiral_sizes,
                            nParts=self.opt.dataset.nParts,
                            spirals=tspirals,
                            D=tD, U=tU,
                            device=self.device
                        )
        model = model.to(self.device)
        return model

    def load_checkpoint(self, checkpoint_dir, prefix):
        if checkpoint_dir is not None:
            for body_part in self.opt.body_part_list:
                checkpoint_path = osp.join(checkpoint_dir, '{}_{}_network.pth'.format(prefix, body_part))
                if checkpoint_path is not None and osp.isfile(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=self.loc)
                    model = getattr(self, '{}_net'.format(body_part))
                    restore(model, checkpoint['model'])
                    print("=> Successfully loaded checkpoint at '{}'".format(checkpoint_path))


    def forward(self, batch, inp_j3d):
        pred_dict = {}
        pred_verts_v_list,pred_z_v_list = [],[]
        pred_verts_j_list,pred_z_j_list = [],[]
        for body_part in self.opt.body_part_list:
            model = getattr(self, '{}_net'.format(body_part))
            inp_verts = batch[body_part+'_verts'].to(self.device)
            inp_rest_verts = batch[body_part+'_rest_verts'].to(self.device)
            z_v, pred_verts_v, z_j, pred_verts_j = model.forward(inp_verts, inp_j3d, inp_rest_verts)
            pred_dict[body_part+'_z_v'] = z_v
            pred_dict[body_part+'_verts_v'] = pred_verts_v
            pred_dict[body_part+'_z_j'] = z_j
            pred_dict[body_part+'_verts_j'] = pred_verts_j
            if self.opt.dataset.dummy_node:
                pred_verts_v_list.append(pred_verts_v[:,:-1])
            else:
                pred_verts_v_list.append(pred_verts_v)
            pred_z_v_list.append(z_v)
            if self.opt.dataset.dummy_node:
                pred_verts_j_list.append(pred_verts_j[:,:-1])
            else:
                pred_verts_j_list.append(pred_verts_j)
            pred_z_j_list.append(z_j)
        
        human_verts_v = torch.cat(pred_verts_v_list, dim=1)
        pred_dict['human_verts_v'] = human_verts_v
        human_z_v = torch.cat(pred_z_v_list, dim=1)
        pred_dict['human_z_v'] = human_z_v
        human_verts_j = torch.cat(pred_verts_j_list, dim=1)
        pred_dict['human_verts_j'] = human_verts_j
        human_z_j = torch.cat(pred_z_j_list, dim=1)
        pred_dict['human_z_j'] = human_z_j
        return pred_dict


    def test(self, j3d, rest_verts_dict):
        pred_dict = {}
        pred_verts_j_list = []
        pred_z_j_list = []
        for body_part in self.opt.body_part_list:
            model = getattr(self, '{}_net'.format(body_part))
            cur_rest_verts = rest_verts_dict[body_part+'_rest_verts']
            pred_verts_j, z_j = model.test(j3d, cur_rest_verts)
            pred_dict[body_part+'_verts_j'] = pred_verts_j
            pred_dict[body_part+'_z_j'] = z_j
            if self.opt.dataset.dummy_node:
                pred_verts_j_list.append(pred_verts_j[:,:-1])
            else:
                pred_verts_j_list.append(pred_verts_j)
            pred_z_j_list.append(z_j)
        human_verts_j = torch.cat(pred_verts_j_list, dim=1)
        human_z_j = torch.cat(pred_z_j_list, dim=1)
        pred_dict['human_verts_j'] = human_verts_j
        pred_dict['human_z_j'] = human_z_j
        return pred_dict
