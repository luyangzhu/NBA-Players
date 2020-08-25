import os
import os.path as osp
from glob import glob
import pickle
import shutil
import numpy as np
import cv2
from utils import *


def load_j2d(data_dir, result_dir, idx, img_name):
    ''' load 2d joints '''
    print('Load 2d joints')
    all_j2d = np.load(osp.join(result_dir,'npy','j2d.npy'))
    proc_param_path = osp.join(data_dir, 'proc_param','{}.pkl'.format(img_name))
    proc_param = pickle.load(open(proc_param_path,'rb'))
    cur_j2d = all_j2d[idx]
    # transform back to orginal image coords
    start_pt = proc_param['start_pt']
    scale = proc_param['scale']
    cur_j2d = (cur_j2d + start_pt.reshape(1,-1) - 128.) / scale
    return cur_j2d

def load_j3d(result_dir, idx):
    ''' load 3d joints '''
    print('Load 3d joints')
    all_j3d = np.load(osp.join(result_dir,'npy','j3d.npy'))
    cur_j3d = all_j3d[idx]
    return cur_j3d

def load_jump_height(result_dir, idx):
    ''' load jump height '''
    print('Load jump height')
    all_jump_dist = np.load(osp.join(result_dir,'npy','all_jump_dist.npy'))
    all_jump_cls = np.load(osp.join(result_dir,'npy','all_jump_cls.npy'))
    cur_jump_dist = all_jump_dist[idx]
    cur_jump_cls = all_jump_cls[idx]
    if cur_jump_cls == 0:
        return 0
    else:
        # set threshold for jumping
        return min(1.0, cur_jump_dist)

def load_cam(result_dir, img_name):
    ''' load camera parameters '''
    print('load camera parameters')
    A = pickle.load(open(osp.join(result_dir, 'cams', '{}_A.npy'.format(img_name)),'rb'))
    R = pickle.load(open(osp.join(result_dir, 'cams', '{}_R.npy'.format(img_name)),'rb'))
    T = pickle.load(open(osp.join(result_dir, 'cams', '{}_T.npy'.format(img_name)),'rb')).ravel()
    return A, R, T

def load_lowet_mesh_vertex(result_dir, img_name):
    ''' get the lowest mesh vertex '''
    print('get the lowest mesh vertex')
    obj_path = osp.join(result_dir, 'objs','{}.obj'.format(img_name))
    all_v,_,_,_ = readObjV2(obj_path)
    ind = np.argmin(all_v, axis=0)
    lowest_v = all_v[ind[1]]
    return lowest_v

def compute_offset(cur_j2d, cur_j3d, cur_lowest_v, A, R, T, jump_height=0):
    ''' compute offset '''
    print('compute offset')
    min_id = np.argmin(cur_j3d,0)[1]
    xp = cur_j2d[min_id,0]
    yp = cur_j2d[min_id,1]
    v_normalize = cur_j3d[min_id]
    jump_height = jump_height + cur_j3d[min_id,1] - cur_lowest_v[1]
    fx = A[0,0]
    fy = A[1,1]
    cx = A[0,2]
    cy = A[1,2]
    cur_R = R
    cur_T = T
    s = np.array([(xp-cx)/fx,(yp-cy)/fy,1])
    r = cur_R[:,1]
    zc = (r.dot(cur_T)+jump_height) / r.dot(s)
    v_cam = s * zc
    v_world = np.dot(cur_R.T, (v_cam - cur_T).reshape(3,1)).ravel()
    offset = v_world - v_normalize
    return offset

def write_mesh(offset, result_dir, img_name, aux_angle=0):
    # write results
    print('write results')
    obj_dir = osp.join(result_dir, 'objs')
    write_dir = osp.join(result_dir, 'global_objs')
    os.makedirs(write_dir, exist_ok=True)

    # if inp image is not in broadcast view, 
    # need to manually set a rough rotation from broadcast view to input view.
    axis = np.array([0,1.,0 ])
    rot_mat = R_axis_angle(axis, aux_angle)
    obj_path = osp.join(obj_dir, '{}.obj'.format(img_name))
    verts,_,_,_ = readObjV2(obj_path)
    verts = np.dot(rot_mat, verts.T).T
    verts += offset
    dst_path = osp.join(write_dir, '{}.obj'.format(img_name))
    writeObj(verts, obj_path, dst_path)

def main():
    subdir = 'lbj_dunk'
    data_dir = '../../data/{}'.format(subdir)
    result_dir = '../../results/{}'.format(subdir)
    img_path_list = sorted(glob(osp.join(data_dir, 'img_crop', '*.png')), key=osp.getmtime)
    
    for idx, img_path in enumerate(img_path_list):
        img_name = img_path.split('/')[-1].replace('.png','')
        cur_j2d = load_j2d(data_dir, result_dir, idx, img_name)
        cur_j3d = load_j3d(result_dir, idx)
        cur_jump_dist = load_jump_height(result_dir, idx)
        substr = '_'+img_path.split('/')[-1].split('_')[-1]
        full_img_name = img_path.split('/')[-1].replace(substr, '')
        A, R, T = load_cam(result_dir, full_img_name)
        cur_lowest_v = load_lowet_mesh_vertex(result_dir, img_name)
        offset = compute_offset(cur_j2d, cur_j3d, cur_lowest_v, A, R, T, cur_jump_dist)
        write_mesh(offset, result_dir, img_name)

if __name__ == '__main__':
    main()