# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
from glob import glob
import time

import pickle

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from mesh_intersection.filter_faces import FilterFaces
from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss


def writeObj(vertices, src_path, dst_path):
    src_obj = open(src_path,'r')
    dst_obj = open(dst_path, 'w')
    src_obj_content = src_obj.readlines()
    
    # write header
    dst_obj.write(src_obj_content[0])
    dst_obj.write(src_obj_content[1])
    dst_obj.write(src_obj_content[2])
    dst_obj.write(src_obj_content[3])
    dst_obj.write(src_obj_content[4])
    dst_obj.write(src_obj_content[5])

    
    # Write vertices
    for vertex in vertices:
        dst_obj.write("v %.5f %.5f %.5f\n" %(vertex[0], vertex[1], vertex[2]))

    # Write texcoords and faces
    for line in src_obj_content[6:]:
        if line.split(' ')[0] == 'vt':
            dst_obj.write(line)
        if line.split(' ')[0] == 'f':
            dst_obj.write(line)
        if line.split(' ')[0] == 'g':
            dst_obj.write(line)
        if line.split(' ')[0] == 'usemtl':
            dst_obj.write(line)
        if line.split(' ')[0] == '\n':
            dst_obj.write(line)
        if line.split(' ')[0] == '#':
            dst_obj.write(line)

def readObj(file_path):
    obj_file = open(file_path,'r')
    contents = obj_file.readlines()
    vertices, texcoords, faces = [],[],[]
    for line in contents:
        parts = line.split(' ')
        if parts[0] == 'v':
            vertex = [float(i) for i in parts[1:4]]
            vertices.append(vertex)
        if parts[0] == 'vt':
            texcoord = [float(i) for i in parts[1:3]]
            texcoords.append(texcoord)
        if parts[0] == 'f':
            face = [int(i.split('/')[0]) for i in parts[1:4]]
            faces.append(face)
    vertices = np.array(vertices)
    texcoords = np.array(texcoords)
    faces = np.array(faces)
    return vertices, texcoords, faces

def main():
    description = 'Example script for untangling Mesh self intersections'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Batch Mesh Untangle')
    parser.add_argument('--data_folder', type=str,
                        default='data',
                        help='The path to data')
    parser.add_argument('--point2plane', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use point to distance')
    parser.add_argument('--sigma', default=0.5, type=float,
                        help='The height of the cone used to calculate the' +
                        ' distance field loss')
    parser.add_argument('--lr', default=1, type=float,
                        help='The learning rate for SGD')
    parser.add_argument('--coll_loss_weight', default=1e-4, type=float,
                        help='The weight for the collision loss')
    parser.add_argument('--verts_reg_weight', default=1e-5, type=float,
                        help='The weight for the verts regularizer')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--iterations', default=100, type=int,
                        help='Number of optimization iterations')

    args = parser.parse_args()
    data_folder = args.data_folder
    point2plane = args.point2plane
    lr = args.lr
    coll_loss_weight = args.coll_loss_weight
    max_collisions = args.max_collisions
    sigma = args.sigma
    iterations = args.iterations


    device = torch.device('cuda')
    
    obj_paths = sorted(glob(data_folder+'/*.obj'))
    verts_list,faces_list = [],[]
    for obj_path in obj_paths:
        cur_verts, _, cur_faces = readObj(obj_path)
        cur_faces -= 1
        verts_list.append(cur_verts)
        faces_list.append(cur_faces)
    all_verts = np.array(verts_list).astype(np.float32)
    all_faces = np.array(faces_list).astype(np.int64)

    verts_tensor = torch.tensor(all_verts.copy(), dtype=torch.float32,
                               device=device)
    face_tensor = torch.tensor(all_faces, dtype=torch.long,
                               device=device)
    param = torch.tensor(all_verts.copy(), dtype=torch.float32,
                               device=device).requires_grad_()
    bs, nv = verts_tensor.shape[:2]
    bs, nf = face_tensor.shape[:2]
    faces_idx = face_tensor + \
        (torch.arange(bs, dtype=torch.long).to(device) * nv)[:, None, None]

    # Create the search tree
    search_tree = BVH(max_collisions=max_collisions)

    pen_distance = \
        collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                     point2plane=point2plane,
                                                     vectorized=True)

    mse_loss = nn.MSELoss(reduction='sum').to(device=device)


    optimizer = torch.optim.SGD([param], lr=lr)

    step = 0
    for i in range(iterations)
        optimizer.zero_grad()

        triangles = param.view([-1, 3])[faces_idx]

        with torch.no_grad():
            collision_idxs = search_tree(triangles)

        pen_loss = coll_loss_weight * \
            pen_distance(triangles, collision_idxs)

        verts_reg_loss = torch.tensor(0, device=device,
                                     dtype=torch.float32)
        if verts_reg_weight > 0:
            verts_reg_loss = verts_reg_weight * \
                mse_loss(param, verts_tensor)

        loss = pen_loss + verts_reg_loss

        np_loss = loss.detach().cpu().squeeze().tolist()
        if type(np_loss) != list:
            np_loss = [np_loss]
        msg = '{:.5f} ' * len(np_loss)
        print('Loss per model:', msg.format(*np_loss))
        
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        step += 1

    optimized_verts = param.detach().cpu().numpy()
    for i in range(optimized_verts.shape[0]):
        

if __name__ == '__main__':
    main()
