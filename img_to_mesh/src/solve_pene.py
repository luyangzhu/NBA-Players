# -*- coding: utf-8 -*-
import sys
import os
import os.path as osp
from glob import glob

import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from mesh_intersection.bvh_search_tree import BVH
from utils.mesh_loss import LaplacianLoss, init_edge_gt, compute_edge_loss

def getVertexNorm_np(vertices, faces):
    def dot(v1, v2):
        return np.sum(v1 * v2, axis = 1)
    def squared_length(v):
        return np.sum(v * v, axis = 1)
    def length(v):
        return np.sqrt(squared_length(v))
    # Nelson Max, "Weights for Computing Vertex Normals from Facet Vectors", 1999
    normals = np.zeros(vertices.shape, dtype = np.float32)
    v = [vertices[faces[:, 0], :],
         vertices[faces[:, 1], :],
         vertices[faces[:, 2], :]]
    for i in range(3):
        v0 = v[i]
        v1 = v[(i + 1) % 3]
        v2 = v[(i + 2) % 3]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_len = length(e1)
        e2_len = length(e2)
        side_a = e1 / e1_len.reshape((-1,1))
        side_b = e2 / e2_len.reshape((-1,1))
        if i == 0:
            n = np.cross(side_a, side_b)
            n = n / length(n).reshape((-1,1))
        angle = np.where(dot(side_a, side_b) < 0, 
            math.pi - 2.0 * np.arcsin(0.5 * length(side_a + side_b)),
            2.0 * np.arcsin(0.5 * length(side_b - side_a)))
        sin_angle = np.sin(angle)
        
        contrib = n * (sin_angle / (e1_len * e2_len)).reshape((-1,1))
        np.add.at(normals,faces[:,i],contrib)

    normals = normals / length(normals).reshape((-1,1))
    return normals


def getVertexNorm(vertices, faces):
    # Nelson Max, "Weights for Computing Vertex Normals from Facet Vectors", 1999
    normals = torch.zeros_like(vertices)
    v = [vertices[faces[:, 0]],
         vertices[faces[:, 1]],
         vertices[faces[:, 2]]]
    for i in range(3):
        v0 = v[i]
        v1 = v[(i + 1) % 3]
        v2 = v[(i + 2) % 3]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_len = torch.norm(e1, dim=1, keepdim=True)
        e2_len = torch.norm(e2, dim=1, keepdim=True)
        side_a = e1 / e1_len
        side_b = e2 / e2_len
        if i == 0:
            n = torch.cross(side_a, side_b)
            n = n / torch.norm(n, dim=1, keepdim=True)
        cond = torch.sum(side_a*side_b,-1) < 0
        val_a = math.pi - 2.0 * torch.asin(0.5 * torch.norm(side_a + side_b, dim=1))
        val_b = 2.0 * torch.asin(0.5 * torch.norm(side_b - side_a, dim=1))
        angle = torch.where(cond, val_a, val_b)
        sin_angle = torch.sin(angle)
        
        # XXX: Inefficient but it's PyTorch's limitation
        contrib = n * sin_angle.unsqueeze_(-1) / (e1_len * e2_len)
        normals.index_add_(0, faces[:,i], contrib)

    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    return normals

def getFaceNormals(verts):
    # verts: (Num_faces,3,3)

    edAB = verts[:,1] - verts[:,0]
    edAC = verts[:,2] - verts[:,0]
    face_norm = torch.cross(edAB, edAC, dim=1)
    face_norm = face_norm / torch.norm(face_norm,dim=1,keepdim=True)
    return face_norm

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

def iterative_solve_collision(verts_tensor, face_tensor, face_segm_tensor, edge_target, device, 
    iter_num=10, max_collisions=8, step_size=0.5, w_data=1, w_lap=0.1, w_el=0.1):
    
    search_tree = BVH(max_collisions=max_collisions)
    bs, nv = verts_tensor.shape[:2]
    bs, nf = face_tensor.shape[:2]
    faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(device) * nv)[:, None, None]
    laplaceloss = LaplacianLoss(faces_idx, verts_tensor, toref=True)
    # detect and remove mesh interpenetration between body and garments.
    for i in range(iter_num):
        triangles = verts_tensor.view([-1, 3])[faces_idx]
        collision_idxs = search_tree(triangles)
        # (collide_num,2): bs and face index for valid collide pairs in collision_idxs
        val_id = collision_idxs[:,:,0].ge(0).nonzero()
        # (collide_num, 2): collide pairs face id
        val_col_idx = collision_idxs[val_id[:,0],val_id[:,1]]
        # (collide_num,): intruder and receiver face type
        face_type_a = face_segm_tensor[val_col_idx[:,0]]
        face_type_b = face_segm_tensor[val_col_idx[:,1]]
        ''' 6: 'pant', 7: 'lleg', 8: 'rleg' 
            1: 'larm', 2: 'rarm 5: 'shirt' '''
        # bvh store collide face in ascending order, face type also in ascending order
        lleg_mask = (face_type_a == 6) & (face_type_b == 7)
        rleg_mask = (face_type_a == 6) & (face_type_b == 8)
        leg_mask = (lleg_mask + rleg_mask).ge(1).to(collision_idxs.dtype)
        leg_mask_idx = leg_mask.nonzero().reshape(-1)
        leg_num = leg_mask_idx.shape[0]
                
        # body-garment has no collision pairs
        if leg_num <= 0:
            break
        # all valid col pairs -> body and garment valid col pairs. (store face tensor index)
        body_garm_col_idx = torch.zeros(leg_num,2).long().to(device)
        body_garm_col_idx[0:leg_num] = val_col_idx[leg_mask_idx]

        # garment: 0 to index
        col_garment_face = face_tensor[0,body_garm_col_idx[:,0]] #(filterd collide_num, 3)
        col_garment_verts = verts_tensor[0,col_garment_face] # (filterd collide num,3,3)
        # body: 1 to index
        col_body_face = face_tensor[0,body_garm_col_idx[:,1]] #(filterd collide_num, 3)
        col_body_verts = verts_tensor[0,col_body_face] # (filterd collide num,3,3)
        # compute collision garment face normals
        col_garment_face_norm = getFaceNormals(col_garment_verts)
        # compute point2face distance from collide body verts to corresponding garment face
        # (filterd collide num,3)
        p2s = torch.sum((col_body_verts - col_garment_verts[:,0:1])*col_garment_face_norm.unsqueeze_(1),-1)
        # for every tri-tri col, only select outside body verts with smallest p2s
        outside_mask = p2s.gt(0).float()
        bound_p2s = outside_mask * p2s + (1-outside_mask)*100
        min_val, min_ind = torch.min(bound_p2s,dim=1)
        col_ind = torch.arange(min_ind.shape[0]).long().to(device)
        out_verts_fidx = torch.cat((col_ind.unsqueeze_(-1), min_ind.unsqueeze_(-1)),-1)
        # get verts index for outside body verts, (outside_vnum,)
        out_verts_vidx = col_body_face[out_verts_fidx[:,0], out_verts_fidx[:,1]]
        # remove dup
        out_verts_vidx_nodup, inverse_ind, counts = torch.unique(out_verts_vidx,
                                        return_inverse=True,return_counts=True)
        print("outside verts num: {}".format(out_verts_vidx_nodup.shape[0]))
        ''' compute offset '''
        # get p2s for outside body verts
        out_verts_p2s = p2s[out_verts_fidx[:,0], out_verts_fidx[:,1]]
        offset = torch.zeros(out_verts_vidx_nodup.shape).float().to(device)
        offset.index_add_(0,inverse_ind, out_verts_p2s)
        offset = offset / counts.float()
        
        ''' compute direction ''' 
        # get corresponding garment face norm for outside body verts
        out_face_norm = col_garment_face_norm[out_verts_fidx[:,0]]
        direction = torch.zeros(out_verts_vidx_nodup.shape[0],3).float().to(device)
        direction.index_add_(0,inverse_ind,out_face_norm)
        direction = direction / counts.unsqueeze_(-1).float()
        verts_tensor[0,out_verts_vidx_nodup] -= step_size * direction * offset.unsqueeze_(-1)


        ''' optimize non-detected vertices '''
        # get non-detected vertices id
        all_vid = np.arange(verts_tensor.shape[1]).tolist()
        out_verts_vidx_nodup_list = out_verts_vidx_nodup.cpu().numpy().tolist()
        optim_vid = [i for i in all_vid if i not in out_verts_vidx_nodup_list]
        optim_vid = torch.Tensor(optim_vid).long().to(device)
        # prepare params and optimizer
        pred_verts = verts_tensor.clone()
        params = pred_verts[:, optim_vid].requires_grad_()
        optimizer = optim.LBFGS([params],
                                lr=0.001, line_search_fn='strong_wolfe', max_iter=20)
        # run optimization
        for i in range(5):
            def closure():
                optimizer.zero_grad()
                pred_verts = verts_tensor.clone()
                pred_verts[:, optim_vid] = params
                lap_loss = laplaceloss(pred_verts)
                edge_loss = compute_edge_loss(pred_verts, faces_idx[0], edge_target)
                data_loss = F.mse_loss(pred_verts, verts_tensor)
                loss = w_data * data_loss + w_lap * lap_loss + w_el * edge_loss
                loss.backward()
                return loss
            optimizer.step(closure)

def main():
    obj_path = '../../results/lbj_dunk/objs/lbj_dunk_0003.obj'
    device = torch.device('cuda')

    seg_map_dict = {
        'head': 0,
        'larm': 1,
        'rarm': 2,
        'lshoes': 3,
        'rshoes': 4,
        'shirt': 5,
        'pant': 6,
        'lleg': 7,
        'rleg': 8
    }
    faces_segm = np.load('../data/mesh/seg_files/faces_segm_lr.npy')

    cur_verts, _, cur_faces = readObj(obj_path)
    cur_faces -= 1
    # init edge length target
    edge_target = init_edge_gt(cur_verts, cur_faces)
    edge_target = np.array(edge_target)
    edge_target = torch.from_numpy(edge_target).float().to(device)
    edge_target = edge_target.unsqueeze(1).expand(3,1,-1)
    # convert to batch tensor
    verts_tensor = torch.from_numpy(cur_verts).unsqueeze(0).float().to(device)
    face_tensor = torch.from_numpy(cur_faces).unsqueeze(0).long().to(device)
    face_segm_tensor = torch.from_numpy(faces_segm).long().to(device)
    iterative_solve_collision(verts_tensor, face_tensor, face_segm_tensor, edge_target, device, step_size=0.5)

    verts_np = verts_tensor[0].detach().cpu().numpy()
    writeObj(verts_np, obj_path, obj_path.replace('.obj','_modifed.obj'))
    os.rename(obj_path, obj_path.replace('.obj', '_origin.obj'))
    os.rename(obj_path.replace('.obj','_modifed.obj'), obj_path)



if __name__ == '__main__':
    main()
