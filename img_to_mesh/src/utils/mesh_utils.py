import os
import os.path as osp
import numpy as np
import shutil
import pickle
import logging
import torch
from sklearn.metrics.pairwise import euclidean_distances
import re
import math

try:
    import psbody.mesh
    found = True
except ImportError:
    found = False
if found:
    from psbody.mesh import Mesh

import trimesh
from utils.shape_data import ShapeData
from utils.mesh_sampling import generate_transform_matrices
from utils.spiral_utils import get_adj_trigs, generate_spirals
import constants

def process_mesh(opt, body_part, device):
    ref_mesh = constants.REF_MESH[body_part]
    ds_factors = constants.DS_FACTORS[body_part]
    reference_points = constants.REFERENCE_POINTS[body_part]
    step_sizes = constants.STEP_SIZES[body_part]
    dilation = constants.DILATION_VAL[body_part]
    shapedata =  ShapeData(
                        train_file=osp.join(opt.processed_verts_dir, body_part, 'preprocessed/train_verts.npy'), 
                        val_file=osp.join(opt.processed_verts_dir, body_part, 'preprocessed/val_verts.npy'), 
                        test_file=osp.join(opt.processed_verts_dir, body_part, 'preprocessed/test_verts.npy'), 
                        reference_mesh_file=ref_mesh,
                        normalization=opt.dataset.normalization,
                        meshpackage=opt.dataset.meshpackage, load_flag = True)

    pkl_path = osp.join(constants.MESH_DATA_DIR, body_part,'{}.pkl'.format(ds_factors))
    print(pkl_path)
    if not osp.exists(pkl_path):
        if shapedata.meshpackage == 'trimesh':
            raise NotImplementedError('Rerun with mpi-mesh as meshpackage')
        
        print("Generating Transform Matrices ..")
        if opt.dataset.downsample_method == 'COMA_downsample':
            M,A,D,U,F = generate_transform_matrices(shapedata.reference_mesh, ds_factors)
            with open(pkl_path, 'wb') as fp:
                M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
                pickle.dump({'M_verts_faces':M_verts_faces,'A':A,'D':D,'U':U,'F':F}, fp)
        else:
            raise NotImplementedError('Rerun with COMA_downsample')

    else:
        print("Loading Transform Matrices ..")
        with open(pkl_path, 'rb') as fp:
            downsampling_matrices = pickle.load(fp, encoding='latin1') # for python3, need to add encoding
                
        M_verts_faces = downsampling_matrices['M_verts_faces']
        if shapedata.meshpackage == 'mpi-mesh':
            M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
        elif shapedata.meshpackage == 'trimesh':
            M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process = False) for i in range(len(M_verts_faces))]
        A = downsampling_matrices['A']
        D = downsampling_matrices['D']
        U = downsampling_matrices['U']
        F = downsampling_matrices['F']
    
    print("Calculating reference points for downsampled versions ..")
    for i in range(len(ds_factors)):
        if shapedata.meshpackage == 'mpi-mesh':
            dist = euclidean_distances(M[i+1].v, M[0].v[reference_points[0]])
        elif shapedata.meshpackage == 'trimesh':
            dist = euclidean_distances(M[i+1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist,axis=0).tolist())

    if shapedata.meshpackage == 'mpi-mesh':
        vnum = [x.v.shape[0] for x in M]
    elif shapedata.meshpackage == 'trimesh':
        vnum = [x.vertices.shape[0] for x in M]
    Adj, Trigs = get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage = shapedata.meshpackage)
    
    print("Generating Spirals ..")
    spirals_np, spiral_sizes, spirals = generate_spirals(step_sizes, 
                                                        M, Adj, Trigs, 
                                                        reference_points = reference_points, 
                                                        dilation = dilation, random = False, 
                                                        meshpackage = shapedata.meshpackage, 
                                                        counter_clockwise = True)
    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1,D[i].shape[0]+1,D[i].shape[1]+1))
        u = np.zeros((1,U[i].shape[0]+1,U[i].shape[1]+1))
        d[0,:-1,:-1] = D[i].todense()
        u[0,:-1,:-1] = U[i].todense()
        d[0,-1,-1] = 1
        u[0,-1,-1] = 1
        bD.append(d)
        bU.append(u)


    tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]
    tD = [torch.from_numpy(s).float().to(device) for s in bD]
    tU = [torch.from_numpy(s).float().to(device) for s in bU]
    return shapedata, tspirals, spiral_sizes, vnum, tD, tU


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

def readObjV2(file_path):
    obj_file = open(file_path,'r')
    contents = obj_file.readlines()
    v,vt,v_idx,vt_idx = [],[],[],[]
    for line in contents:
        parts = line.split(' ')
        if parts[0] == 'v':
            vertex = [float(i) for i in parts[1:4]]
            v.append(vertex)
        if parts[0] == 'vt':
            texcoord = [float(i) for i in parts[1:3]]
            vt.append(texcoord)
        if parts[0] == 'f':
            vertex_index = [int(i.split('/')[0]) for i in parts[1:4]]
            v_idx.append(vertex_index)
            texcoord_index = [int(i.split('/')[1]) for i in parts[1:4]]
            vt_idx.append(texcoord_index)
    v = np.array(v)
    vt = np.array(vt)
    v_idx = np.array(v_idx)
    vt_idx = np.array(vt_idx)
    return v,vt,v_idx,vt_idx

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

def R_axis_angle(axis, angle):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]

    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """
    # Trig factors.
    ca = math.cos(angle)
    sa = math.sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    matrix = np.zeros((3,3))
    # Update the rotation matrix.
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca
    return matrix
