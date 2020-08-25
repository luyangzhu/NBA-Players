import numpy as np
import os
import os.path as osp
import cv2
import math

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
