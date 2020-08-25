import numpy as np
import os.path as osp

''' IMG_MEAN, IMG_NORM '''
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_NORM = [0.229, 0.224, 0.225]

''' JOINTS INFO '''
JOINTS_NPARTS = {
    'aug': 35,
}

JOINTS_FLIP_PAIRS = {
    'aug': [[1,4],[2,5],[3,6],[13,10],[14,11],[15,12],
            [16,25],[17,26],[18,27],[19,28],[20,29],[21,30],
            [22,31],[23,32],[24,33]],
}

JOINTS_PARENT_IDS = {
    'aug': np.array([-1,0,1,2,0,4,5,0,7,8,7,10,11,7,13,14,
            15,15,15,15,15,3,3,34,23,12,12,12,12,12,6,6,34,32,8]),
}

JOINTS_INDEX = {
    'aug':np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]),
}

RIGHT_JOINTS = {
    'aug': [1,2,3,13,14,15,16,17,18,19,20,21,22,23,24],
}

FOOT_JOINTS = {
    # rtoe, rheel, ltoe, lheel
    'aug': [21,22,30,31]
}

LSP_INDEX = np.array([3,2,1,4,5,6,15,14,13,10,11,12,8,9])

''' JUMP CLASSES '''
NCLASSES = {
    'jump': 2,
}

BONE_LENGTH = np.array([0.10117956, 0.5606053, 0.42826986, 0.10086109, 0.5608561, 0.42826986, 
    0.48895445, 0.13046326, 0.14554071, 0.24569063, 0.30577067, 0.27409163, 0.24526644, 
    0.30515963, 0.27403432, 0.1345959, 0.16684785, 0.16965915, 0.1619055, 0.14833298, 
    0.17824635, 0.13662696, 0.0710501, 0.08761071, 0.13424419, 0.16657521, 0.1696622, 
    0.1621918, 0.14895742, 0.1782432, 0.13671178, 0.07097481, 0.08761071, 0.2093108])

'''Mesh TLNet setting '''
MESH_DATA_DIR = '../data/mesh'
J3D_REGRESSOR_PATH=osp.join(MESH_DATA_DIR, 'j3d_regressor.npy')

BODY_PART_INDEX = {
    'head': np.arange(0,348), 
    'arm': np.arange(348,1190),
    'shoes': np.arange(1190,2075),
    'shirt': np.arange(2075,4175),
    'pant': np.arange(4175,4343),
    'leg': np.arange(4343,4715)
}

REF_MESH = {
    'head': osp.join(MESH_DATA_DIR, 'head/1283_head.obj'),
    'arm': osp.join(MESH_DATA_DIR, 'arm/1263_arm.obj'),
    'shoes': osp.join(MESH_DATA_DIR, 'shoes/0_shoes.obj'),
    'shirt': osp.join(MESH_DATA_DIR, 'shirt/1206_shirt.obj'),
    'pant': osp.join(MESH_DATA_DIR, 'pant/1222_pant.obj'),
    'leg': osp.join(MESH_DATA_DIR, 'leg/1272_leg.obj'),
    'human': osp.join(MESH_DATA_DIR, 'human/0_person_simple.obj')
}

NVERTS = {
    'head': 348, 
    'arm': 842,
    'shoes': 885,
    'shirt': 2100,
    'pant': 168,
    'leg': 372,
    'human': 4715,
}

# downsample factors for Spiral Conv
DS_FACTORS = {
    'head': [2,2,1,1], 
    'shoes': [2,2,2,1],
    'arm': [2,2,2,1],
    'shirt': [4,2,2,2],
    'pant': [2,1,1,1],
    'leg': [2,2,1,1]
}

# reference points for Spiral Conv
REFERENCE_POINTS = {
    'head': [[50]], 
    'arm': [[144,565]],
    'shoes': [[33,550]],
    'shirt': [[1159]],
    'pant': [[66]],
    'leg': [[91,259]]
}

# encoder channels for Spiral Conv
FILTER_SIZES_ENC = {
    'head': [[3,16,32,64,64],[[],[],[],[],[]]], 
    'arm': [[3,16,32,64,64],[[],[],[],[],[]]],
    'shoes': [[3,16,32,64,64],[[],[],[],[],[]]],
    'shirt': [[3,16,32,64,64],[[],[],[],[],[]]],
    'pant': [[3,16,32,64,64],[[],[],[],[],[]]],
    'leg': [[3,16,32,64,64],[[],[],[],[],[]]]
}

# decoder channels for Spiral Conv
FILTER_SIZES_DEC = {
    'head': [[64,64,32,16,16],[[],[],[],[],3]], 
    'arm': [[64,64,32,16,16],[[],[],[],[],3]],
    'shoes': [[64,64,32,16,16],[[],[],[],[],3]],
    'shirt': [[64,64,32,16,16],[[],[],[],[],3]],
    'pant': [[64,64,32,16,16],[[],[],[],[],3]],
    'leg': [[64,64,32,16,16],[[],[],[],[],3]]

}

# hops for Spiral Conv
STEP_SIZES = {
    'head': [2,2,1,1,1], 
    'arm': [2,2,1,1,1],
    'shoes': [2,2,1,1,1],
    'shirt': [2,2,1,1,1],
    'pant': [2,2,1,1,1],
    'leg': [2,2,1,1,1]
}

# dilation ratio for Spiral Conv
DILATION_VAL = {
    'head': [2,2,1,1,1], 
    'arm': [2,2,1,1,1],
    'shoes': [2,2,1,1,1],
    'shirt': [2,2,1,1,1],
    'pant': [2,2,1,1,1],
    'leg': [2,2,1,1,1]
}

# hidden vector channel for Spiral Conv
NZ = {
    'head': 32, 
    'shoes': 32,
    'arm': 32,
    'shirt': 32,
    'pant': 32,
    'leg': 32
}

''' Dataset Split '''
TRAIN_PLAYERS = [
    'alfred','chad', 'donell', 'erik', 'guy','jamaal','juwan',
    'kedrick','martin','nick','randall','zach','zack', 'lamond', 
    'cedric', 'lucas', 'barney','allen', 'darrell', 'bradley'
]

VALID_PLAYERS = [
    'devin', 'dion', 'brendan', 'leo'
]

TEST_PLAYERS = [
   'cory', 'glen', 'tomas', 'oscar'
]