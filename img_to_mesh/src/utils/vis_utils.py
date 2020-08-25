import numpy as np
from glob import glob
import math
import os.path as osp
import cv2
import imageio
import torch
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.pose_inference import get_max_preds


def vis_skeleton(img, kp_preds, parent_ids, right_joint_list):
    left_color = (191,255,77)
    right_color = (0,77,255)
    p_color = (0, 255, 255)
    # Draw keypoints
    for n in range(kp_preds.shape[0]):
        cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
        cv2.circle(img, (cor_x, cor_y), 2, p_color, -1)
    # Draw limbs
    for son, father in enumerate(parent_ids):
        if son==0:
            continue
        start_xy = (int(kp_preds[father,0]),int(kp_preds[father,1]))
        end_xy = (int(kp_preds[son,0]),int(kp_preds[son,1]))
        if son in right_joint_list:
            line_color = right_color
        else:
            line_color = left_color
        cv2.line(img, start_xy, end_xy, line_color, 2)

    return img


def plot_3d_skeleton_overlay(pred_3d, gt_3d, parent_ids, ax,
            c0='r',c1='b',c2='g',c3='y',c4='k', 
            right_joint_list = [1,2,3,13,14,15,16,17,18,19,20,21]):    
    ''' subplot '''
    # ax.set_aspect('equal')
    # pred 3d joints
    X = pred_3d[:, 0]
    Y = pred_3d[:, 1]
    Z = pred_3d[:, 2]
    for i in range(1, pred_3d.shape[0]):
        # ax.scatter(X[i], Y[i], Z[i], c=c0, marker='.')
        x = np.array([X[i], X[parent_ids[i]]], dtype=np.float32)
        y = np.array([Y[i], Y[parent_ids[i]]], dtype=np.float32)
        z = np.array([Z[i], Z[parent_ids[i]]], dtype=np.float32)
        if i in right_joint_list:
            c = c2 # 'g'
        else:
            c = c1 # 'b'
        ax.plot(x, y, z, c=c)
    
    # target 3d joints
    X = gt_3d[:, 0]
    Y = gt_3d[:, 1]
    Z = gt_3d[:, 2]
    for i in range(1, gt_3d.shape[0]):
        # ax.scatter(X[i], Y[i], Z[i], c=c0, marker='.')
        x = np.array([X[i], X[parent_ids[i]]], dtype=np.float32)
        y = np.array([Y[i], Y[parent_ids[i]]], dtype=np.float32)
        z = np.array([Z[i], Z[parent_ids[i]]], dtype=np.float32)
        if i in right_joint_list:
            c = c4 # 'k'
        else:
            c = c3 # 'y'
        ax.plot(x, y, z, c=c)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_3d_skeleton(joints_3d, parent_ids, ax, 
                    c0='r', c1='b',c2='g', 
                    right_joint_list = [1,2,3,13,14,15,16,17,18,19,20,21]):    
    ''' subplot '''
    # ax.set_aspect('equal')
    # joints
    X = joints_3d[:, 0]
    Y = joints_3d[:, 1]
    Z = joints_3d[:, 2]
    for i in range(1, joints_3d.shape[0]):
        # ax.scatter(X[i], Y[i], Z[i], c=c0, marker='.')
        x = np.array([X[i], X[parent_ids[i]]], dtype=np.float32)
        y = np.array([Y[i], Y[parent_ids[i]]], dtype=np.float32)
        z = np.array([Z[i], Z[parent_ids[i]]], dtype=np.float32)
        if i in right_joint_list:
            c = c2 # 'g'
        else:
            c = c1 # 'b'
        ax.plot(x, y, z, c=c)


    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')



def plot_3d_skeleton_demo(joints_3d, img, parent_ids, title, right_joint_list=[1,2,3,13,14,15,16,17,18,19,20,21]):
    fig = plt.figure()
    # 2k coordinates to matplotlib coordinates
    trans_mat = np.array([[1.,0,0],[0,0,-1.],[0,1,0]])
    joints_3d = np.dot(trans_mat,joints_3d.T).T
    
    ''' plot Input '''
    ax = fig.add_subplot(131)
    ax.set_title('Input')
    imgplot = plt.imshow(img)
    plt.axis('off')

    ''' plot Pred view 1'''
    ax = fig.add_subplot(132, projection='3d')
    ax.view_init(elev=20, azim=0)
    ax.set_title('Pred view 1')
    plot_3d_skeleton(joints_3d, parent_ids, ax, right_joint_list=right_joint_list)

    ''' plot Pred view 2'''
    ax = fig.add_subplot(133, projection='3d')
    ax.set_title('Pred view 2')
    plot_3d_skeleton(joints_3d, parent_ids, ax, right_joint_list=right_joint_list)
    
    plt.savefig(title)
    plt.close(fig)
    # return fig


def plot_3d_skeleton_train(pred_3d, gt_3d, img, parent_ids, title, 
                pred_name='Pred', gt_name='GT', ball=False, right_joint_list=[1,2,3,13,14,15,16,17,18,19,20,21]):
    fig = plt.figure()
    # 2k coordinates to matplotlib coordinates
    trans_mat = np.array([[1.,0,0],[0,0,-1.],[0,1,0]])
    pred_3d = np.dot(trans_mat,pred_3d.T).T
    gt_3d = np.dot(trans_mat,gt_3d.T).T
    
    ''' plot Input '''
    ax = fig.add_subplot(241)
    ax.set_title('Input')
    imgplot = plt.imshow(img)
    plt.axis('off')

    ''' plot Pred view 1'''
    ax = fig.add_subplot(242, projection='3d')
    ax.view_init(elev=20, azim=0)
    ax.set_title('{} view 1'.format(pred_name))
    if ball:
        plot_3d_skeleton(pred_3d[:-1], parent_ids, ax, right_joint_list=right_joint_list)
        ax.scatter(pred_3d[-1,0], pred_3d[-1,1], pred_3d[-1,2], c='r', marker='.')
    else:
        plot_3d_skeleton(pred_3d, parent_ids, ax, right_joint_list=right_joint_list)
 
    ''' plot GT view 1'''
    ax = fig.add_subplot(243, projection='3d')
    ax.view_init(elev=20, azim=0)
    ax.set_title('{} view 1'.format(gt_name))
    if ball:
        plot_3d_skeleton(gt_3d[:-1], parent_ids, ax, right_joint_list=right_joint_list)
        ax.scatter(gt_3d[-1,0], gt_3d[-1,1], gt_3d[-1,2], c='c', marker='.')
    else:
        plot_3d_skeleton(gt_3d, parent_ids, ax, c1='y',c2='k', right_joint_list=right_joint_list)
    

    ''' plot Overlay view 1'''
    ax = fig.add_subplot(244, projection='3d')
    ax.view_init(elev=20, azim=0)
    ax.set_title('Overlay view 1')
    if ball:
        plot_3d_skeleton_overlay(pred_3d[:-1], gt_3d[:-1], parent_ids, ax, right_joint_list=right_joint_list)
        ax.scatter(pred_3d[-1,0], pred_3d[-1,1], pred_3d[-1,2], c='r', marker='.')
        ax.scatter(gt_3d[-1,0], gt_3d[-1,1], gt_3d[-1,2], c='c', marker='.')
    else:
        plot_3d_skeleton_overlay(pred_3d, gt_3d, parent_ids, ax, right_joint_list=right_joint_list)

    ''' plot Pred view 2'''
    ax = fig.add_subplot(246, projection='3d')
    ax.set_title('{} view 2'.format(pred_name))
    if ball:
        plot_3d_skeleton(pred_3d[:-1], parent_ids, ax, right_joint_list=right_joint_list)
        ax.scatter(pred_3d[-1,0], pred_3d[-1,1], pred_3d[-1,2], c='r', marker='.')
    else:
        plot_3d_skeleton(pred_3d, parent_ids, ax, right_joint_list=right_joint_list)

    ''' plot GT view 2'''
    ax = fig.add_subplot(247, projection='3d')
    ax.set_title('{} view 2'.format(gt_name))
    if ball:
        plot_3d_skeleton(gt_3d[:-1], parent_ids, ax, right_joint_list=right_joint_list)
        ax.scatter(gt_3d[-1,0], gt_3d[-1,1], gt_3d[-1,2], c='c', marker='.')
    else:
        plot_3d_skeleton(gt_3d, parent_ids, ax, c1='y',c2='k', right_joint_list=right_joint_list)

    ''' plot Overlay view 2'''
    ax = fig.add_subplot(248, projection='3d')
    ax.set_title('Overlay view 2')
    if ball:
        plot_3d_skeleton_overlay(pred_3d[:-1], gt_3d[:-1], parent_ids, ax, right_joint_list=right_joint_list)
        ax.scatter(pred_3d[-1,0], pred_3d[-1,1], pred_3d[-1,2], c='r', marker='.')
        ax.scatter(gt_3d[-1,0], gt_3d[-1,1], gt_3d[-1,2], c='c', marker='.')
    else:
        plot_3d_skeleton_overlay(pred_3d, gt_3d, parent_ids, ax, right_joint_list=right_joint_list)
    
    plt.savefig(title, dpi=300)
    return fig
