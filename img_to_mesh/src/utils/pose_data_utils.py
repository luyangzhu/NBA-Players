# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# modified by Luyang
# ------------------------------------------------------------------------------

import os
import numpy as np
import cv2


def generate_heatmaps(joints, nParts, rgb_res, out_res, sigma, heatmaps_type='gaussian'):
    '''
    :param joints:  [num_joints, 3]
    :param nParts:  joints number
    :param rgb_res:  rgb img size, (2,)
    :param out_res:  heatmap size, (2,)
    :param sigma:  sigma of gaussian
    :return: heatmaps [num_joints,out_res[0],out_res[1]]
    '''

    assert heatmaps_type == 'gaussian', \
        'Only support gaussian map now!'

    if heatmaps_type == 'gaussian':
        heatmaps = np.zeros((nParts,out_res[0],out_res[1]),
                            dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(nParts):
            feat_stride = rgb_res / out_res
            mu_x = int(joints[joint_id][0] / feat_stride[1] + 0.5) # feat_stride[1] width -> x
            mu_y = int(joints[joint_id][1] / feat_stride[0] + 0.5) # feat_stride[0] height -> y
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= out_res[1] or ul[1] >= out_res[0] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], out_res[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], out_res[0]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], out_res[1])
            img_y = max(0, ul[1]), min(br[1], out_res[0])

            heatmaps[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return heatmaps

def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()

    return joints


def transform_preds(coords, center, scale, output_size, input_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, input_size=np.array(input_size, dtype=np.float32), inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         input_size=np.array([256,256], dtype=np.float32),
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        # print(scale)
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale * input_size
    src_w = scale_tmp[1]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    # swap scale_tmp
    scale_tmp_xy = scale_tmp[::-1].copy()
    src[0, :] = center + scale_tmp_xy * shift
    src[1, :] = center + src_dir + scale_tmp_xy * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img
