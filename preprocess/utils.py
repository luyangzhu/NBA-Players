import os.path as osp
import numpy as np
import cv2

def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor

def scaleCrop(image, keypoints, scale, center, img_size=256):
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = np.array([scale_factors[1], scale_factors[0]])
    center_scaled = np.round(center * scale_factors).astype(np.int)
    keypoints_scaled = keypoints * scale_factors.reshape(1,-1)

    margin = int(img_size / 2)
    center_pad = center_scaled + margin
    keypoints_pad = keypoints_scaled + margin
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    keypoints_crop = keypoints_pad - start_pt
    if len(image.shape) == 3: # rgb
        image_pad = np.pad(image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
        img_crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    else: # mask
        image_pad = np.pad(image_scaled, ((margin, ), (margin, )), mode='edge')
        img_crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }
    return img_crop, keypoints_crop, proc_param
