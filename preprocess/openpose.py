import os
import os.path as osp
from glob import glob
import pickle
import json
import numpy as np
import cv2
from utils import *

def library_detect(root_folder, openpose_path, model_folder):
    img_dir = '{}/images'.format(root_folder)
    json_dir = '{}/keypoints'.format(root_folder)
    os.makedirs(json_dir, exist_ok=True)
    vis_dir = '{}/visualize'.format(root_folder)
    os.makedirs(vis_dir, exist_ok=True)
    cmd = '{} --image_dir {} --write_json {} --write_images {} '.format(
            openpose_path, img_dir, json_dir, vis_dir) + \
            '--display 0 --model_folder {} --model_pose COCO'.format(model_folder)
    print(cmd)
    os.system(cmd)

def center_crop(root_folder):
    os.makedirs('{}/img_crop'.format(root_folder), exist_ok=True)
    os.makedirs('{}/kp_crop'.format(root_folder), exist_ok=True)
    os.makedirs('{}/proc_param'.format(root_folder), exist_ok=True)

    img_paths = glob('{}/images/*.jpg'.format(root_folder))
    if len(img_paths) != 0:
        for img_path in img_paths:
            img = cv2.imread(img_path)
            cv2.imwrite(img_path.replace('jpg','png'), img)

    json_paths = sorted(glob('{}/keypoints/*.json'.format(root_folder)))
    # windows specific
    json_paths = [i.replace('\\', '/') for i in json_paths]
    for json_path in json_paths:
        frame_name = json_path.split('/')[-1].replace('_keypoints.json','')
        img_path = '{}/images/{}.png'.format(root_folder, frame_name)
        print(img_path)
        frame_img = cv2.imread(img_path)
        with open(json_path) as f:
            data = json.load(f)
        person_id = 0
        for idx, cur_people in enumerate(data['people']):
            j2d = np.array(cur_people["pose_keypoints_2d"]).reshape(-1,3)
            conf = j2d[:,-1]
            j2d_pos = j2d[:,0:2]
            if np.mean(conf) < 0.5:
                continue
            # get center and scale
            if conf[8] > 0.2:
                center = j2d_pos[8]
            elif conf[9] > 0.2 and conf[12] > 0.2:
                center = (j2d_pos[9] + j2d_pos[12]) / 2.
            else:
                continue # skip current person
            mask = (conf > 0.2)
            valid_j2d = j2d_pos[mask]
            min_pt = np.amin(valid_j2d,0)
            max_pt = np.amax(valid_j2d,0)
            person_height = np.linalg.norm(max_pt - min_pt)
            scale = 150. / person_height 
            # crop image and keypoints
            img_crop, j2d_pos_crop, proc_param = scaleCrop(frame_img.copy(), j2d_pos, scale, center, img_size=256)
            # save results
            cv2.imwrite('{}/img_crop/{}_{:04d}.png'.format(root_folder,frame_name,person_id), img_crop)
            pickle.dump(proc_param, open('{}/proc_param/{}_{:04d}.pkl'.format(root_folder,frame_name,person_id),'wb'))
            j2d_crop = np.concatenate((j2d_pos_crop,conf.reshape(-1,1)),1)
            np.save('{}/kp_crop/{}_{:04d}.npy'.format(root_folder,frame_name,person_id), j2d_crop)
            person_id += 1

def main():
    openpose_folder = 'C:/Users/lyzhu/Downloads/openpose-1.5.1-binaries-win64-gpu-python-flir-3d_recommended/openpose'
    openpose_path = '{}/bin/OpenPoseDemo.exe'.format(openpose_folder)
    model_folder = '{}/models/'.format(openpose_folder)
    root_folder = '../data/harden_3'
    library_detect(root_folder, openpose_path, model_folder)
    center_crop(root_folder)

if __name__ == '__main__':
    main()