import os
import os.path as osp
from glob import glob
import pickle
import math
import numpy as np
import cv2
import img_io as io_utils
import camera as cam_utils
import draw as draw_utils
import transform as transf_utils
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def _fun_distance_transform(params_, dist_map_, points3d, cam_pos):
    theta_x_, theta_y_, theta_z_, fx_= params_
    h_, w_ = dist_map_.shape[0:2]
    n_ = points3d.shape[0]

    cx_, cy_ = float(dist_map_.shape[1])/2.0, float(dist_map_.shape[0])/2.0

    R_ = transf_utils.Rz(theta_z_).dot(transf_utils.Ry(theta_y_)).dot(transf_utils.Rx(theta_x_))
    A_ = np.eye(3, 3)
    A_[0, 0], A_[1, 1], A_[0, 2], A_[1, 2] = fx_, fx_, cx_, cy_

    T_ = -np.dot(R_,cam_pos)

    p2_ = A_.dot(R_.dot(points3d.T) + np.tile(T_, (1, n_)))
    p2_ /= p2_[2, :]
    p2_ = p2_.T[:, 0:2]
    p2_ = np.round(p2_).astype(int)
    _, valid_id_ = cam_utils.inside_frame(p2_, h_, w_)

    residual = np.zeros((n_,)) + 0.0
    residual[valid_id_] = dist_map_[p2_[valid_id_, 1], p2_[valid_id_, 0]]
    return np.sum(residual)


def _calibrate_camera_dist_transf(A, R, T, dist_transf, points3d):

    theta_x, theta_y, theta_z = transf_utils.get_angle_from_rotation(R)
    fx, fy, cx, cy = A[0, 0], A[1, 1], A[0, 2], A[1, 2]
    cam_pos = -np.dot(R.T, T)
    # print(theta_x, theta_y, theta_z, fx, cam_pos)

    params = np.hstack((theta_x, theta_y, theta_z, fx))

    res_ = minimize(_fun_distance_transform, params, args=(dist_transf, points3d, cam_pos),
                    method='Powell', options={'disp': False, 'maxiter': 10000})
    result = res_.x

    theta_x_, theta_y_, theta_z_, fx_ = result

    cx_, cy_ = float(dist_transf.shape[1]) / 2.0, float(dist_transf.shape[0]) / 2.0

    R__ = transf_utils.Rz(theta_z_).dot(transf_utils.Ry(theta_y_)).dot(transf_utils.Rx(theta_x_))
    T__ = -np.dot(R__, cam_pos)
    A__ = np.eye(3, 3)
    A__[0, 0], A__[1, 1], A__[0, 2], A__[1, 2] = fx_, fx_, cx_, cy_

    return A__, R__, T__

def _set_correspondences(img, field_img_path='field.png'):

    field_img = cv2.imread(field_img_path)

    h2, w2 = field_img.shape[0:2]
    W, H = 28.65 , 15.24

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img[:,:,::-1])
    ax[1].imshow(field_img)

    ax[0].axis('off')
    ax[1].axis('off')

    points2d = []
    points3d = []

    def onclick(event):
        x, y = event.xdata, event.ydata
        if event.inaxes.axes.get_position().x0 < 0.5:
            ax[0].plot(x, y,  'r.', markersize=10)
            points2d.append([x, y])
        else:
            ax[1].plot(x, y,  'b+', markersize=20)
            points3d.append([y, 0, x])
        plt.show()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    points2d = np.array(points2d)
    points3d = np.array(points3d)

    points3d[:, 0] = -((points3d[:, 0] - h2 / 2.) / h2) * H
    points3d[:, 2] = ((points3d[:, 2] - w2 / 2.) / w2) * W

    return points2d, points3d

def _get_pixel_coords(img):

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img[:,:,::-1])

    ax.axis('off')

    points2d = []

    def onclick(event):
        x, y = event.xdata, event.ydata
        ax.plot(x, y,  color='red', marker='o', markersize=12)
        points2d.append([x, y])
        plt.show()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    points2d = np.array(points2d)
    
    return points2d


def calibrate_by_click(img, edges, field_img_path='field.png'):

    h, w = img.shape[0:2]

    points2d, points3d = _set_correspondences(img, field_img_path=field_img_path)
    print(points3d)

    # ------------------------------------------------------------------------------------------------------------------
    # OpenCV initial calibration
    fx, fy = cam_utils.grid_search_focal_length(points3d, points2d, h, w, same_f=True)
    A = cam_utils.intrinsic_matrix_from_focal_length(fx, fy, h, w)

    points_3d_cv = points3d[:, np.newaxis, :].astype(np.float32)
    points_2d_cv = points2d[:, np.newaxis, :].astype(np.float32)

    _, rvec, tvec, _ = cv2.solvePnPRansac(points_3d_cv, points_2d_cv, A, None)
    rvec, tvec = np.squeeze(rvec), np.squeeze(tvec)
    R, _ = cv2.Rodrigues(rvec)
    T = np.array([tvec]).T
    cam_pnp = cam_utils.Camera('tmp', A, R, T, h, w)

    # ------------------------------------------------------------------------------------------------------------------
    # Photometric refinement
    A__, R__, T__, field3d = calibrate_from_initialization(img, edges, A, R, T)
    cam_opt = cam_utils.Camera('tmp', A__, R__, T__, h, w)

    # Sanity check, project the basketball field points
    p2_pnp, _ = cam_pnp.project(field3d)
    p2_pnp, _ = cam_utils.inside_frame(p2_pnp, cam_pnp.height, cam_pnp.width)

    p2_opt, _ = cam_opt.project(field3d)
    p2_opt, valid_id = cam_utils.inside_frame(p2_opt, cam_opt.height, cam_opt.width)

    A_out, R_out, T_out = [None], [None], [None]

    class Index(object):

        def save_pnp(self, event):
            A_out.append(cam_pnp.A)
            R_out.append(cam_pnp.R)
            T_out.append(cam_pnp.T)
            plt.close()

        def save_opt(self, event):
            A_out.append(cam_opt.A)
            R_out.append(cam_opt.R)
            T_out.append(cam_opt.T)
            plt.close()

        def discard(self, event):
            plt.close()

    fig, ax = plt.subplots(1, 2)
    io_utils.imshow(img[:,:,::-1], ax=ax[0], points=p2_pnp)
    io_utils.imshow(img[:,:,::-1], ax=ax[1], points=p2_opt)
    callback = Index()
    axdisc = plt.axes([0.6, 0.05, 0.1, 0.075])
    axpnp = plt.axes([0.7, 0.05, 0.1, 0.075])
    axopt = plt.axes([0.8, 0.05, 0.1, 0.075])
    bdisc = Button(axdisc, 'Discard')
    bdisc.on_clicked(callback.discard)
    bpnp = Button(axpnp, 'Save pnp')
    bpnp.on_clicked(callback.save_pnp)
    bopt = Button(axopt, 'Save opt')
    bopt.on_clicked(callback.save_opt)
    plt.show()

    return A_out[-1], R_out[-1], T_out[-1]

def calibrate_from_initialization(img, edges, A_init, R_init, T_init, visualize=False):

    h, w = img.shape[:2]
    dist_transf = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 0)

    cam_init = cam_utils.Camera('tmp', A_init, R_init, T_init, h, w)
    template, field_mask = draw_utils.draw_field(cam_init)

    II, JJ = (template > 0).nonzero()
    synth_field2d = np.array([[JJ, II]]).T[:, :, 0]

    field3d = cam_utils.plane_points_to_3d(synth_field2d, cam_init)

    A, R, T = _calibrate_camera_dist_transf(A_init, R_init, T_init, dist_transf, field3d)

    if visualize:
        cam_res = cam_utils.Camera('tmp', A, R, T, h, w)
        field2d, __ = cam_res.project(field3d)
        class Index(object):
            def save_pnp(self, event):
                plt.close()
                A__, R__, T__ = calibrate_by_click(img, edges, field_img_path='field.png')
                if not (A__ is None and R__ is None and T__ is None):
                    A = A__
                    R = R__
                    T = T__

            def save_opt(self, event):
                plt.close()
        
        fig, ax = plt.subplots(1, 1)
        io_utils.imshow(img[:,:,::-1], ax=ax, points=field2d)
        callback = Index()
        axpnp = plt.axes([0.7, 0.05, 0.1, 0.075])
        bpnp = Button(axpnp, 'Save pnp')
        bpnp.on_clicked(callback.save_pnp)
        axopt = plt.axes([0.8, 0.05, 0.1, 0.075])
        bopt = Button(axopt, 'Save opt')
        bopt.on_clicked(callback.save_opt)
        plt.show()

    return A, R, T, field3d

def calibrate_frame(img_path, edges_path, write_dir):
    os.makedirs(write_dir, exist_ok=True)
    img = cv2.imread(img_path)
    edges = cv2.imread(edges_path,0).astype(np.float32) / 255.

    A, R, T = calibrate_by_click(img, edges, field_img_path='field.png')

    img_name = img_path.split('/')[-1].replace('.png','')
    pickle.dump(A, open('{}/{}_A.npy'.format(write_dir, img_name), 'wb'))
    pickle.dump(R, open('{}/{}_R.npy'.format(write_dir, img_name), 'wb'))
    pickle.dump(T, open('{}/{}_T.npy'.format(write_dir, img_name), 'wb'))

def main():
    subdir = 'lbj_dunk'
    img_path = glob('../../data/{}/images/*.png'.format(subdir))[0]
    img_path = img_path.replace('\\', '/')
    img_name  = img_path.split('/')[-1].strip('.png')
    edges_path = '../../results/{}/field_lines/{}.png'.format(subdir, img_name)
    print(img_path, edges_path)
    write_dir = '../../results/{}/cams'.format(subdir)
    calibrate_frame(img_path, edges_path, write_dir)

if __name__ == '__main__':
    main()