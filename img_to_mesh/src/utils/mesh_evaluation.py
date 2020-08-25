import sys
import os
import os.path as osp
from glob import glob
import numpy as np
import open3d as o3d
import copy

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    # distance_threshold = voxel_size * 0.5
    distance_threshold = voxel_size * 1.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
    #         % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPoint())
    return result


def perform_registration(pred_verts, gt_verts, voxel_size=0.05):
    # perform registration
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pred_verts)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(gt_verts)
    # downsample
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    # global RANSAC
    # result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    result_ransac = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    # local ICP
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, result_ransac, voxel_size)
    source_tmp = copy.deepcopy(source)
    source_tmp.transform(result_icp.transformation)
    pred_verts_align = np.asarray(source_tmp.points)
    return pred_verts_align
