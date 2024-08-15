import copy
import open3d as o3d
import numpy as np
from typing import Tuple


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float
                           ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """
    Preprocess a point cloud by downsampling it, computing the normals, and then computing
    Fast Point Feature Histograms.

    :param pcd: the point cloud to preprocess
    :param voxel_size: the voxel size to use to downsample the point cloud, compute its normals and its features.
    :return: the preprocessed point cloud and the Fast Point Feature Histograms
    """

    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


def registration_pipeline(source_path: str, target_path: str, voxel_size: float
                          ) -> Tuple[o3d.pipelines.registration.RegistrationResult,
                                     o3d.pipelines.registration.RegistrationResult]:
    """
    Pipeline to preprocess two point cloud and align the source point cloud with the target point cloud.
    Use two methodes, RANSAC and fine registration, and return both results.

    :param source_path: path to the source point cloud file.
    :param target_path: path to the target point cloud file.
    :param voxel_size: voxel size used for various parameters within the pipeline.
    :return: return two transformation to align the source point cloud and the target point cloud, one
             based on RANSAC and the other on fine registration.
    """

    # Load point clouds
    source: o3d.geometry.PointCloud = o3d.io.read_point_cloud(source_path)
    target: o3d.geometry.PointCloud = o3d.io.read_point_cloud(target_path)

    # Preprocess
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Rough registration (with RANSAC)
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    # Fine registration (with ICP)
    distance_threshold = voxel_size * 0.4
    result_fine_registration = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return result_ransac, result_fine_registration


if __name__ == '__main__':
    voxel_size = 0.05  # means 5cm for this dataset
    source_path = "/Users/alexislechapelain/open3d_data/extract/DemoICPPointClouds/cloud_bin_0.pcd"
    target_path = "/Users/alexislechapelain/open3d_data/extract/DemoICPPointClouds/cloud_bin_1.pcd"
    result_ransac, result_fine_registration = registration_pipeline(source_path, target_path, voxel_size)

