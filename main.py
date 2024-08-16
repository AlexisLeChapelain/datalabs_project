import open3d as o3d


from pcd_register.data_processing import data_processing_pipeline
from pcd_register.registration_pipeline import registration_pipeline


if __name__ == "__main__":

    # load data path for two point clouds
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source_path = demo_icp_pcds.paths[0]
    target_path = demo_icp_pcds.paths[1]

    # align point clouds using two different methods
    voxel_size = 0.05  # means 5cm for this dataset
    result_ransac, result_fine_registration = registration_pipeline(source_path, target_path, voxel_size)

    # do some data processing on a point cloud
    point_index = 100
    radius = 0.01
    cov_matrix, sigma_square, normal_vector, pcd = data_processing_pipeline(source_path, point_index, radius)

    print("The code successfully completed.")
