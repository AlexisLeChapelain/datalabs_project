import open3d as o3d
from open3d.pipelines.registration import RegistrationResult

from tests.util import get_data_path
from pcd_register.registration_pipeline import registration_pipeline, preprocess_point_cloud


def test_preprocess_point_cloud():

    voxel_size = 0.05
    pcd_path, _ = get_data_path()
    pcd = o3d.io.read_point_cloud(pcd_path)

    pcd_processed, feature = preprocess_point_cloud(pcd, voxel_size)

    assert type(pcd_processed) is o3d.geometry.PointCloud
    assert type(feature) is  o3d.pipelines.registration.Feature


def test_registration_pipeline():

    voxel_size = 0.05  # means 5cm for this dataset
    source_path, target_path = get_data_path()
    result_ransac, result_fine_registration = registration_pipeline(source_path, target_path, voxel_size)

    assert type(result_ransac) is RegistrationResult
    assert type(result_fine_registration) is RegistrationResult
