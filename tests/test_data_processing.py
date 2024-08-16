import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import KDTreeFlann

from typing import List

from tests.util import get_data_path
from pcd_register.data_processing import projector, compute_covariance_at_index, compute_approximate_curvature, \
    data_processing_pipeline


def get_covariance_matrix(point_list: List[List[float]]):
    """
    Return the covariance matrix of a point cloud.
    :param point_list: list of point constituting a point cloud
    :return: the covariance matrix
    """
    test_pcd = o3d.t.geometry.PointCloud(point_list).to_legacy()
    test_pcd_tree: KDTreeFlann = o3d.geometry.KDTreeFlann(test_pcd)
    # NB : this is for test only, so we use a huge radius to include every point
    return compute_covariance_at_index(test_pcd, test_pcd_tree, 0, 100)


def test_compute_covariance_at_index():

    # Test normal case
    cov_matrix = get_covariance_matrix([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert type(cov_matrix) is np.ndarray
    assert cov_matrix.shape == (3,3)
    assert (np.isclose(cov_matrix, np.array([[0.22222222, -0.11111111, -0.11111111],
                                         [-0.11111111, 0.22222222, -0.11111111],
                                         [-0.11111111, -0.11111111, 0.22222222]]))
            ).all()

    # test degenerate case where all points are equal
    cov_matrix = get_covariance_matrix([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    assert type(cov_matrix) is np.ndarray
    assert cov_matrix.shape == (3,3)
    assert (np.isclose(cov_matrix, np.array([[0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.]]))
            ).all()


def test_compute_approximate_curvature():

    cov_matrix = get_covariance_matrix([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    sigma_square, normal_vector = compute_approximate_curvature(cov_matrix)

    assert type(sigma_square) is float
    # the smallest eigen value will be 0 here given the shape of the covariance matrix
    # Hence sigma_square will be 0.
    assert sigma_square == 0.0

    assert type(normal_vector) is np.ndarray
    assert normal_vector.shape == (3,)
    # given the shape of the point cloud, we can directly deduce an orthogonal vector
    assert np.array_equal(normal_vector, np.array([0, 1, 0]))



def test_projector():
    normal_ = np.array([0, 1, 0])
    target_ = np.array([10, 20, -5])
    point_of_origin = np.array([0, 10, 0])

    result = projector(point_of_origin, normal_, target_)

    assert type(result) is np.ndarray
    assert result.shape == (3,)
    assert np.array_equal(result, np.array([10, 10, -5]))


def test_data_processing_pipeline():

    pcd_path, _ =  get_data_path()
    point_index = 1000
    radius = 0.01

    cov_matrix, sigma_square, normal_vector, pcd = data_processing_pipeline(pcd_path, point_index, radius)

    assert type(cov_matrix) is np.ndarray
    assert cov_matrix.shape == (3,3)

    assert type(sigma_square) is float

    assert type(normal_vector) is np.ndarray
    assert normal_vector.shape == (3,)

    assert type(pcd) is o3d.geometry.PointCloud
