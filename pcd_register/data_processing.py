import open3d as o3d
import numpy as np
from typing import Tuple

from open3d.cpu.pybind.geometry import KDTreeFlann


def compute_covariance_at_index(pcd: o3d.geometry.PointCloud, pcd_tree: KDTreeFlann, index: int,
                                radius: float) -> np.ndarray:
    """
    Compute covariance matrix for a point in a point cloud at a given index

    :param pcd: point cloud.
    :param pcd_tree: point cloud tree.
    :param index: index of the point where we compute the covariance matrix.
    :param radius: length of the radius. Every point within the radius will be used to compute the covariance matrix.
    :return: a covariance matrix.
    """

    # extract point index within a radius
    [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[index], radius)

    # get coordinate for each index and transform into a numpy array
    neighbours = np.array([pcd.points[elem] for elem in idx])

    # compute centroid
    centroid = neighbours.mean(axis=0)

    # compute covariance
    cov = np.zeros((3, 3))
    for elem in neighbours:
        elem = elem - centroid
        elem = elem.reshape(-1, 1)
        temp = np.dot(elem, elem.T)
        cov += temp

    cov = cov / len(neighbours)

    return cov


def compute_approximate_curvature(cov_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the normal vector and the approximate curvature based on a covariance matrix
    at a given point.

    :param cov_matrix: a convariance matrix around a 3D point.
    :return: normal vector and approximate curvature.
    """
    # compute eigen values and eigen vectors of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

    # get array of index which would sort the array of eigen values
    # (i.e. first index is the index of the smallest eigen value)
    sorted_index_of_eigen_values = np.argsort(eigen_values)

    # get normal vector
    normal_vector = eigen_vectors[sorted_index_of_eigen_values[0]]

    # compute sigma square: smallest eigen value divided by sum of eigen values
    sigma_square = eigen_values[sorted_index_of_eigen_values[0]] / eigen_values.sum()

    return float(sigma_square), normal_vector


def projector(point_of_origin: np.ndarray, normal: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Project a target point toward a plane define by a normal vector and a point of origin.
    The plane is orthogonal to the normal vector and pass by the point of origin.

    Based on this question: https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d

    :param point_of_origin: point of origin
    :param normal: normal vector.
    :param target: target point to be projected toward the plane.
    """
    p0, p1, p2 = point_of_origin  # the point by which the plane has to pass
    n0, n1, n2 = normal  # the vector to which the plane has to be orthogonal
    t0, t1, t2 = target  # the point we want to project to the plane

    # First, we compute the necessary distance to make the plane move along the normal vector from (0,0,0)
    # until the point of origin.
    # This need to satisfy the following equation : n0*p0 + n1*p1 + n2*p2 + distance = 0
    distance = - (n0 * p0 + n1 * p1 + n2 * p2)

    # we can now compute the scalar distance of any point to the plane:
    # normal * target + distance
    target_distance_to_the_plane = ((n0 * t0 + n1 * t1 + n2 * t2) + distance)

    # we do a translation toward the plane from the target point, alongside the normal vector,
    # equal to the distance between the plane and the target point:
    pt0, pt1, pt2 = (t0 - target_distance_to_the_plane * n0, t1 - target_distance_to_the_plane * n1,
                     t2 - target_distance_to_the_plane * n2)

    return np.array([pt0, pt1, pt2])


def data_processing_pipeline(pcd_path: str, point_index: int, radius: float):
    """
    Pipeline to process the point cloud at a given point of origin:
    - retrieve all the pojnt in a given radius
    - compute a covariance based on those point
    - use the covariance to compute an approximate curvature and a normal vector.
    - project the point in the radius toward a plane defined by a normal vector and the point of origin.

    :param pcd_path: path of the point cloud we want to process.
    :param point_index: index of the point
    :param radius: radius around the point
    :return cov_matrix, sigma_square, normal_vector, pcd:
    """

    # load point cloud
    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcd_path)

    # build KD-Tree
    pcd_tree: KDTreeFlann = o3d.geometry.KDTreeFlann(pcd)

    # compute covariance matrix
    cov_matrix = compute_covariance_at_index(pcd, pcd_tree, point_index, radius)

    # compute approximate curvature, and return normal vector
    sigma_square, normal_vector = compute_approximate_curvature(cov_matrix)

    # project points within a radius of the point of origin onto the plane defined
    # by the point of origin and the normal vector

    point_of_origin = pcd.points[point_index]

    [_, idx, _] = pcd_tree.search_radius_vector_3d(point_of_origin, radius)
    neighbours = np.array([pcd.points[elem] for elem in idx])

    np.asarray(pcd.points)[idx, :] = np.array([projector(point_of_origin, normal_vector, neighbour)
                                               for neighbour in neighbours])  # Broadcast and change in place

    return cov_matrix, sigma_square, normal_vector, pcd
