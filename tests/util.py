import open3d as o3d

from typing import Tuple


def get_data_path() -> Tuple[str, str]:
    """
    Get path to datasets used for testing purpose.
    :return: Tuple containing two data path in string.
    """
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    path_1 = demo_icp_pcds.paths[0]
    path_2 = demo_icp_pcds.paths[1]
    return path_1, path_2