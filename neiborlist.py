import numpy as np
def get_neighbor_atoms(coordinates, cutoff=5, lattice=None):
    """
    获取原子相邻的原子序号

    Args:
        coordinates (np.ndarray): 一个n * 3的np.array，表示n个原子的三维坐标
        cutoff (float): 两个原子之间的距离小于cutoff时，认为这两个原子相邻。默认值为5
        lattice (np.ndarray): 原子的三个晶格常数构成的矩阵。默认值为None，表示不考虑晶格周期性边界条件

    Returns:
        list: 一个长度为n的列表，其中每个元素是一个列表，包含与当前原子距离小于cutoff的原子序数。

    Examples:
        有三个原子，其坐标分别为(0, 0, 0)，(1, 1, 1)和(3, 3, 3)。则：

        >>> coordinates = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
        >>> get_neighbor_atoms(coordinates, cutoff=2.1)
        [[1], [0, 2], [1]]

    """
    # 初始化结果列表
    result = []

    # 遍历所有原子组合
    for i in range(coordinates.shape[0]):
        close_atoms = []
        for j in range(coordinates.shape[0]):
            if i == j:
                continue
            distance_list = []

            # 考虑晶格周期性边界条件
            if lattice is not None:
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        for m in range(-1, 2):
                            distance_list.append(np.linalg.norm(
                                coordinates[i, :] - coordinates[j, :] +
                                k * lattice[0] + l * lattice[1] + m * lattice[2]))
            else:
                distance_list.append(np.linalg.norm(coordinates[i, :] - coordinates[j, :]))

            distance = min(distance_list)
            if distance < cutoff:
                close_atoms.append(j)
        result.append(close_atoms)

    return result


def get_coordinates_from_indices(coordinates: np.ndarray, indices_list: list) -> list:
    """
    该函数接受坐标数组和原子序数列表作为输入，并返回一个列表，
    其中每个元素是一个与对应原子序数对应的原子坐标列表

    Args:
    coordinates: numpy array, 储存了每一个原子的坐标信息
    indices_list: list, 储存了需要获取坐标的原子的序号

    Returns:
    result: list of numpy array, 包含了与原子序号对应的坐标信息
    """
    result = []
    for indices in indices_list:
        coord_list = [coordinates[i] for i in indices]
        result.append(np.array(coord_list))
    return result

def translate_coordinates(coords_within_distance, coordinates):
    """
    将坐标系平移，使得指定的`coordinates`坐标为原点，返回平移后的坐标系。

    Args:
        coords_within_distance (List[np.array]): 坐标系列表，每个坐标系用一个numpy数组表示。
        coordinates (List[np.array]): 坐标列表，每个坐标用一个numpy数组表示。

    Returns:
        List[np.array]: 平移后的坐标系列表。
    """
    translated_coords = []
    for i, coords in enumerate(coords_within_distance):
        translation_vector = -coordinates[i]
        translated_group = [coord + translation_vector for coord in coords]
        translated_coords.append(np.array(translated_group))
    return translated_coords