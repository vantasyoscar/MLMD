import numpy as np
import math


def get_neighbor_atoms(coordinates: np.ndarray, cutoff: float = 5, lattice: np.ndarray = None) -> list:
    r"""
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
    r"""
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

def cartesian_to_spherical(coordinates):
    """
    将n*3的numpy数组中每一行的直角坐标系转换成球坐标系

    输入：
    coordinates : n*3的numpy数组，每一行表示一个点在三维直角坐标系上的坐标

    输出：
    n*3的numpy数组，每一行表示一个点在球坐标系上的坐标
    """
    # 计算球坐标系上的点的坐标
    radii = np.linalg.norm(coordinates, axis=1)
    theta = np.arccos(coordinates[:, 2] / radii)
    phi = np.arctan2(coordinates[:, 1], coordinates[:, 0])


    # 将弧度转换为角度
    #theta = np.degrees(theta)
    #phi = np.degrees(phi)

    # 将球坐标系上的点的坐标存储在一个n*3的numpy数组中
    spherical_coordinates = np.stack((radii, theta, phi), axis=1)

    return spherical_coordinates


def special_cartesian_to_spherical(coordinates):
    """
    将n*3的numpy数组中每一行的直角坐标系转换成球坐标系

    输入：
    coordinates : n*3的numpy数组，每一行表示一个点在三维直角坐标系上的坐标

    输出：
    n*3的numpy数组，每一行表示一个点在球坐标系上的坐标
    """
    # 计算球坐标系上的点的坐标
    radii = np.linalg.norm(coordinates, axis=1)
    
    theta = np.arccos(coordinates[:, 2] / radii)

    phi = np.arctan2(coordinates[:, 1], coordinates[:, 0])

    radii = 10/(radii**2)
    # 将弧度转换为角度
    #theta = np.degrees(theta)
    #phi = np.degrees(phi)

    # 将球坐标系上的点的坐标存储在一个n*3的numpy数组中
    spherical_coordinates = np.stack((radii, theta, phi), axis=1)

    return spherical_coordinates


def special_translate_vec(coordinates: np.ndarray, lattice: np.ndarray = None) -> np.ndarray:
    """
    计算晶格周期性边界条件下，原点到最近的点的距离，即移动成WS原胞

    Parameters
    ----------
    coordinates : np.ndarray
        三维坐标数组，表示晶格中的点。
    lattice : np.ndarray, optional
        三行三列的矩阵，表示晶格矢量。默认为 None。

    Returns
    -------
    np.ndarray
        新的WS原胞的坐标

    Examples
    --------
    >>> coordinates = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> lattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> special_translate(coordinates, lattice)
    array([[0. , 0. , 0. ],
           [0.5, 0.5, 0.5]])
    """

    # 首先生成晶格矢量的所有组合
    vec_list = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3)
    # 将晶格矢量乘以系数得到所有的位移向量
    shift_vec = vec_list @ lattice
    # 将每个坐标向量平移并找到距离原点最近的位置
    shifted_coords = coordinates[:, None, :] + shift_vec[None, :, :]
    distance = np.linalg.norm(shifted_coords, axis=-1)
    min_index = np.argmin(distance, axis=1)
    coord_ws = shifted_coords[np.arange(len(coordinates)), min_index]
    return coord_ws - coordinates  


def encode_coordinantes(train_data,neighborlist,Lattice,num):
    neighbor_coord = []
    vec = []
    for i in range(num):
        neighboratom = get_coordinates_from_indices(train_data[i].reshape(-1,3),neighborlist)
        for j ,k in enumerate(neighboratom):
            #平移到中心           
            vector_3d = np.tile(train_data[i].reshape(-1,3)[j],(k.shape[0],1))
            #coord_ws = MLMD.neighbor_list.special_translate(k-vector_3d,lattice = LiF["Li"].lattice)
            if i == 0:
                vec.append(special_translate_vec(k - vector_3d,lattice = Lattice))
                #WS原胞中心 MLMD.neighbor_list.special_translate(k-vector_3d,lattice = LiF["Li"].lattice)
                #球坐标 MLMD.neighbor_list.cartesian_to_spherical(k-vector_3d)
                neighbor_coord.append([k - vector_3d + vec[j]])
            else:
                neighbor_coord[j].append(k - vector_3d + vec[j])
    return neighbor_coord,vec

def get_atom_kind_list(neighborlist,atom_dict):
    #输入临近原子序数，输出原子类型（0，1……）
    atom_kind_list = []
    for i in neighborlist:
        atom_kind_list.append([])
        for j in i:
            for k,l in enumerate(atom_dict.values()): 
                j -= l.number
                if j < 0:
                    atom_kind_list[-1].append(k)
                    break
    return atom_kind_list

'''废弃的球坐标编码
def encode_coordinantes(train_data,neighborlist,Lattice,num):
    neighbor_coord = []
    vec = []
    for i in range(num):
        neighboratom = MLMD.neighbor_list.get_coordinates_from_indices(train_data[i].reshape(-1,3),neighborlist)
        for j ,k in enumerate(neighboratom):
            #平移到中心           
            vector_3d = np.tile(train_data[i].reshape(-1,3)[j],(k.shape[0],1))
            #coord_ws = MLMD.neighbor_list.special_translate(k-vector_3d,lattice = LiF["Li"].lattice)
            if i == 0:
                vec.append(MLMD.neighbor_list.special_translate_vec(k - vector_3d,lattice = Lattice))
                #WS原胞中心 MLMD.neighbor_list.special_translate(k-vector_3d,lattice = LiF["Li"].lattice)
                #球坐标 MLMD.neighbor_list.cartesian_to_spherical(k-vector_3d)
                neighbor_coord.append([MLMD.neighbor_list.special_cartesian_to_spherical(k - vector_3d + vec[j])])
            else:
                neighbor_coord[j].append(MLMD.neighbor_list.special_cartesian_to_spherical(k - vector_3d + vec[j]))
    return neighbor_coord,vec
'''