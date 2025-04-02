"""
使用pygmsh来划分网格,然后写入.msh文件
"""
import os
import pygmsh
import numpy as np
from typing import Tuple, List
import numpy.typing as npt
import yaml
# import pyvista as pv
from scipy.spatial import Delaunay
import random


def mesh_obj_tri(obj_shape:List[float], seed_size:float)->Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """将二维对象生成三角形网格
    Args:
        obj_shape (List[float]): [length, width]
        seed_size (float): 网格尺寸
    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]: 节点、边、单元
    """
    length, width = obj_shape

    length_n = int(length / seed_size)
    width_n = int(width / seed_size)

    length_n = length_n if abs(length - length_n * seed_size) < 1.e-6 else length_n + 1
    width_n = width_n if abs(width - width_n * seed_size) < 1.e-6 else width_n + 1

    xx, yy = np.meshgrid(np.linspace(0, length, length_n+1), np.linspace(0, width, width_n+1))
    xx_pad = xx.flatten('C')
    yy_pad = yy.flatten('C')
    node = np.array([xx_pad, yy_pad], dtype=float).T         # dim: N*2

    tri = Delaunay(node)
    element = np.sort(tri.simplices, axis=1)

    edge_set = set()
    for simplices in element:
        for i in range(3):
            edge_temp = tuple(sorted(simplices[[i, (i + 1) % 3]]))
            edge_set.add(edge_temp)

    edge = np.array(list(edge_set), dtype=int)

    return node, edge, element


def write_msh2_tri(filename:str, nodes:npt.NDArray, triangles:npt.NDArray):
    """写入版本为2的三角形网格的.msh文件
    Args:
        nodes (npt.NDArray): 节点列表，每个元素为 (x, y) 或 (x, y, z)
        triangles (npt.NDArray): 三角形面片列表，每个元素为 (i, j, k), 假定索引从0开始
        filename (str): 输出文件名
    """
    with open(filename, "w") as f:
        # 写入 MeshFormat 部分
        f.write("$MeshFormat\n")
        # 版本号2.2，文件类型0（ASCII），数据大小8
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # 写入 Nodes 部分
        f.write("$Nodes\n")
        f.write("{}\n".format(len(nodes)))
        for i, node in enumerate(nodes, start=1):
            # 如果节点只有两个坐标，则默认 z=0.0
            if node.shape[0] == 2:
                x, y = node
                z = 0.0
            else:
                x, y, z = node
            f.write("{} {} {} {}\n".format(i, x, y, z))
        f.write("$EndNodes\n")
        
        # 写入 Elements 部分
        f.write("$Elements\n")
        f.write("{}\n".format(triangles.shape[0]))
        for i, tri in enumerate(triangles, start=1):
            # Gmsh中，三角形单元的类型为2
            # 此处 tags 数量设为0（可以根据需要增加物理区域等信息）
            # 注意：将0开始的索引转换为1开始
            n1, n2, n3 = tri
            f.write("{} 2 0 {} {} {}\n".format(i, n1+1, n2+1, n3+1))
        f.write("$EndElements\n")


if __name__ == '__main__':
    pass