"""
created on 
"""
import os
import pygmsh
import numpy as np
from typing import Tuple, List
import numpy.typing as npt
import yaml
from scipy.spatial import Delaunay
import random

def mesh_obj_tri(obj_shape:List[float], seed_size:float)->Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """ Generate a triangular mesh for a 2D object
    Args:
        obj_shape (List[float]): [length, width]
        seed_size (float): mesh size
    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]: nodes, edges, elements
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
    """ Write a triangular mesh to a .msh file (Version 2.2 format)
    Args:
        nodes (npt.NDArray): Vertex node list, each element is (x, y) or (x, y, z)
        triangles (npt.NDArray): Triangle face list, each element is (i, j, k), assuming 0-based indexing
        filename (str): Output file name
    """
    with open(filename, "w") as f:
        f.write("$MeshFormat\n")
        # Version 2.2, file type 0 (ASCII), data size 8
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        f.write("$Nodes\n")
        f.write("{}\n".format(len(nodes)))
        for i, node in enumerate(nodes, start=1):
            # If the node has only two coordinates, default z=0.0
            if node.shape[0] == 2:
                x, y = node
                z = 0.0
            else:
                x, y, z = node
            f.write("{} {} {} {}\n".format(i, x, y, z))
        f.write("$EndNodes\n")

        # Write the Elements section
        f.write("$Elements\n")
        f.write("{}\n".format(triangles.shape[0]))
        for i, tri in enumerate(triangles, start=1):
            # In Gmsh, the element type for triangles is 2
            # Here, the number of tags is set to 0 (can be increased for physical regions, etc.)
            # Note: Convert 0-based indexing to 1-based
            n1, n2, n3 = tri
            f.write("{} 2 0 {} {} {}\n".format(i, n1+1, n2+1, n3+1))
        f.write("$EndElements\n")

if __name__ == '__main__':
    pass