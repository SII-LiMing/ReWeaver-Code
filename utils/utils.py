import trimesh
from scipy.spatial import cKDTree
import numpy as np
from collections import defaultdict
import json
import os
from pathlib import Path
from tqdm import tqdm
from copy import copy
from utils import rotation as rotation_tools
import numpy as np
import random
import torch


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    
    return pc,centroid,m

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def merge_stitch(mesh: trimesh.Trimesh,return_stitch=True):
    """
    合并 mesh 中的重复顶点，并返回合并后的 mesh 以及属于 stitch 的点坐标。

    参数：
        mesh (trimesh.Trimesh): 输入网格
        threshold (float): 判定为重复点的距离阈值（默认几乎为0）

    返回：
        new_mesh (trimesh.Trimesh): 合并后的 mesh
        stitch_coords (np.ndarray): 新 mesh 中属于 stitch 的点坐标 (去重后的)
    """
    vertices = mesh.vertices
    tree = cKDTree(vertices)
    duplicates = tree.query_pairs(r=0)
    duplicates = list(duplicates)

    if not duplicates:
        # 没有重复点，直接返回复制
        if return_stitch:
            return mesh.copy(), []
        else:
            return mesh.copy()

    # 找出所有被认为重复的点的索引
    stitch_indices = set()
    for i, j in duplicates:
        stitch_indices.add(i)
        stitch_indices.add(j)
    stitch_indices = sorted(list(stitch_indices))

    # 获取这些点的坐标（去重前）
    stitch_coords_orig = vertices[stitch_indices]

    # 复制并合并顶点（默认合并法线和纹理）
    new_mesh = mesh.copy()
    new_mesh.merge_vertices(merge_tex=True,merge_norm=True)

    # 在新的 mesh 中查找哪些点属于 stitch（可能坐标发生了变化）
    merged_tree = cKDTree(new_mesh.vertices)
    _, idxs = merged_tree.query(stitch_coords_orig, k=1)

    if return_stitch:
        return new_mesh, list(set(idxs.tolist()))
    else:
        return new_mesh


def find_stitch(mesh_vertices):
    tree = cKDTree(mesh_vertices)
    duplicates = tree.query_pairs(r=0)
    duplicates=list(duplicates)
    duplicates= [list(it) for it in duplicates]
    return duplicates




def find_border(mesh: trimesh.Trimesh):
    """
    通用方法：提取构成边界的顶点索引。
    
    参数:
        mesh (trimesh.Trimesh): 输入三角网格。

    返回:
        list[int]: 边界顶点索引。
    """
    # 创建一个字典来统计每条边的出现次数
    edge_count = defaultdict(int)

    # 遍历所有面
    for face in mesh.faces:
        # 三角形的三条边（用排序后的元组保证一致性）
        edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ]
        for edge in edges:
            edge_count[edge] += 1

    # 边界边只出现过一次
    border_edges = [tuple(map(int,edge)) for edge, count in edge_count.items() if count == 1]

    # 提取所有边界边的顶点索引
    border_vertices = set()
    for edge in border_edges:
        border_vertices.update(edge)

    return list(border_vertices)


def point_in_3D(local_coord, rotation, translation):
    """Apply 3D transformation to the point given in 2D local coordinated, e.g. on the panel
    * rotation is expected to be given in 'xyz' Euler anges (as in Autodesk Maya) or as 3x3 matrix"""

    # 2D->3D local
    local_coord = np.append(local_coord, 0)

    # Rotate
    rotation = np.array(rotation)
    
    if rotation.size == 3:  # transform Euler angles to matrix
        rotation = rotation_tools.euler_xyz_to_R(rotation)
        # otherwise we already have the matrix
    elif rotation.size != 9:
        raise ValueError('BasicPattern::ERROR::You need to provide Euler angles or Rotation matrix for _point_in_3D(..)')
    rotated_point = rotation.dot(local_coord)

    # translate
    return rotated_point + translation


def get_pattern_json_with_3d_vertices(pattern):
    # TODOLOW Support arcs / curves (use linearization)
    for panel in pattern['panels']:
        # import ipdb;ipdb.set_trace()
        p = pattern['panels'][panel]
        rot = p['rotation']
        tr = p['translation']
        verts_2d = p['vertices']
        verts_to_plot = copy(verts_2d)

        verts3d = np.vstack(tuple([point_in_3D(v, rot, tr) for v in verts_to_plot]))
        p['vertices_3d']= verts3d.tolist()
        


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx   

if __name__=="__main__":
    mesh=trimesh.load("D:/GarmentCode_5000_0_gt/test/000001/000001.obj")
    stitch=find_stitch(mesh.vertices)
    
    import ipdb
    ipdb.set_trace()

