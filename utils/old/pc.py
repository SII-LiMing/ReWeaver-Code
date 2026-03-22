import numpy as np
from scipy.spatial import cKDTree

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    
    return pc,centroid,m

def resample_curve(points, K):
    """
    Resample a curve given by `points` into K evenly spaced points along arc length.
    Args:
        points: np.ndarray of shape (N, 3)
        K: int, number of resampled points
    Returns:
        np.ndarray of shape (K, 3)
    """
    deltas = np.linalg.norm(points[1:] - points[:-1], axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(deltas)])
    total_length = arc_lengths[-1]
    target_lengths = np.linspace(0, total_length, K)

    resampled = np.zeros((K, 3))
    for d in range(3):
        resampled[:, d] = np.interp(target_lengths, arc_lengths, points[:, d])
    return resampled


def find_closest_points(pc1, pc2, exclude_self=False, atol=1e-8):
    """
    对于 pc2 中的每个点，找到 pc1 中距离其最近的点，可选排除“自己”。

    参数:
        pc1: numpy 数组，形状为 (N, D)
        pc2: numpy 数组，形状为 (M, D)
        exclude_self: bool，是否排除 pc2 中与 pc1 重合的点（默认 False）
        atol: float，判断“自己”的距离容差（默认 1e-8）

    返回:
        matched_pc1: (M, D) 数组，pc1 中每个与 pc2 最近的点
        indices: (M,) 数组，pc1 中这些点的索引
    """
    tree = cKDTree(pc1)
    if pc2.ndim==1:
        pc2=pc2.reshape(1,3)
    if exclude_self:
        # 查询两个最近的点，排除自己（距离≈0）
        distances, indices = tree.query(pc2, k=2)
        # 对于每个查询点，选择第一个非自身（即距离大于 atol 的）
        final_indices = np.where(distances[:, 0] > atol, indices[:, 0], indices[:, 1])
    else:
        # 查询单个最近点
        distances, final_indices = tree.query(pc2)

    matched_pc1 = pc1[final_indices]
    if pc2.shape[0]==1:
        return matched_pc1[0],final_indices[0]
    return matched_pc1, final_indices


def find_closest_points_many(pc1, pc2, num=1):
    """
    对于 pc2 中的每个点，找到 pc1 中距离其最近的点，可选排除“自己”。

    参数:
        pc1: numpy 数组，形状为 (N, D)
        pc2: numpy 数组，形状为 (M, D)
        exclude_self: bool，是否排除 pc2 中与 pc1 重合的点（默认 False）
        atol: float，判断“自己”的距离容差（默认 1e-8）

    返回:
        matched_pc1: (M, D) 数组，pc1 中每个与 pc2 最近的点
        indices: (M,) 数组，pc1 中这些点的索引
    """
    tree = cKDTree(pc1)
    if pc2.ndim==1:
        pc2=pc2.reshape(1,3)

    distances, final_indices = tree.query(pc2,k=num)

    matched_pc1 = pc1[final_indices]
    if pc2.shape[0]==1:
        return matched_pc1[0],final_indices[0]
    return matched_pc1, final_indices