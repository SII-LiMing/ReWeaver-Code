#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
# =========================
# 直接在这里改参数
# =========================
KNN = 40
THETA_DEG = 150.0
MIN_K = 12
R_SCALE = 2.0
# =========================


@torch.no_grad()
def boundary_by_angle_gap_torch(pts: torch.Tensor,
                                k: int = KNN,
                                theta_deg: float = THETA_DEG,
                                min_k: int = MIN_K,
                                r_scale: float = R_SCALE) -> torch.Tensor:
    """
    pts: (N,3) float32/64, on CPU/GPU均可
    return: (M,) long 边界点索引
    """
    N = pts.shape[0]
    # kNN（包含自身）
    D = torch.cdist(pts, pts)                              # (N,N)
    d_sorted, idx_sorted = torch.topk(D, k=min(k, N), dim=1, largest=False)
    # 局部尺度（中位最近邻距）→ 过滤极近邻
    local_r = torch.median(d_sorted[:, 1:], dim=1).values * r_scale + 1e-12

    is_boundary = torch.zeros(N, dtype=torch.bool, device=pts.device)
    th = math.radians(theta_deg)

    a0 = torch.tensor([1.0, 0.0, 0.0], dtype=pts.dtype, device=pts.device)
    a1 = torch.tensor([0.0, 1.0, 0.0], dtype=pts.dtype, device=pts.device)

    for i in range(N):
        neigh_ids = idx_sorted[i]          # (k,)
        # 去掉自身（第一项距离0）
        neigh_ids = neigh_ids[1:]
        Pi = pts[neigh_ids] - pts[i]       # (k-1,3)

        # 过滤过近邻
        keep = torch.linalg.norm(Pi, dim=1) > local_r[i] * 0.2
        Pi = Pi[keep]
        if Pi.shape[0] < min_k:
            continue

        # 局部协方差 + 最小特征向量为法向
        C = (Pi.T @ Pi) / Pi.shape[0]      # (3,3)
        w, V = torch.linalg.eigh(C)        # V[:,j] 为特征向量
        n = V[:, 0]
        n = n / (torch.linalg.norm(n) + 1e-12)

        # 切平面基 u,v
        a = a0 if torch.abs(torch.dot(a0, n)) <= 0.9 else a1
        u = a - torch.dot(a, n) * n
        u = u / (torch.linalg.norm(u) + 1e-12)
        v = torch.linalg.cross(n, u)

        # 投影到切平面
        x = Pi @ u
        y = Pi @ v
        ang, _ = torch.sort(torch.atan2(y, x))

        # 极角环最大间隙
        ang_wrap = torch.cat([ang, ang[:1] + 2 * math.pi], dim=0)
        gaps = ang_wrap[1:] - ang_wrap[:-1]
        max_gap = torch.max(gaps).item()

        if max_gap > th:
            is_boundary[i] = True

    return torch.where(is_boundary)[0]


@torch.no_grad()
def pca_plane_torch(points: torch.Tensor):
    c = points.mean(dim=0, keepdim=True)
    X = points - c
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    basis2 = Vh[:2, :].T                      # (3,2)
    return c, basis2


@torch.no_grad()
def project_to_plane_torch(points: torch.Tensor, center: torch.Tensor, basis2: torch.Tensor):
    return (points - center) @ basis2         # (N,2)


@torch.no_grad()
def get_boundary(pts):
    center, basis2 = pca_plane_torch(pts)
    pts_c = pts - center
    idx = boundary_by_angle_gap_torch(pts)    # 索引 (M,)
    bnd_centered = pts_c.index_select(0, idx) # (M,3)
    bnd_world = bnd_centered + center
    return bnd_world, idx

