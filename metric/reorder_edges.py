import numpy as np
from typing import Optional, Tuple, List

def _endpoints(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
      starts: (K,2) 每条边的起点
      ends  : (K,2) 每条边的终点
    """
    starts = edges[:, 0, :]
    ends   = edges[:, -1, :]
    return starts, ends

def _edge_len(edges: np.ndarray) -> np.ndarray:
    """
    估计每条边的长度（首末点距离），用于选择起点的启发。
    """
    s, e = _endpoints(edges)
    return np.linalg.norm(e - s, axis=1)


def reorder_and_flip_edges(
    edges: np.ndarray,
    *,
    start_index: Optional[int] = None,   # 指定从哪条边开始；None 表示自动选择一个“联通性最好”的起点
    make_cycle: bool = True,             # 是否尽量让最后一条的末端接近第一条的起点
    tol: Optional[float] = None,         # 容差（仅用于诊断；不影响贪心选择），None 表示不阈值判断
    prefer_long_start: bool = True,      # 自动起点时优先选“较长”的边
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将 edges 重新排序并按需翻转，使得相邻两条边在数组中“顺次连接”：
      new_edges[i][-1] ≈ new_edges[i+1][0]

    参数：
      edges: (K,N,2)
      start_index: 指定起点边的原索引；None 则自动选择
      make_cycle: 最后尝试让最后一条的末端接近第一条的起点（不强制，仅报告距离）
      tol: 若给定，返回的 distances 中若有 > tol 的项，可据此后处理或报警
      prefer_long_start: 自动起点时倾向选择首末距离较大的边

    返回：
      edges_reordered: (K,N,2) 新次序且已按需翻转
      order          : (K,)    新次序对应的原索引
      flips          : (K,)    每条边是否翻转（True=反向）
      distances      : (K-1 + 可选收尾) 相邻连接的欧氏距离（最后一项是闭环距离，若 make_cycle=True）
    """
    E = np.asarray(edges, dtype=float)
    assert E.ndim == 3 and E.shape[2] == 2, "edges 必须为 (K,N,2)"
    K, N, _ = E.shape

    starts, ends = _endpoints(E)

    # ---------- 选择起点 ----------
    if start_index is None:
        # 简单启发：优先选择“更长”的边作为起点；若有并列，选平均到其它端点总距离最小的
        lengths = _edge_len(E)
        cand = np.argsort(-lengths) if prefer_long_start else np.arange(K)
        # 进一步用到其它端点的平均距离来细化
        all_points = np.vstack([starts, ends])  # (2K,2)
        best_idx, best_score = None, np.inf
        for idx in cand[:min(10, K)]:  # 只看前 10 个候选即可
            # 起点取该边两个端之一，使得“到其它端点的最近距离”较小
            s0, e0 = starts[idx], ends[idx]
            d1 = np.min(np.linalg.norm(all_points - s0, axis=1))
            d2 = np.min(np.linalg.norm(all_points - e0, axis=1))
            score = min(d1, d2) - 1e-9*idx  # 轻微偏置，稳定tie
            if score < best_score:
                best_score = score
                best_idx = idx
        start_index = int(best_idx)

    # 决定起点边的方向：选择其“尾巴”更接近其它任何边的某个端点
    s0, e0 = starts[start_index], ends[start_index]
    others = np.delete(np.arange(K), start_index)
    # 距离：当前“末端”到其它边两个端点，取最小者
    def _min_to_others(pt):
        s_oth = starts[others]
        e_oth = ends[others]
        d = np.minimum(np.linalg.norm(pt - s_oth, axis=1),
                       np.linalg.norm(pt - e_oth, axis=1))
        return float(np.min(d))
    # 若以“正向”作为第一条，当前末端是 e0；若以“翻转”，末端是 s0
    use_flip0 = _min_to_others(s0) < _min_to_others(e0)
    cur_idx   = start_index
    cur_flip  = use_flip0
    used      = np.zeros(K, dtype=bool)
    used[cur_idx] = True

    order: List[int] = [cur_idx]
    flips: List[bool] = [cur_flip]
    distances: List[float] = []

    # 当前“游标点” = 已放入序列的最后一条边的“末端”
    def _end_point(idx: int, flip: bool) -> np.ndarray:
        return starts[idx] if flip else ends[idx]
    def _start_point(idx: int, flip: bool) -> np.ndarray:
        return ends[idx] if flip else starts[idx]

    cur_end = _end_point(cur_idx, cur_flip)

    # ---------- 贪心扩展 ----------
    for _ in range(K - 1):
        candidates = np.where(~used)[0]
        # 对每个候选，比较两种方向，使其“起点”尽量靠近 cur_end
        best_j, best_flip, best_d = None, False, np.inf
        for j in candidates:
            # 方向1：不翻转，则“起点”= starts[j]
            d1 = np.linalg.norm(cur_end - starts[j])
            # 方向2：翻转，则“起点”= ends[j]
            d2 = np.linalg.norm(cur_end - ends[j])
            if d1 <= d2:
                d, flip = d1, False
            else:
                d, flip = d2, True
            if d < best_d - 1e-12 or (abs(d - best_d) <= 1e-12 and j < (best_j if best_j is not None else 1<<30)):
                best_j, best_flip, best_d = j, flip, d

        # 选择最佳候选并更新“游标点”为其末端
        order.append(int(best_j))
        flips.append(bool(best_flip))
        distances.append(float(best_d))
        used[best_j] = True
        cur_idx, cur_flip = int(best_j), bool(best_flip)
        cur_end = _end_point(cur_idx, cur_flip)

    order = np.array(order, dtype=int)
    flips = np.array(flips, dtype=bool)

    # ---------- 组装输出 ----------
    edges_reordered = []
    for idx, fl in zip(order, flips):
        ei = E[idx]
        if fl:
            ei = ei[::-1].copy()
        edges_reordered.append(ei)
    edges_reordered = np.stack(edges_reordered, axis=0)

    # 补充闭环距离（如果需要）
    if make_cycle:
        d_close = np.linalg.norm(edges_reordered[-1, -1] - edges_reordered[0, 0])
        distances.append(float(d_close))

    distances = np.array(distances, dtype=float)

    # ---------- 可选：容差诊断 ----------
    if tol is not None:
        bad = np.where(distances > tol)[0]
        if bad.size > 0:
            # 这里只做提示；真正的强制可在上面加约束或做局部交换2-opt
            print(f"[reorder] 有 {bad.size} 处连接距离大于 tol={tol:.3g}，索引: {bad.tolist()}，最大={distances.max():.6g}")

    return edges_reordered, order, flips, distances