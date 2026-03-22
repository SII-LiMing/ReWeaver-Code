import torch


def r2_linear(y: torch.Tensor) -> torch.Tensor:
    """输入 (B,K,N)，沿最后一维用样本索引 x=0..N-1 拟合，返回 (B,K) 的 R^2"""
    y = y.to(torch.float64)
    N = y.size(-1)
    x = torch.arange(N, dtype=torch.float64, device=y.device)
    x = x - x.mean()                          # (N,)
    yc = y - y.mean(dim=-1, keepdim=True)     # (B,K,N)
    num = (yc * x).sum(dim=-1)                # (B,K)
    den = torch.sqrt((x.pow(2).sum()) * (yc.pow(2).sum(dim=-1)))  # (B,K)
    r = num / den
    return r * r                               # (B,K)

def scale_point_cloud_to_aabb(
    pts: torch.Tensor,
    tgt_min: torch.Tensor = torch.tensor([0., 0., 0.]),
    tgt_max: torch.Tensor = torch.tensor([1., 1., 1.]),
    keep_aspect: bool = True,   # True 等比缩放；False 各轴分别缩放
    center_align: bool = True,  # True 按中心对齐；False 按最小角(min)对齐
    mode: str = "fit",          # 等比时：'fit' 取最小比例完整放入；'fill' 取最大比例铺满
    eps: float = 1e-12,
):
    """
    将点云 pts 按目标 AABB[tgt_min, tgt_max] 做线性缩放/平移。
    - pts: (N,3) float 张量
    - tgt_min, tgt_max: (3,) 目标边界（同 dtype/device）
    - keep_aspect:
        True  => 等比缩放（单一比例 s）；False => 各轴独立缩放（s_x, s_y, s_z）
    - center_align:
        True  => 源与目标按中心对齐；False => 源 min 对齐到目标 min
    - mode (仅在 keep_aspect=True 时生效):
        'fit'  => s = min(target_size / source_size) 让点云完整放入目标框
        'fill' => s = max(target_size / source_size) 铺满目标框（可能越界）
    返回：
        new_pts: (N,3) 缩放+平移后的点
        info: dict 包含变换参数（scale, translation, 源/目标 AABB）
    """
    assert pts.ndim == 2 and pts.size(-1) == 3, "pts 应为 (N,3)"
    device = pts.device
    dtype = pts.dtype

    tgt_min = tgt_min.to(device=device, dtype=dtype)
    tgt_max = tgt_max.to(device=device, dtype=dtype)

    # 源 AABB
    src_min = pts.min(dim=0).values
    src_max = pts.max(dim=0).values
    src_size = (src_max - src_min).clamp_min(eps)     # 防零
    tgt_size = (tgt_max - tgt_min).clamp_min(eps)

    # 缩放因子
    if keep_aspect:
        ratios = tgt_size / src_size
        if mode == "fill":
            s = torch.max(ratios)   # 铺满
        else:
            s = torch.min(ratios)   # fit (默认)
        scale = torch.full((3,), s, device=device, dtype=dtype)
    else:
        scale = tgt_size / src_size  # 各轴各缩

    # 平移：先缩放到源坐标系，再对齐
    pts_scaled = (pts - src_min) * scale  # 先把 src_min 当作原点缩放
    if center_align:
        src_center = (src_max + src_min) * 0.5
        tgt_center = (tgt_max + tgt_min) * 0.5
        # 注意：按中心对齐时，缩放应围绕 src_center 进行，否则会偏移
        # 重新按中心缩放更稳妥：
        pts_scaled = (pts - src_center) * scale + tgt_center
        translation = tgt_center - src_center * scale
    else:
        # min 对齐：缩放后把最小角对齐到 tgt_min
        # 已以 src_min 为基点缩放，当前 min 在 0 处
        pts_scaled = pts_scaled + tgt_min
        translation = tgt_min - src_min * scale

    new_pts = pts_scaled

    info = {
        "scale": scale,                 # (3,)
        "translation": translation,     # (3,)
        "src_min": src_min, "src_max": src_max,
        "tgt_min": tgt_min, "tgt_max": tgt_max,
        "keep_aspect": keep_aspect, "center_align": center_align, "mode": mode
    }
    return new_pts, info

def apply_affine_from_info(pts: torch.Tensor, info: dict) -> torch.Tensor:
    """
    用 scale / translation 对另一组点云做同样的缩放+平移
    p' = p * scale + translation
    """
    scale = info["scale"].to(device=pts.device, dtype=pts.dtype)         # (3,) 或标量
    translation = info["translation"].to(device=pts.device, dtype=pts.dtype)  # (3,)
    return pts * scale + translation



def chamfer_distance(x: torch.Tensor, y: torch.Tensor, direction: str = 'bi') -> float:
    """
    x, y: (N,3) / (M,3) 的 torch.Tensor
    返回值为 Python float(便于与阈值比较)
    方向语义与原实现一致：
    - 'x_to_y':  x 中每个点到 y 的最近距离的平均
    - 'y_to_x':  y 中每个点到 x 的最近距离的平均
    - 'bi':      上面两者的平均
    """
    def _mean_min_dist(a: torch.Tensor, b: torch.Tensor) -> float:
        # (1, Na, Nb) 的成对距离矩阵（L2）
        d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2)
        # 对 b 取最小，再对所有点取均值 -> 标量
        return d.min(dim=-1).values.mean().item()

    if direction == 'x_to_y':
        return _mean_min_dist(x, y)
    elif direction == 'y_to_x':
        return _mean_min_dist(y, x)
    elif direction == 'bi':
        return 0.5 * (_mean_min_dist(x, y) + _mean_min_dist(y, x))
    else:
        raise ValueError("Invalid direction type. Supported: 'y_to_x', 'x_to_y', 'bi'")

def chamfer_distance_max(x: torch.Tensor, y: torch.Tensor, direction: str = 'bi') -> float:
    """
    x, y: (N,3) / (M,3) 的 torch.Tensor
    返回值为 Python float(便于与阈值比较)
    方向语义与原实现一致：
    - 'x_to_y':  x 中每个点到 y 的最近距离的平均
    - 'y_to_x':  y 中每个点到 x 的最近距离的平均
    - 'bi':      上面两者的平均
    """
    def _mean_min_dist(a: torch.Tensor, b: torch.Tensor) -> float:
        # (1, Na, Nb) 的成对距离矩阵（L2）
        d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2)
        # 对 b 取最小，再对所有点取均值 -> 标量
        return d.min(dim=-1).values.max().item()

    if direction == 'x_to_y':
        return _mean_min_dist(x, y)
    elif direction == 'y_to_x':
        return _mean_min_dist(y, x)
    elif direction == 'bi':
        return 0.5 * (_mean_min_dist(x, y) + _mean_min_dist(y, x))
    else:
        raise ValueError("Invalid direction type. Supported: 'y_to_x', 'x_to_y', 'bi'")
    

def global_scale_min_t(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12):
    assert A.dim() == 2 and B.dim() == 2 and A.size(1) in (2, 3) and B.size(1) == A.size(1)
    A_min, A_max = A.min(dim=0).values, A.max(dim=0).values
    B_min, B_max = B.min(dim=0).values, B.max(dim=0).values
    sA = torch.clamp(torch.norm(A_max - A_min), min=eps)
    sB = torch.clamp(torch.norm(B_max - B_min), min=eps)
    s = torch.minimum(sA, sB)
    return A / s, B / s, sA, sB




def _unit_tangent_at_t(P: torch.Tensor,
                       t: torch.Tensor,
                       u: torch.Tensor,
                       t0: torch.Tensor,
                       degree: int = 3,
                       reg: float = 1e-6,
                       eps: float = 1e-8,
                       sigma: torch.Tensor | None = None) -> torch.Tensor:
    """
    已知点集P与其主轴参数t，以及想要求导的t0，返回在t0处的单位切向量。
    采用带权岭回归：权重 w_i = exp(-((t_i - t0)^2) / (2*sigma^2))
    """
    M = P.size(0)
    if M < 2:
        return torch.zeros(3, dtype=P.dtype, device=P.device)

    # 设计矩阵 T = [1, t, t^2, ..., t^deg]
    deg = int(max(1, degree))
    T = torch.stack([t**k for k in range(deg + 1)], dim=-1)  # (M, deg+1)

    # 权重（局部拟合），sigma 默认取 t 的尺度
    if sigma is None:
        # 用 IQR/范围 的一部分作为带宽，避免全局/过窄
        tr = (t.max() - t.min()).clamp_min(eps)
        sigma = 0.25 * tr
    w = torch.exp(-0.5 * ((t - t0) / (sigma + eps))**2)  # (M,)

    # 带权岭回归闭式解： (T^T W T + reg I)^{-1} T^T W P
    W = w.unsqueeze(-1)                                   # (M,1)
    TT = T.transpose(0, 1)                                # (deg+1, M)
    A = TT @ (W * T)                                      # (deg+1, deg+1)
    A = A + reg * torch.eye(deg + 1, dtype=T.dtype, device=T.device)
    B = TT @ (W * P)                                      # (deg+1, 3)
    coeff = torch.linalg.solve(A, B)                      # (deg+1, 3)

    # 在 t0 处求导：r'(t0) = [c1 + 2 c2 t0 + 3 c3 t0^2 + ...]
    deriv_vec = torch.zeros(deg + 1, dtype=T.dtype, device=T.device)
    if deg >= 1:
        powers = torch.stack([t0**(k-1) for k in range(1, deg + 1)], dim=0)  # (deg,)
        ks = torch.arange(1, deg + 1, device=T.device, dtype=T.dtype)
        deriv_vec[1:] = ks * powers
    v = coeff.transpose(0, 1) @ deriv_vec  # (3,)

    nrm = v.norm()
    if nrm < eps:
        # 退化：回退到主轴方向
        v = u
        nrm = v.norm()
        if nrm < eps:
            return torch.zeros(3, dtype=P.dtype, device=P.device)

    return v / (nrm + eps)


def _curve_unit_tangent_at_index(P: torch.Tensor,
                                 idx: int,
                                 degree: int = 3,
                                 reg: float = 1e-6,
                                 eps: float = 1e-8) -> torch.Tensor:
    """
    在“最近点索引 idx ”处求单位切向量：
    1) 以第一主轴做 1D 参数化 t
    2) 在 t0 = t[idx] 处做局部加权多项式拟合并求导
    """
    assert P.dim() == 2 and P.size(-1) == 3
    P = P.to(torch.float32)
    if P.size(0) < 2:
        return torch.zeros(3, dtype=P.dtype, device=P.device)

    # 去中心化 + 主轴
    Pc = P - P.mean(dim=0, keepdim=True)
    if torch.linalg.norm(Pc) < eps:
        return torch.zeros(3, dtype=P.dtype, device=P.device)
    _, _, Vh = torch.linalg.svd(Pc, full_matrices=False)
    u = Vh[0]
    u = u / (u.norm() + eps)

    # 主轴参数化 & 取该点的 t0
    t = Pc @ u  # (M,)
    t0 = t[idx]

    # 在 t0 处的单位切向量（局部加权拟合）
    vhat = _unit_tangent_at_t(P, t, u, t0, degree=degree, reg=reg, eps=eps)
    return vhat


def cos_dist_curve(P: torch.Tensor,
                              Q: torch.Tensor,
                              degree: int = 3,
                              reg: float = 1e-6,
                              eps: float = 1e-8) -> torch.Tensor:
    """
    1) 找到 P 与 Q 的最近点对 (i*, j*)
    2) 分别在 P[i*]、Q[j*] 处估计单位切向量
    3) 返回两切向量的无符号余弦相似度 ∈ [0,1]
    """
    assert P.dim() == 2 and P.size(-1) == 3 and Q.dim() == 2 and Q.size(-1) == 3
    P = P.to(torch.float32)
    Q = Q.to(torch.float32)
    if P.size(0) < 2 or Q.size(0) < 2:
        return P.new_tensor(0.0)

    # 最近点对
    D = torch.cdist(P, Q, p=2)  # (M, N)
    # 直接取全局最小
    flat_idx = torch.argmin(D)
    i_star = (flat_idx // D.size(1)).item()
    j_star = (flat_idx %  D.size(1)).item()

    # 各自在最近点处的单位切向量
    vP = _curve_unit_tangent_at_index(P, i_star, degree=degree, reg=reg, eps=eps)
    vQ = _curve_unit_tangent_at_index(Q, j_star, degree=degree, reg=reg, eps=eps)

    if (vP.abs().sum() < eps) or (vQ.abs().sum() < eps):
        return P.new_tensor(0.0)

    sim = torch.dot(vP, vQ).clamp(-1.0, 1.0).abs()
    return sim