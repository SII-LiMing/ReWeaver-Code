import numpy as np
from pathlib import Path
import json
import cv2
import sys

from scipy.spatial import cKDTree

from reorder_edges import reorder_and_flip_edges

from datetime import datetime

def _safe_stats(x, *, filter_fn=None):
    """
    返回 (mean, var, std, n)。方差为样本方差(ddof=1)；当 n<2 时方差/std 置 0。
    可选 filter_fn 对原列表做过滤（如过滤负值）。
    """
    if filter_fn is not None:
        x = [v for v in x if filter_fn(v)]
    arr = np.asarray(x, dtype=float)
    # 清理 nan
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(arr.mean())
    if n > 1:
        var = float(arr.var(ddof=1))
        std = float(np.sqrt(var))
    else:
        var, std = 0.0, 0.0
    return mean, var, std, n

def pc_matching(gt_patch_pts,pred_patch_pts):
    from scipy.spatial import cKDTree
    num_gt=len(gt_patch_pts)
    num_pred=len(pred_patch_pts)
    cost_matrix=np.zeros((num_gt,num_pred))
    for i in range(num_gt):
        gt_pc=gt_patch_pts[i]
        gt_kd= cKDTree(gt_pc)
        for j in range(num_pred):
            pred_pc=pred_patch_pts[j]
            dist, _ = gt_kd.query(pred_pc, k=1)
            cost_matrix[i,j]=np.mean(dist)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matching_cost=cost_matrix[row_ind, col_ind].sum()
    return row_ind, col_ind, matching_cost, cost_matrix



def pc_f1_score(gt_patch_pts,pred_patch_pts,threshold):
    from scipy.spatial import cKDTree
    total_precision=0.0
    total_recall=0.0
    for i in range(len(gt_patch_pts)):
        gt_pc=gt_patch_pts[i]
        pred_pc=pred_patch_pts[i]
        gt_kd= cKDTree(gt_pc)
        pred_kd= cKDTree(pred_pc)

        dist1, _ = gt_kd.query(pred_pc, k=1)
        dist2, _ = pred_kd.query(gt_pc, k=1)

        precision= np.sum(dist1 < threshold) / len(pred_pc)
        recall= np.sum(dist2 < threshold) / len(gt_pc)

        total_precision+=precision
        total_recall+=recall

    total_precision=total_precision/len(gt_patch_pts)
    total_recall=total_recall/len(gt_patch_pts)
    f1_score=2*total_precision*total_recall/(total_precision+total_recall+1e-8)
    return f1_score

def curve_f1_score(gt_curves,pred_curves,threshold):    
    from scipy.spatial import cKDTree
    total_precision=0.0
    total_recall=0.0
    for i in range(len(gt_curves)):
        gt_curve=gt_curves[i]
        pred_curve=pred_curves[i]
        gt_kd= cKDTree(gt_curve)
        pred_kd= cKDTree(pred_curve)

        dist1, _ = gt_kd.query(pred_curve, k=1)
        dist2, _ = pred_kd.query(gt_curve)

        precision= np.sum(dist1 < threshold) / len(pred_curve)
        recall= np.sum(dist2 < threshold) / len(gt_curve)

        total_precision+=precision
        total_recall+=recall

    total_precision=total_precision/len(gt_curves)
    total_recall=total_recall/len(gt_curves)
    f1_score=2*total_precision*total_recall/(total_precision+total_recall+1e-8)
    return f1_score




def _ensure_closed(poly: np.ndarray) -> np.ndarray:
    """若首尾未闭合，则补上首点；输入形状 (N,2)."""
    if poly.shape[0] >= 3 and not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly

def _two_polys_to_masks(gt_xy: np.ndarray, pred_xy: np.ndarray, pad: int = 2, y_up: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    将两组 2D 顶点坐标（闭合多边形）栅格化到同一“正方形”画布，输出 0/1 掩膜。
    画布边长取联合外接框的长边 + 2*pad，并在短边方向做对称留白（不缩放、不拉伸）。
    """
    gt_xy   = _ensure_closed(gt_xy.astype(np.float32))
    pred_xy = _ensure_closed(pred_xy.astype(np.float32))

    # 联合外接框
    all_xy = np.vstack([gt_xy, pred_xy])
    mins = np.floor(all_xy.min(axis=0)).astype(int)   # (xmin, ymin)
    maxs = np.ceil(all_xy.max(axis=0)).astype(int)    # (xmax, ymax)

    # 原始所需画布尺寸（未正方形前）
    w0 = int(max(1, (maxs[0] - mins[0]) + 1 + 2 * pad))
    h0 = int(max(1, (maxs[1] - mins[1]) + 1 + 2 * pad))

    # 统一为正方形：边长取长边
    size = max(w0, h0)

    # 把坐标平移到画布内（基础平移：左上角齐到 pad 处）
    base_shift = mins - pad          # 被减去的量
    gt_shift   = gt_xy   - base_shift
    pred_shift = pred_xy - base_shift

    if y_up:
        gt_shift[:, 1]   = (h0 - 1) - gt_shift[:, 1]
        pred_shift[:, 1] = (h0 - 1) - pred_shift[:, 1]

    # 为了在正方形画布中居中，对短边方向再加对称留白偏移
    dx = (size - w0) // 2   # 宽方向需要额外留白的一半
    dy = (size - h0) // 2   # 高方向需要额外留白的一半

    shift2 = np.array([dx, dy], dtype=np.float32)
    gt_shift   = gt_shift   + shift2
    pred_shift = pred_shift + shift2

    # 光栅化为 0/1 掩膜（正方形画布）
    mask_shape = (size, size)
    gt_mask   = np.zeros(mask_shape, dtype=np.uint8)
    pred_mask = np.zeros(mask_shape, dtype=np.uint8)

    cv2.fillPoly(gt_mask,   [np.round(gt_shift).astype(np.int32)], 1)
    cv2.fillPoly(pred_mask, [np.round(pred_shift).astype(np.int32)], 1)

    return gt_mask, pred_mask

def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a == 1, mask_b == 1).sum()
    union = np.logical_or(mask_a == 1, mask_b == 1).sum()
    return float(inter) / float(union) if union > 0 else 0.0



def vis_mask(gt_mask: np.ndarray, pred_mask: np.ndarray, save_dir: str = None, prefix: str = ""):
    """
    可视化两个 0/1 掩膜：
      - gt.png / pred.png：灰度图（0/255）
      - overlay.png：GT=绿、Pred=红、重叠=黄
      - side_by_side.png：GT | Pred | Overlay 三联图

    Args:
        gt_mask   : (H, W) uint8 / bool, 0/1
        pred_mask : (H, W) uint8 / bool, 0/1
        save_dir  : 可选，若给出则保存到该目录
        prefix    : 可选，文件名前缀（如 panel_id）

    Returns:
        dict: {"gt": gt_bgr, "pred": pred_bgr, "overlay": overlay, "side": side}
              均为 BGR uint8 图像数组
    """
    assert gt_mask.shape == pred_mask.shape, "gt_mask 与 pred_mask 尺寸不一致"
    h, w = gt_mask.shape[:2]

    # 统一为 0/255 的 uint8
    gt_u8   = (gt_mask.astype(np.uint8) * 255) if gt_mask.max() <= 1 else gt_mask.astype(np.uint8)
    pred_u8 = (pred_mask.astype(np.uint8) * 255) if pred_mask.max() <= 1 else pred_mask.astype(np.uint8)

    # 单图转 BGR（便于拼接）
    gt_bgr   = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_u8, cv2.COLOR_GRAY2BGR)

    # 叠加：G=GT，R=Pred（重叠→黄）
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[..., 1] = gt_u8    # G
    overlay[..., 2] = pred_u8  # R

    # 三联图
    side = np.concatenate([gt_bgr, pred_bgr, overlay], axis=1)

    # 选配：描边（便于看边界），默认关闭；需要时取消注释
    # contours_gt, _ = cv2.findContours((gt_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours_pd, _ = cv2.findContours((pred_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(overlay, contours_gt, -1, (0, 255, 0), 1)   # 绿边
    # cv2.drawContours(overlay, contours_pd, -1, (0, 0, 255), 1)   # 红边
    # side = np.concatenate([gt_bgr, pred_bgr, overlay], axis=1)   # 重画 side

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        pre = (prefix + "_") if prefix else ""
        cv2.imwrite(str(out / f"{pre}gt.png"), gt_u8)
        cv2.imwrite(str(out / f"{pre}pred.png"), pred_u8)
        cv2.imwrite(str(out / f"{pre}overlay.png"), overlay)
        cv2.imwrite(str(out / f"{pre}side_by_side.png"), side)

    return {"gt": gt_bgr, "pred": pred_bgr, "overlay": overlay, "side": side}



def _fftconv2_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D full 卷积（相关）实现：返回形状 (Ha+Hb-1, Wa+Wb-1)。"""
    Ha, Wa = a.shape
    Hb, Wb = b.shape
    H = Ha + Hb - 1
    W = Wa + Wb - 1
    # 零填充到 >= H, W 的尺寸（用下一个 2 的幂可更快，但 numpy 也够用）
    FH = 1 << (H - 1).bit_length()
    FW = 1 << (W - 1).bit_length()
    Fa = np.fft.rfftn(a, s=(FH, FW))
    Fb = np.fft.rfftn(b, s=(FH, FW))
    F = Fa * Fb
    out = np.fft.irfftn(F, s=(FH, FW))[:H, :W]
    return np.maximum(out, 0.0)  # 数值误差修正

def maximize_iou_by_translation(maskA: np.ndarray, maskB: np.ndarray, max_shift: int = None):
    """
    在所有整像素平移 (dy, dx) 下，使 IoU 最大。
    返回 best_iou, best_dy, best_dx。
    可选 max_shift 限制 |dy|,|dx| 的搜索半径（不填则全局）。
    """
    A = (maskA > 0).astype(np.float32)
    B = (maskB > 0).astype(np.float32)
    Sa, Sb = A.sum(), B.sum()
    if Sa == 0 and Sb == 0:
        return 1.0, 0, 0
    if Sa == 0 or Sb == 0:
        return 0.0, 0, 0

    # inter(dy,dx) = sum A * shift(B, dy,dx)
    # 用相关：conv2(A, flip(B)) == corr2(A,B)
    inter_full = _fftconv2_full(A, B[::-1, ::-1])  # 形状 (Ha+Hb-1, Wa+Wb-1)

    # union(dy,dx) = Sa + Sb - inter
    union_full = Sa + Sb - inter_full
    iou_full = inter_full / np.maximum(union_full, 1e-9)

    # 平移索引 -> (dy, dx)：中心点对应 (Hb-1, Wb-1)
    center = (B.shape[0] - 1, B.shape[1] - 1)

    if max_shift is not None:
        H, W = iou_full.shape
        yy, xx = np.ogrid[:H, :W]
        dy = yy - center[0]
        dx = xx - center[1]
        mask = (np.abs(dy) <= max_shift) & (np.abs(dx) <= max_shift)
        # 不在窗口内的设为 -1，避免选到
        iou_full = np.where(mask, iou_full, -1.0)

    idx = np.unravel_index(np.argmax(iou_full), iou_full.shape)
    best_dy = int(idx[0] - center[0])
    best_dx = int(idx[1] - center[1])
    best_iou = float(iou_full[idx])
    return best_iou, best_dy, best_dx

def shift_mask_zp(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """零填充的平移（不环绕）。返回与输入相同大小。"""
    H, W = mask.shape
    out = np.zeros_like(mask)
    ys = max(0, dy)
    xs = max(0, dx)
    yt = max(0, -dy)
    xt = max(0, -dx)
    h = min(H - ys, H - yt)
    w = min(W - xs, W - xt)
    if h > 0 and w > 0:
        out[ys:ys+h, xs:xs+w] = mask[yt:yt+h, xt:xt+w]
    return out


def place_on_new_canvas(maskA: np.ndarray, maskB: np.ndarray, dy: int, dx: int, pad: int = 0):
    """
    将 A 保持原位(参考坐标原点)，将 B 以 (dy, dx) 的整数位移放到同一“新”画布中。
    新画布尺寸恰好覆盖两图（可加 pad）。不丢失任何超出原边界的像素。
    返回: A2, B2, (offset_y, offset_x)
    """
    Ha, Wa = maskA.shape
    Hb, Wb = maskB.shape

    top    = min(0, dy)
    left   = min(0, dx)
    bottom = max(Ha, dy + Hb)
    right  = max(Wa, dx + Wb)

    H = (bottom - top) + 2 * pad
    W = (right  - left) + 2 * pad

    oy = -top + pad  # A 放置偏移
    ox = -left + pad

    A2 = np.zeros((H, W), dtype=maskA.dtype)
    B2 = np.zeros((H, W), dtype=maskB.dtype)

    # 放置 A（原位）
    A2[oy:oy+Ha, ox:ox+Wa] = maskA

    # 放置 B（平移后）
    by = oy + dy
    bx = ox + dx
    B2[by:by+Hb, bx:bx+Wb] = maskB

    return A2, B2, (oy, ox)

class Metric:
    def __init__(self,gt_root,pred_root,name):
        if isinstance(gt_root,str):
            gt_root=Path(gt_root)
        if isinstance(pred_root,str):
            pred_root=Path(pred_root)

        self.name=name
        

        path=gt_root/name/f"{name}_2d_panel.json"
        pattern=json.load(open(path,"r"))
        panel_order=pattern['panel_order']
        gt_panels={}
        for p in pattern['panels']:
            gt_panels[int(p)]=pattern['panels'][str(p)]

        path=gt_root/name/f"{name}_3d_geo.npz"
        geo_3d_npz=np.load(path)
        pc_sampled=geo_3d_npz["pc_sampled"]
        gt_curve_points=geo_3d_npz["curves_sampled"]
        pc_labels=geo_3d_npz["pc_labels"]
        gt_PC_connectivity=geo_3d_npz["PC_mat"]
        PC_mean=geo_3d_npz["pc_mean"]
        PC_scale=geo_3d_npz["pc_scale"]

        pc_labels_unique=np.unique(pc_labels)
        pc_labels_unique=pc_labels_unique[pc_labels_unique>=0]
        
        gt_patch_pts=[]
        for label in pc_labels_unique:
            # print("label:", label)
            pc_part=pc_sampled[pc_labels==label]
            # print("pc_part shape:", pc_part.shape)
            gt_patch_pts.append(pc_part)
        

        # pred_path=pred_root/name/f"{name}_processed.npz"
        pred_path=pred_root/name/f"{name}.npz"
        npz=np.load(pred_path,allow_pickle=True)
        patch_curve_similarity=npz["patch_curve_similarity"]
        pred_PC_connectivity=npz["patch_curve_connectivity"]
        pred_curve_points=npz["curve_points"]
        curve_valid_prob=npz["curve_valid_prob"]
        pred_patch_pts=npz["patch_points"]
        patch_valid_prob=npz["patch_valid_prob"]
        flatten_pred=npz["flatten_pred"].item()
        pred_patch_pts=[p for p in pred_patch_pts]
        
        pred_patch_pts_scaled=npz["patch_points_scaled"]
        
        gt_idx, pred_idx, matching_cost, cost_matrix=pc_matching(gt_patch_pts,pred_patch_pts)
        gt_patch_pts=[gt_patch_pts[i] for i in gt_idx]
        pred_patch_pts=[pred_patch_pts[i] for i in pred_idx]
        pred_patch_pts_scaled=[pred_patch_pts_scaled[i] for i in pred_idx]
        
        pred_PC_connectivity=pred_PC_connectivity[pred_idx]
        gt_PC_connectivity=gt_PC_connectivity[gt_idx]

        new_flatten_pred={}
        for i in range(len(pred_idx)):
            new_idx=i
            old_idx=pred_idx[i]
            new_flatten_pred[new_idx]=flatten_pred[old_idx]
            new_flatten_pred[new_idx]["old_idx"]=old_idx
        

        new_gt_panels={}
        for i in range(len(gt_idx)):
            new_idx=i
            old_idx=gt_idx[i]
            new_gt_panels[new_idx]=gt_panels[old_idx]
            new_gt_panels[new_idx]["old_idx"]=old_idx


        self.gt_patch_label_unique=pc_labels_unique
        self.gt_PC_mean=PC_mean
        self.gt_PC_scale=PC_scale
        self.gt_curve_pts=gt_curve_points
        self.gt_patch_pts=gt_patch_pts
        self.gt_PC_connectivity=gt_PC_connectivity
        self.gt_panels=new_gt_panels

        self.pred_PC_connectivity=pred_PC_connectivity
        self.pred_patch_pts=pred_patch_pts
        self.pred_curve_pts=pred_curve_points
        self.pred_panels=new_flatten_pred
        

        self.pred_patch_pts_scaled=pred_patch_pts_scaled
        
        
    def cal_panel_acc(self):
        pred_panel_num=len(self.pred_patch_pts)
        gt_panel_num=len(self.gt_patch_label_unique)
        
        if pred_panel_num==gt_panel_num:
            return 1
        else:
            return 0
    
    def cal_scale_l2(self):
        if self.cal_panel_acc()==0:
            return -1
        
        pred_panels=self.pred_panels
        gt_panels=self.gt_panels
        
        panel_num=len(gt_panels)
        
        total_scale_l2=0
        for i in range(panel_num):
            scale_l2=(pred_panels[i]["scale_pred"]-gt_panels[i]['scale'])**2
            total_scale_l2+=scale_l2
        return total_scale_l2/panel_num
    
    def cal_edge_acc(self):
        if self.cal_panel_acc()==0:
            return -1
        
        pred_PC_connectivity=self.pred_PC_connectivity
        gt_PC_connectivity=self.gt_PC_connectivity

        assert gt_PC_connectivity.shape[0]==self.pred_PC_connectivity.shape[0],"panel number not match"
        panel_num=gt_PC_connectivity.shape[0]

        gt_num=np.sum(gt_PC_connectivity,axis=1)
        pred_num=np.sum(pred_PC_connectivity,axis=1)

        correct_num=(gt_num==pred_num).sum()
        acc=correct_num/panel_num
        return acc

    def cal_edge_cd(self):

        if self.cal_panel_acc()==0:
            return -1
        
        pred_panels = self.pred_panels
        gt_panels = self.gt_panels
        panel_num=len(gt_panels)

        total_cd=0.0
        for p_id in gt_panels:
            gt_edges=gt_panels[p_id]['edge_points']
            pred_edges=pred_panels[p_id]['edge_points']

            gt_edges=np.array(gt_edges).reshape(-1,2)
            pred_edges=np.array(pred_edges).reshape(-1,2)

            # 计算 gt_edges 和 pred_edges 之间的 Chamfer Distance
            gt_kd= cKDTree(gt_edges)
            pred_kd= cKDTree(pred_edges)

            dist1, _ = gt_kd.query(pred_edges, k=1)
            dist2, _ = pred_kd.query(gt_edges)
            cd=(np.mean(dist1)+np.mean(dist2))/2.0
            total_cd+=cd
        total_cd=total_cd/panel_num
        return total_cd

    
    def cal_panel_iou(self):
        if self.cal_panel_acc() == 0:
            return -1

        pred_panels = self.pred_panels
        gt_panels = self.gt_panels
        panel_num = len(gt_panels)
        total_iou = 0.0

        for p_id in gt_panels.keys():
            
            try:
                gt_edges = np.array(gt_panels[p_id]['edge_points'])
                pred_edges = np.array(pred_panels[p_id]['edge_points'])

                gt_edges, *_ = reorder_and_flip_edges(gt_edges)
                pred_edges, *_ = reorder_and_flip_edges(pred_edges)
                gt_edges = gt_edges.reshape(-1, 2) * 100
                pred_edges = pred_edges.reshape(-1, 2) * 100

                # 直接同画布生成掩码，不做 translation
                gt_mask, pred_mask = _two_polys_to_masks(gt_edges, pred_edges, pad=2)

                inter = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
                union = np.logical_or(gt_mask > 0, pred_mask > 0).sum()
                iou = float(inter) / float(union) if union > 0 else 0.0

                total_iou += iou
            except:
                total_iou += 0

        return total_iou / max(panel_num, 1)

    def cal_patch_cd(self):
        
        pred_patch_pts=self.pred_patch_pts
        gt_patch_pts=self.gt_patch_pts
        pred_pc=np.concatenate(pred_patch_pts,axis=0)
        gt_pc=np.concatenate(gt_patch_pts,axis=0)

        gt_kd= cKDTree(gt_pc)
        pred_kd= cKDTree(pred_pc)

        dist1, _ = gt_kd.query(pred_pc, k=1)
        dist2, _ = pred_kd.query(gt_pc, k=1)
        cd=(np.mean(dist1)+np.mean(dist2))/2.0
        return cd
    
    def cal_patch_cd_scaled(self):
        
        pred_patch_pts=self.pred_patch_pts_scaled
        gt_patch_pts=self.gt_patch_pts
        pred_pc=np.concatenate(pred_patch_pts,axis=0)
        gt_pc=np.concatenate(gt_patch_pts,axis=0)

        gt_kd= cKDTree(gt_pc)
        pred_kd= cKDTree(pred_pc)

        dist1, _ = gt_kd.query(pred_pc, k=1)
        dist2, _ = pred_kd.query(gt_pc, k=1)
        cd=(np.mean(dist1)+np.mean(dist2))/2.0
        return cd
    
    def cal_curve_cd(self):
            
        pred_curve_pts=self.pred_curve_pts.reshape(-1,3)
        gt_curve_pts=self.gt_curve_pts.reshape(-1,3)


        gt_kd= cKDTree(gt_curve_pts)
        pred_kd= cKDTree(pred_curve_pts)

        dist1, _ = gt_kd.query(pred_curve_pts, k=1)
        dist2, _ = pred_kd.query(gt_curve_pts, k=1)
        cd=(np.mean(dist1)+np.mean(dist2))/2.0
        return cd
    
    def cal_F_score(self, tau=0.07, use_scipy: bool = True):
        """
        计算两个点云的 F-score / precision / recall
        参数:
            P: (Np, 3) 预测点云
            G: (Ng, 3) GT 点云
            tau: 距离阈值（与坐标单位一致）
            use_scipy: 若为 True，使用 scipy.spatial.cKDTree（推荐）
        返回:
            fscore, precision, recall
        """

        P=self.pred_patch_pts
        P=np.concatenate(P,axis=0)
        G=self.gt_patch_pts
        G=np.concatenate(G,axis=0)
        assert P.ndim == 2 and P.shape[1] == 3 and G.ndim == 2 and G.shape[1] == 3, "输入需为(N,3)"
        Np, Ng = len(P), len(G)
        if Np == 0 or Ng == 0:
            # 约定：若一方为空，precision/recall 为 0，F-score 为 0
            return 0.0, 0.0, 0.0

        tau2 = tau * tau

        if use_scipy:
            try:
                from scipy.spatial import cKDTree
                tree_G = cKDTree(G)
                dist_P2G, _ = tree_G.query(P, k=1, n_jobs=-1)
                # 反向
                tree_P = cKDTree(P)
                dist_G2P, _ = tree_P.query(G, k=1, n_jobs=-1)
            except Exception:
                # 回退到纯 NumPy
                use_scipy = False

        if not use_scipy:
            # 纯 NumPy 最近邻（O(Np*Ng)），小点云可用
            # P->G
            # 为避免内存爆炸，可分块；这里给简单实现
            dist_P2G = np.sqrt(((P[:, None, :] - G[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            # G->P
            dist_G2P = np.sqrt(((G[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)).min(axis=1)

        precision = float((dist_P2G ** 2 <= tau2).sum()) / Np
        recall    = float((dist_G2P ** 2 <= tau2).sum()) / Ng

        if precision + recall == 0:
            fscore = 0.0
        else:
            fscore = 2.0 * precision * recall / (precision + recall)

        return fscore, precision, recall


        
if __name__=="__main__":
    from tqdm import tqdm


    gt_root=Path(f"/inspire/hdd/global_user/liming-253108120187/Datasets/GCD_TS/GCD_TS/test")
    pred_root=Path(f"/inspire/hdd/global_user/liming-253108120187/GarmentRecon/Model_final/saved/final_ori/pred/test")
    
    names=list(pred_root.iterdir())
    names=[n.name for n in names]
    
    
    
    datapoiont_num=0
    panel_acc_lst=[]
    edge_acc_lst=[]
    edge_cd_lst=[]
    panel_iou_lst=[]
    patch_cd_lst=[]
    patch_cd_scaled_lst=[]
    curve_cd_lst=[]
    f_score_lst=[]
    scale_l2_lst=[]
    pbar = tqdm(names, dynamic_ncols=True, desc="Evaluating", leave=True)
    for name in pbar:
        data = Metric(gt_root, pred_root, name)
        panel_acc = data.cal_panel_acc()
        edge_acc  = data.cal_edge_acc()
        edge_cd   = data.cal_edge_cd()
        panel_iou = data.cal_panel_iou()
        patch_cd  = data.cal_patch_cd()
        patch_cd_scaled = data.cal_patch_cd_scaled()
        curve_cd  = data.cal_curve_cd()
        scale_l2 = data.cal_scale_l2()
        
        panel_acc_lst.append(panel_acc)
        edge_acc_lst.append(edge_acc)
        edge_cd_lst.append(edge_cd)
        panel_iou_lst.append(panel_iou)
        patch_cd_lst.append(patch_cd)
        curve_cd_lst.append(curve_cd)
        patch_cd_scaled_lst.append(patch_cd_scaled)
        scale_l2_lst.append(scale_l2)
        
        datapoiont_num += 1

        # ===== 实时进度条后缀（轻量口径：显示若干均值）=====
        # 用你已有的 _safe_stats 计算“过滤版”均值，避免负值干扰
        pa_mean, *_ = _safe_stats(panel_acc_lst)
        ea_mean, *_ = _safe_stats(edge_acc_lst,  filter_fn=lambda v: v >= 0)
        ec_mean, *_ = _safe_stats(edge_cd_lst,   filter_fn=lambda v: v >= 0)
        pi_mean, *_ = _safe_stats(panel_iou_lst, filter_fn=lambda v: v >= 0)
        pc_mean, *_ = _safe_stats(patch_cd_lst)
        cc_mean, *_ = _safe_stats(curve_cd_lst)
        sl_mean, *_ = _safe_stats(scale_l2_lst, filter_fn=lambda v: v >= 0)

        # 进度条简洁展示；数值用 .3g 以避免过长
        pbar.set_postfix({
            "N": datapoiont_num,
            "panel_acc": f"{pa_mean:.3g}" if np.isfinite(pa_mean) else "nan",
            "edge_acc":  f"{ea_mean:.3g}" if np.isfinite(ea_mean) else "nan",
            "edge_cd":   f"{ec_mean:.3g}" if np.isfinite(ec_mean) else "nan",
            "panel_iou": f"{pi_mean:.3g}" if np.isfinite(pi_mean) else "nan",
            "patch_cd":  f"{pc_mean:.3g}" if np.isfinite(pc_mean) else "nan",
            "curve_cd":  f"{cc_mean:.3g}" if np.isfinite(cc_mean) else "nan",
            "scale_l2":  f"{sl_mean:.3g}" if np.isfinite(cc_mean) else "nan",
        })


    # ---------- 统计（均值/方差/标准差/数量） ----------
    panel_acc_mean, panel_acc_var, panel_acc_std, panel_acc_n = _safe_stats(panel_acc_lst)
    edge_acc_mean,  edge_acc_var,  edge_acc_std,  edge_acc_n  = _safe_stats(edge_acc_lst, filter_fn=lambda v: v >= 0)

    patch_cd_mean, patch_cd_var, patch_cd_std, patch_cd_n = _safe_stats(patch_cd_lst)
    curve_cd_mean, curve_cd_var, curve_cd_std, curve_cd_n = _safe_stats(curve_cd_lst)
    patch_cd_scaled_mean, patch_cd_scaled_var, patch_cd_scaled_std, patch_cd_scaled_n = _safe_stats(patch_cd_scaled_lst)

    edge_cd_mean, edge_cd_var, edge_cd_std, edge_cd_n = _safe_stats(edge_cd_lst, filter_fn=lambda v: v >= 0)
    panel_iou_mean ,panel_iou_var, panel_iou_std, panel_iou_n = _safe_stats(panel_iou_lst,filter_fn=lambda v: v >= 0)
    
    scale_l2_mean, scale_l2_var, scale_l2_std, scale_l2_n = _safe_stats(scale_l2_lst, filter_fn=lambda v: v >= 0)
    # 可选：组合指标（这里仍按你的口径：面板精度×边精度的乘积）
    # 组合均值直乘（不是严格的期望乘积，但与原口径一致）
    acc_combo_mean = panel_acc_mean * edge_acc_mean



    # ---------- 写日志（追加模式），更详细 ----------
    with open("log", "a", encoding="utf-8") as f:
        f.write(f"===== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        f.write("[Patch CD @ base(400)]\n")
        f.write(f"  count={patch_cd_n}, mean={patch_cd_mean:.6g}, var={patch_cd_var:.6g}, std={patch_cd_std:.6g}\n")
        f.write("[Patch CD @ adaptive]\n")
        f.write(f"  count={patch_cd_scaled_n}, mean={patch_cd_scaled_mean:.6g}, var={patch_cd_scaled_var:.6g}, std={patch_cd_scaled_std:.6g}\n")
        f.write("[Curve CD]\n")
        f.write(f"  count={curve_cd_n}, mean={curve_cd_mean:.6g}, var={curve_cd_var:.6g}, std={curve_cd_std:.6g}\n")
        f.write("[Panel Accuracy]\n")
        f.write(f"  count={panel_acc_n}, mean={panel_acc_mean:.6g}, var={panel_acc_var:.6g}, std={panel_acc_std:.6g}\n")
        f.write("[Edge Accuracy]\n")
        f.write(f"  (filtered >=0) count={edge_acc_n}, mean={edge_acc_mean:.6g}, var={edge_acc_var:.6g}, std={edge_acc_std:.6g}\n")
        f.write("[Combined Acc = panel_acc_mean * edge_acc_mean]\n")
        f.write(f"  mean={acc_combo_mean:.6g}\n")
        f.write("[Edge CD]\n")
        f.write(f"  count={edge_cd_n}, mean={edge_cd_mean:.6g}, var={edge_cd_var:.6g}, std={edge_cd_std:.6g}\n")
        f.write("[Panel IoU]\n")
        f.write(f"  count={panel_iou_n}, mean={panel_iou_mean:.6g}, var={panel_iou_var:.6g}, std={panel_iou_std:.6g}\n")
        f.write("[Scale L2]\n")
        f.write(f"  count={scale_l2_n}, mean={scale_l2_mean:.6g}, var={scale_l2_var:.6g}, std={scale_l2_std:.6g}\n")
        f.write("\n")
            
            
            



