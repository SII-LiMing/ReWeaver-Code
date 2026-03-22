import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FlattenConfig
from models.core_module import MLP_hn, MLP
from torch.nn.utils.rnn import pad_sequence
from utils.get_boundary import get_boundary
from scipy.special import softmax
from utils.linear_alg import r2_linear,scale_point_cloud_to_aabb,apply_affine_from_info,chamfer_distance,global_scale_min_t,cos_dist_curve,chamfer_distance_max
import math
from typing import Optional, Tuple, List


def index_b_in_a(a,b):
    # 长：a
    # 短：b

    idxs = []
    for v in b:
        idxs.append((a == v).nonzero(as_tuple=True)[0].item())

    idxs = torch.tensor(idxs, device=a.device)
    return idxs


def rigid_transform_3D(P, Q):
    """
    P, Q: shape (N, 3), P -> Q
    Return: R (3x3), T (3,)
    """
    H = P.T @ Q
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

class TransformNet(nn.Module):
    def __init__(self, k=3):
        super(TransformNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # 初始化为单位矩阵
        self.iden = torch.eye(k).view(1, k * k)

    def forward(self, x):
        # x: B, k, N
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # B,1024,N
        x = torch.max(x, 2)[0]               # B,1024

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                      # B,9

        iden = self.iden.to(x.device).repeat(batch_size, 1)
        x = x + iden                         # Add identity
        x = x.view(batch_size, self.k, self.k)
        return x


class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim=None, dropout=0.0, activation='relu'):
        """
        参数:
            embed_dim: 输入输出的维度（通常为 transformer 的 hidden size)
            ffn_dim: 隐藏层维度（默认 4*embed_dim)
            dropout: dropout 概率
            activation: 激活函数类型，支持 'relu' 或 'gelu'
        """
        super(FFN, self).__init__()
        ffn_dim = ffn_dim or embed_dim * 4

        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# for early version debug
# class KeypointsEncoder(nn.Module):
#     def __init__(self, args):
#         super(KeypointsEncoder, self).__init__()
#         self.fc1 = nn.Linear(3, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 768)

#         self.activation = nn.ReLU()
#         self.norm = nn.LayerNorm(128)

#     def forward(self,x ):
#         residual = self.activation(self.fc1(x))  # (B, T, hidden_dim)
#         out = self.fc2(residual)
#         out = self.norm(out + residual)          # 残差连接 + LayerNorm
#         out = self.activation(out)
#         out = self.fc3(out)                      # 最终输出为 3D
#         return out
    
# for early version debug
# class EdgeEncoder(nn.Module):
#     def __init__(self, args):
#         super(EdgeEncoder, self).__init__()
#         self.first_encoder = nn.Linear(150, 150)
#         self.fc1 = nn.Linear(150, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 768)

#         self.activation = nn.ReLU()
#         self.norm = nn.LayerNorm(512)

#     def forward(self, x):
#         """
#         输入:
#             x: (B, T, N, 3) —— token
#         输出:
#             (B, T, 3) —— 解码的3D点
#         """
#         B,T,N,D=x.shape
#         x_reverse = x[:, :, ::-1, :]  # (B, T, N, 2)
#         x_reverse=x_reverse.reshape(B,T,N*D)
#         x=x.reshape(B,T,N*D)
#         x=self.first_encoder(x)+self.first_encoder(x_reverse)/2  # (B, T, N*D)
#         residual = self.activation(self.fc1(x))  # (B, T, hidden_dim)
#         out = self.fc2(residual)
#         out = self.norm(out + residual)          # 残差连接 + LayerNorm
#         out = self.activation(out)
#         out = self.fc3(out)                      # 最终输出为 3D
#         return out


class EdgeDecoder(nn.Module):
    def __init__(self, args:FlattenConfig):
        super(EdgeDecoder, self).__init__()
        self.points_per_edge=args.points_per_edge
        curve_pe_tensor = (torch.arange(self.points_per_edge, dtype=torch.float32) / (self.points_per_edge - 1)).view(self.points_per_edge, 1)
        self.register_buffer("curve_pe", curve_pe_tensor)
        # self.classifier = MLP(768,768,5,3)
        self.curve_start_point_embed=MLP(768, 768, 2, 3)
        self.curve_shape_embed = MLP_hn(1, 64, 2, 3, 768)
        
    def forward(self, x):
        # logits = self.classifier(x)   
        outputs_start_point_coord = self.curve_start_point_embed(x).tanh()
        outputs_start_point_coord = outputs_start_point_coord.reshape(1,outputs_start_point_coord.shape[0],1, outputs_start_point_coord.shape[1]) 
        outputs_start_point_coord = outputs_start_point_coord.repeat(1,1,self.points_per_edge,1)
        
        sampled_points_feature = self.curve_pe.view(1,self.points_per_edge, 1).repeat(x.shape[0], 1, 1)
        
        edge_points = outputs_start_point_coord + self.curve_shape_embed(sampled_points_feature,x.unsqueeze(0))
        
        # return {"logits":logits,"edge_points":edge_points.squeeze(0)}
        return {"edge_points":edge_points.squeeze(0)}
        
class SelfAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        参数:
            embed_dim: 每个 token 的特征维度
            num_heads: 多头注意力的头数
            dropout: 注意力输出的 dropout 比例
        """
        super(SelfAttentionModule, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        参数:
            x: 输入张量，形状为 (B, T, C),B 是 batch size,T 是序列长度,C 是 embed_dim
            attn_mask: 可选的注意力掩码（用于控制 token 间注意力），形状为 (T, T)
            key_padding_mask: 可选的 padding 掩码，形状为 (B, T)

        返回:
            输出张量，形状为 (B, T, C)
        """
        # 自注意力（注意 batch_first=True）
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        # 残差连接 + 层归一化
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        参数:
            embed_dim: query 和 key/value 的特征维度（必须一致）
            num_heads: 多头注意力的头数
            dropout: 注意力 dropout 比例
        """
        super(CrossAttentionModule, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        """
        参数:
            query: (B, T_q, C) —— decoder 查询（例如当前解码的 token 序列）
            key_value: (B, T_kv, C) —— encoder 提供的上下文序列
            attn_mask: (T_q, T_kv) 可选注意力掩码
            key_padding_mask: (B, T_kv) 可选 padding 掩码

        返回:
            输出: (B, T_q, C)
        """
        attn_output, _ = self.cross_attn(query, key_value, key_value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = query + self.dropout(attn_output)  # 残差
        x = self.norm(x)
        return x




def _endpoints_torch(edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回：
      starts: (K,D) 每条边的起点
      ends  : (K,D) 每条边的终点
    """
    assert edges.ndim == 3 and edges.size(-1) >= 2, "edges 应为 (K,N,D>=2)"
    starts = edges[:, 0, :]
    ends   = edges[:, -1, :]
    return starts, ends

def _edge_len_torch(edges: torch.Tensor) -> torch.Tensor:
    """
    估计每条边的长度（首末点距离），用于选择起点的启发。
    """
    s, e = _endpoints_torch(edges)
    return torch.norm(e - s, dim=-1)

def reorder_and_flip_edges_torch(
    edges: torch.Tensor,
    *,
    start_index: Optional[int] = None,   # 指定从哪条边开始；None 表示自动选择
    make_cycle: bool = True,             # 是否尽量让最后一条的末端接近第一条的起点
    tol: Optional[float] = None,         # 容差（仅用于诊断；不影响贪心选择），None 表示不阈值判断
    prefer_long_start: bool = True,      # 自动起点时优先选“较长”的边
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 edges 重新排序并按需翻转，使相邻两条边在数组中“顺次连接”：
      new_edges[i][-1] ≈ new_edges[i+1][0]

    参数：
      edges: (K,N,D>=2)  (支持 2D/3D)
    返回：
      edges_reordered: (K,N,D) 新次序且已按需翻转
      order          : (K,)    新次序对应的原索引 (long)
      flips          : (K,)    每条边是否翻转（True=反向）(bool)
      distances      : (K-1 + 可选收尾) 相邻连接的欧氏距离 (float)
    """
    E = edges.clone()
    assert E.ndim == 3 and E.size(-1) >= 2, "edges 必须为 (K,N,D>=2)"
    device, dtype = E.device, E.dtype
    K, N, D = E.shape

    starts, ends = _endpoints_torch(E)

    # ---------- 选择起点 ----------
    if start_index is None:
        lengths = _edge_len_torch(E)  # (K,)
        if prefer_long_start:
            cand = torch.argsort(-lengths)  # 从长到短
        else:
            cand = torch.arange(K, device=device)

        all_points = torch.cat([starts, ends], dim=0)  # (2K,D)
        best_idx, best_score = None, float("inf")
        # 只看前 10 个候选即可
        topk = int(min(10, K))
        for t in range(topk):
            idx = cand[t].item()
            s0, e0 = starts[idx], ends[idx]
            d1 = torch.norm(all_points - s0, dim=1).min().item()
            d2 = torch.norm(all_points - e0, dim=1).min().item()
            score = min(d1, d2) - 1e-9 * idx  # 轻微偏置
            if score < best_score:
                best_score = score
                best_idx = idx
        start_index = int(best_idx)

    # 决定起点边方向：让“末端”更接近其它边任一端点
    s0, e0 = starts[start_index], ends[start_index]
    others = torch.tensor(
        [i for i in range(K) if i != start_index],
        device=device, dtype=torch.long
    )

    def _min_to_others(pt: torch.Tensor) -> float:
        if others.numel() == 0:
            return 0.0
        s_oth = starts.index_select(0, others)  # (K-1,D)
        e_oth = ends.index_select(0, others)    # (K-1,D)
        d1 = torch.norm(pt - s_oth, dim=1)
        d2 = torch.norm(pt - e_oth, dim=1)
        return torch.min(torch.minimum(d1, d2)).item()

    use_flip0 = _min_to_others(s0) < _min_to_others(e0)
    cur_idx   = start_index
    cur_flip  = use_flip0
    used      = torch.zeros(K, dtype=torch.bool, device=device)
    used[cur_idx] = True

    order_list: List[int] = [cur_idx]
    flips_list: List[bool] = [bool(cur_flip)]
    dists_list: List[float] = []

    def _end_point(idx: int, flip: bool) -> torch.Tensor:
        return starts[idx] if flip else ends[idx]
    def _start_point(idx: int, flip: bool) -> torch.Tensor:
        return ends[idx] if flip else starts[idx]

    cur_end = _end_point(cur_idx, cur_flip)

    # ---------- 贪心扩展 ----------
    for _ in range(K - 1):
        candidates = torch.nonzero(~used, as_tuple=False).squeeze(1)  # (M,)
        # 向量化比较两种方向的“起点”与 cur_end 的距离
        s_cand = starts.index_select(0, candidates)   # (M,D)
        e_cand = ends.index_select(0, candidates)     # (M,D)
        d1s = torch.norm(cur_end - s_cand, dim=1)     # 起点=starts[j]
        d2s = torch.norm(cur_end - e_cand, dim=1)     # 起点=ends[j]
        # 选择更小的那一侧作为接入方向
        min_ds = torch.minimum(d1s, d2s)              # (M,)
        min_val, rel_idx = torch.min(min_ds, dim=0)
        best_j = candidates[rel_idx].item()
        # 决定是否翻转：d1<=d2 则不翻转；否则翻转
        best_flip = bool(d1s[rel_idx] > d2s[rel_idx])  # True 表示反向
        best_d = float(min_val.item())

        order_list.append(int(best_j))
        flips_list.append(bool(best_flip))
        dists_list.append(best_d)
        used[best_j] = True
        cur_idx, cur_flip = int(best_j), bool(best_flip)
        cur_end = _end_point(cur_idx, cur_flip)

    order = torch.tensor(order_list, dtype=torch.long, device=device)
    flips = torch.tensor(flips_list, dtype=torch.bool, device=device)

    # ---------- 组装输出 ----------
    reordered = []
    for idx, fl in zip(order.tolist(), flips.tolist()):
        ei = E[idx]
        if fl:
            ei = torch.flip(ei, dims=[0]).clone()
        reordered.append(ei)
    edges_reordered = torch.stack(reordered, dim=0)

    # 补充闭环距离（如果需要）
    if make_cycle:
        d_close = torch.norm(edges_reordered[-1, -1] - edges_reordered[0, 0]).item()
        dists_list.append(d_close)

    distances = torch.tensor(dists_list, dtype=dtype, device=device)

    # ---------- 可选：容差诊断 ----------
    if tol is not None:
        bad = torch.nonzero(distances > tol, as_tuple=False).squeeze(1).cpu().tolist()
        if len(bad) > 0:
            print(f"[reorder] 有 {len(bad)} 处连接距离大于 tol={tol:.3g}，索引: {bad}，最大={float(distances.max()):.6g}")

    return edges_reordered, order, flips, distances

def detect_extra_edges_by_cost(
    edges: torch.Tensor,
    tol: float = 1e-9
) -> torch.Tensor:
    """
    使用“串联路径总距离”作为 cost。
    只允许两种输出：
      1) 全为 True（不删边）
      2) 恰有一个 False（只删一条：移除后整体 cost 最小的那条）
    参数:
      edges: (K,N,D>=2)
      tol  : 容差；(cost_wo - cost_full) > tol 视为“移除会变差”，因此保留
    返回:
      keep_mask: (K,) bool
    """
    K = edges.size(0)
    if K <= 3:
        # 没法删，全部保留
        return torch.ones(K, dtype=torch.bool, device=edges.device)

    assert edges.ndim == 3 and edges.size(-1) >= 2

    # 全量 cost
    # _, _, _, dist_full = reorder_and_flip_edges_torch(edges, make_cycle=True,tol=0.001)
    _, _, _, dist_full = reorder_and_flip_edges_torch(edges, make_cycle=True)
    cost_full = dist_full.sum()

    # 依次尝试移除每条边，记录移除后的 cost
    costs_wo = torch.empty(K, dtype=edges.dtype, device=edges.device)
    for i in range(K):
        edges_wo_i = torch.cat([edges[:i], edges[i+1:]], dim=0)
        _, _, _, dist_wo = reorder_and_flip_edges_torch(edges_wo_i, make_cycle=True)
        costs_wo[i] = dist_wo.sum()

    # 基于阈值的初判
    keep_mask = (costs_wo - cost_full) > tol
    num_false = int((~keep_mask).sum().item())

    if num_false <= 1:
        # 0 或 1 个 False，直接返回
        return keep_mask

    # 多个 False：只删除移除后 cost 最小的那条
    false_idx = torch.nonzero(~keep_mask, as_tuple=False).squeeze(1)
    # 在候选集合里找 argmin
    sub_costs = costs_wo.index_select(0, false_idx)
    j = torch.argmin(sub_costs).item()
    i_star = false_idx[j].item()

    final_keep = torch.ones(K, dtype=torch.bool, device=edges.device)
    final_keep[i_star] = False
    return final_keep
    
class RotTransDecoder(nn.Module):
    def __init__(self, args):
        super(RotTransDecoder, self).__init__()
        self.rot_decoder = nn.Linear(768, 3)
    
    def forward(self, kpts_token, edges_token, pc_token):
        pass

class FlattenModel(nn.Module):
    # OK
    th_nms_dist_patch = 0.03
    th_valid_patch=0.7
    th_valid_patch_curve_sim=0.5
    th_valid_curve=0.5
    
    # maybe OK
    d_patch_curve = 0.1
    th_nms_dist_curve = 0.1
    d_curve_curve = 0.2
    
    def __init__(self, args:FlattenConfig):
        super(FlattenModel, self).__init__()
        
        self.args = args

        self.dropout = 0.
        
        
        self.edge_self_att = nn.ModuleList([
            SelfAttentionModule(768, 6, 0)
            for _ in range(12)
        ])
        
        # self.patch_self_att = nn.ModuleList([
        #     SelfAttentionModule(768, 6, 0)
        #     for _ in range(12)
        # ])
        

        self.edge_pc_cross_att = nn.ModuleList([
            CrossAttentionModule(768, 6, 0)
            for _ in range(12)
        ])


        self.edge_fnn = nn.ModuleList([
            FFN(768)
            for _ in range(12)
        ])
        
        # self.patch_fnn = nn.ModuleList([
        #     FFN(768)
        #     for _ in range(12)
        # ])
        
        self.edge_ffn_norm = nn.ModuleList([nn.LayerNorm(768) for _ in range(12)])
        # self.patch_ffn_norm = nn.ModuleList([nn.LayerNorm(768) for _ in range(12)])
        self.edge_decoder = EdgeDecoder(args)
        self.scale_decoder = nn.Linear(768, 1)
    
    def _pad_and_masks(self, edges_list):
        """
        verts_list: List[Tensor[Lv_i, C]]
        edges_list: List[Tensor[Le_i, C]]
        ck_sim_list: Optional[List[Tensor[Le_i, Lv_i]]]
        return:
            verts_token  [N, Lv_max, C]
            edges_token  [N, Le_max, C]
            verts_mask   [N, Lv_max]  bool  (True=PAD)
            edges_mask   [N, Le_max]  bool
            verts_lens   [N]
            edges_lens   [N]
            ck_sim_padded [N, Le_max, Lv_max] (if ck_sim_list is not None)
            ck_sim_mask   [N, Le_max, Lv_max] (if ck_sim_list is not None)
        """
        edges_token = pad_sequence(edges_list, batch_first=True)  # [N, Le_max, C]

        device = edges_token.device
        N, Le_max = edges_token.size(0), edges_token.size(1)

        edges_lens = torch.as_tensor([e.size(0) for e in edges_list], device=device)

        # === 生成 mask ===
        if Le_max == 0:
            edges_mask = torch.empty((N, 0), dtype=torch.bool, device=device)
        else:
            rng_e = torch.arange(Le_max, device=device).unsqueeze(0).expand(N, Le_max)
            edges_mask = rng_e >= edges_lens.unsqueeze(1)


        return edges_token, edges_mask, edges_lens


    def _gather_panel_indices(self,panel_num,
                          c_patch_ids, c_curve_ids, c_PC_mat):
        panel_patch_ids = []
        panel_edge_ids  = []
        for panel_idx in range(panel_num):
            # patch 索引（每 panel 一个 token）
            p_ids = (c_patch_ids[1] == panel_idx).nonzero(as_tuple=True)[0][0].item()
            panel_patch_ids.append(c_patch_ids[0][p_ids])


            # 边（变长）
            edge_ids_local = torch.nonzero(c_PC_mat[panel_idx] == 1).reshape(-1)
            e_ids = index_b_in_a(c_curve_ids[1], edge_ids_local)
            panel_edge_ids.append(c_curve_ids[0][e_ids])
        return panel_patch_ids, panel_edge_ids


    def filter_by_valid_th_ensure_patch_closed(self,data):
            """ 过滤掉 prob 太小的；再过滤 patch_curve_similarity 全为 0 的行/列 """
            # 过滤patch
            valid_patch_idx = (data['patches']['valid_prob'] > self.th_valid_patch).nonzero(as_tuple=True)[0]
            patches = data['patches']
            for k in patches:
                if k == "points_scaled":
                    valid_scaled_pts=[]
                    for p_id in valid_patch_idx:
                        valid_scaled_pts.append(patches[k][p_id])
                    patches[k] = valid_scaled_pts
                else:
                    patches[k] = patches[k][valid_patch_idx]

            # ---------------------------------------------------------------------------------
            patch_curve_similarity = data['patch_curve_similarity'][valid_patch_idx]

            top_vals, top_idx = torch.topk(patch_curve_similarity, k=3, dim=1)

            M = patch_curve_similarity.size(0)
            K = top_idx.size(1)   # 这里就是 3

            row_idx = torch.arange(M, device=patch_curve_similarity.device).unsqueeze(1).expand(-1, K)

            patch_curve_similarity[row_idx, top_idx] = 1.0
            
            data['patch_curve_similarity']=patch_curve_similarity
            
            marked_curve_ids=top_idx.reshape(-1).unique()
            # data['curves']['valid_prob'][marked_curve_ids]=1.0
            # ---------------------------------------------------------------------------------
            
            valid_curve_idx = (data['curves']['valid_prob'] > self.th_valid_curve).nonzero(as_tuple=True)[0]
            curves = data['curves']
            for k in curves:
                curves[k] = curves[k][valid_curve_idx]

            data['patch_curve_similarity'] = data['patch_curve_similarity'][:, valid_curve_idx]
            
            # 2) 过滤 patch_curve_similarity 中“全 0”的行(补丁)与列(曲线)
            pcs = data['patch_curve_similarity']                              # 形状: (n_patch, n_curve)
            row_keep = (pcs > self.th_valid_patch_curve_sim).any(dim=1)                                  # 哪些 patch 行保留
            col_keep = (pcs > self.th_valid_patch_curve_sim).any(dim=0)                                  # 哪些 curve 列保留

            # 紧随其后对 patches / curves 及矩阵做相同的二次过滤
            for k in patches:
                if k == "points_scaled":
                    valid_scaled_pts=[]
                    for p_id,keep in enumerate(row_keep):
                        if keep:
                            valid_scaled_pts.append(patches[k][p_id])
                    patches[k] = valid_scaled_pts
                else:
                    patches[k] = patches[k][row_keep]
            for k in curves:
                curves[k] = curves[k][col_keep]

            data['patch_curve_similarity'] = pcs[row_keep][:, col_keep]
            
            data['patch_curve_connectivity'] = data['patch_curve_similarity']>self.th_valid_patch_curve_sim
            return data
        
        
    def filter_by_valid_th(self,data):
            """ 过滤掉 prob 太小的；再过滤 patch_curve_similarity 全为 0 的行/列 """
            # 1) 先按 valid_prob 阈值过滤
            valid_curve_idx = (data['curves']['valid_prob'] > self.th_valid_curve).nonzero(as_tuple=True)[0]
            valid_patch_idx = (data['patches']['valid_prob'] > self.th_valid_patch).nonzero(as_tuple=True)[0]

            curves = data['curves']
            for k in curves:
                curves[k] = curves[k][valid_curve_idx]

            patches = data['patches']
            
            for k in patches:
                if k == "points_scaled":
                    valid_scaled_pts=[]
                    for p_id in valid_patch_idx:
                        valid_scaled_pts.append(patches[k][p_id])
                    patches[k] = valid_scaled_pts
                else:
                    patches[k] = patches[k][valid_patch_idx]
                    
            data['patch_curve_similarity'] = data['patch_curve_similarity'][valid_patch_idx][:, valid_curve_idx]
            if 'patch_curve_connectivity' in data:
                data['patch_curve_connectivity'] = data['patch_curve_connectivity'][valid_patch_idx][:, valid_curve_idx]

            # 2) 过滤 patch_curve_similarity 中“全 0”的行(补丁)与列(曲线)
            pcs = data['patch_curve_similarity']                              # 形状: (n_patch, n_curve)
            row_keep = (pcs > self.th_valid_patch_curve_sim).any(dim=1)                                  # 哪些 patch 行保留
            col_keep = (pcs > self.th_valid_patch_curve_sim).any(dim=0)                                  # 哪些 curve 列保留

            # 紧随其后对 patches / curves 及矩阵做相同的二次过滤
            for k in patches:
                if k == "points_scaled":
                    valid_scaled_pts=[]
                    for p_id,keep in enumerate(row_keep):
                        if keep:
                            valid_scaled_pts.append(patches[k][p_id])
                    patches[k] = valid_scaled_pts
                else:
                    patches[k] = patches[k][row_keep]
            for k in curves:
                curves[k] = curves[k][col_keep]

            data['patch_curve_similarity'] = pcs[row_keep][:, col_keep]
            if 'patch_curve_connectivity' in data:
                data['patch_curve_connectivity'] = data['patch_curve_connectivity'][row_keep][:, col_keep]
            else:
                data['patch_curve_connectivity'] = data['patch_curve_similarity']>self.th_valid_patch_curve_sim
            return data
        

    def extract_topo_step0(self,patch_curve_similarity,curve_points,curve_valid_prob,curve_features,
                              patch_points,patch_points_scaled,patch_valid_prob,patch_features):
        
        """
            从实验结果来看,最好的做法反而是不做patch的NMS,这样才可以保证小的patch不被过滤掉。
            因此,step1只根据阈值做过滤
        """
                    
        data={}
        data['patch_curve_similarity'] = patch_curve_similarity

        data['curves']={}
        data['curves']['points']=curve_points
        data['curves']['valid_prob']=curve_valid_prob
        data['curves']['features']=curve_features
        data['patches']={}
        data['patches']['points']=patch_points
        data['patches']['points_scaled']=patch_points_scaled
        data['patches']['valid_prob']=patch_valid_prob
        data['patches']['features']=patch_features
        
        # self.filter_by_valid_th_ensure_patch_closed(data)
        
        
        # 这里是一个初步的过滤
        self.filter_by_valid_th(data)
        
        
        assert data['curves']['valid_prob'].shape[0]==data['patch_curve_connectivity'].shape[1]
        assert data['patches']['valid_prob'].shape[0]==data['patch_curve_connectivity'].shape[0]
        return (
            data['patch_curve_similarity'],
            data['curves']['points'],
            data['curves']['valid_prob'],
            data['curves']['features'],
            data['patches']['points'],
            data['patches']['points_scaled'],
            data['patches']['valid_prob'],
            data['patches']['features'],
            data['patch_curve_connectivity']
        )
        
    def extract_topo_step(self,patch_curve_similarity,curve_points,curve_valid_prob,curve_features,
                              patch_points,patch_points_scaled,patch_valid_prob,patch_features):
        
        def curve_NMS(data):
            curves = data['curves']
            
            patch_curve_similarity = data['patch_curve_similarity']
            patch_curve_connectivity = data['patch_curve_connectivity']
            curve_pts = curves['points']
            
            curve_num=patch_curve_similarity.shape[1]
            

            for i in range(curve_num-1):
                for j in range(i+1,curve_num):
                    c1=curve_pts[i]
                    c2=curve_pts[j]
                    d1 = torch.cdist(c1.unsqueeze(0), c2.unsqueeze(0), p=2)
                    cost1=d1.min(dim=-1).values.mean().item()
                    d2 = torch.cdist(c2.unsqueeze(0), c1.unsqueeze(0), p=2)
                    cost2=d2.min(dim=-1).values.mean().item()
                    if cost1<0.03 and cost2<0.03:
                        better_idx=i if curve_valid_prob[i]>=curve_valid_prob[j] else j
                        worse_idx=i if curve_valid_prob[i]<curve_valid_prob[j] else j

                        worse_matched_patches=patch_curve_connectivity[:,worse_idx].nonzero().reshape(-1)
                        better_matched_patches=patch_curve_connectivity[:,better_idx].nonzero().reshape(-1)
                        total_patches=set(worse_matched_patches.tolist()).union(set(better_matched_patches.tolist()))
                        if len(total_patches)>2:continue
                        data['patch_curve_connectivity'][:,worse_idx]=0.0
                        data['patch_curve_similarity'][:,worse_idx]=0.0
                        
                        data['patch_curve_connectivity'][worse_matched_patches,better_idx]=1.0
                        data['patch_curve_similarity'][worse_matched_patches,better_idx]=1.0
        
                        
        # 这个比较tricy，如果一条边只邻接一个面，且这条边是一条长边的子边，则过滤
        def filter_sub_curve(data):
            curves = data['curves']
            
            patch_curve_similarity = data['patch_curve_similarity']
            patch_curve_connectivity = data['patch_curve_connectivity']
            curve_pts = curves['points']
            
            patch_num=patch_curve_similarity.shape[0]
            
            for p_id in range(patch_num):
                conn_curves_ids=patch_curve_connectivity[p_id].nonzero().reshape(-1)
                conn_curve_num=len(conn_curves_ids)
                if conn_curve_num<=3:continue
                for i in range(conn_curve_num-1):
                    for j in range(i+1,conn_curve_num):
                        if patch_curve_connectivity[:,conn_curves_ids[i]].sum().item()==1 and patch_curve_connectivity[:,conn_curves_ids[j]].sum().item()==1:
                            c1=curve_pts[conn_curves_ids[i]]
                            c2=curve_pts[conn_curves_ids[j]]
                            
                            d = torch.cdist(c1.unsqueeze(0), c2.unsqueeze(0), p=2)
                            cost=d.min(dim=-1).values.mean().item()
                            if cost<0.04:
                                data['curves']['valid_prob'][conn_curves_ids[i]]=0.0
                                # print(f"Sub curve {conn_curves_ids[i]} is filtered")
                            d = torch.cdist(c2.unsqueeze(0), c1.unsqueeze(0), p=2)
                            cost=d.min(dim=-1).values.mean().item()
                            if cost<0.04:
                                data['curves']['valid_prob'][conn_curves_ids[j]]=0.0
                                # print(f"Sub curve {conn_curves_ids[j]} is filtered")
                                
        data={}
        data['patch_curve_similarity'] = patch_curve_similarity

        data['curves']={}
        data['curves']['points']=curve_points
        data['curves']['valid_prob']=curve_valid_prob
        data['curves']['features']=curve_features
        data['patches']={}
        data['patches']['points']=patch_points
        data['patches']['points_scaled']=patch_points_scaled
        data['patches']['valid_prob']=patch_valid_prob
        data['patches']['features']=patch_features
        
        self.filter_by_valid_th(data)

        curve_NMS(data)
        self.filter_by_valid_th(data)
        
        filter_sub_curve(data)
        self.filter_by_valid_th(data)
        
        assert data['curves']['valid_prob'].shape[0]==data['patch_curve_connectivity'].shape[1]
        assert data['patches']['valid_prob'].shape[0]==data['patch_curve_connectivity'].shape[0]
        return (
            data['patch_curve_similarity'],
            data['curves']['points'],
            data['curves']['valid_prob'],
            data['curves']['features'],
            data['patches']['points'],
            data['patches']['points_scaled'],
            data['patches']['valid_prob'],
            data['patches']['features'],
            data['patch_curve_connectivity']
        )   
    
  
       
    def extract_topo(self,patch_curve_similarity,curve_points,curve_valid_prob,curve_features,
                              patch_points,patch_valid_prob,patch_features):
        
        # OK
        th_nms_dist_patch = 0.03
        th_valid_patch=0.7
        th_valid_patch_curve_sim=0.6
        th_valid_curve=0.5
        
        # maybe OK
        d_patch_curve = 0.1
        th_nms_dist_curve = 0.1
        d_curve_curve = 0.2

        def filter_by_valid_th(data):
            """ 过滤掉 prob 太小的；再过滤 patch_curve_similarity 全为 0 的行/列 """
            # 1) 先按 valid_prob 阈值过滤
            valid_curve_idx = (data['curves']['valid_prob'] > th_valid_curve).nonzero(as_tuple=True)[0]
            valid_patch_idx = (data['patches']['valid_prob'] > th_valid_patch).nonzero(as_tuple=True)[0]

            curves = data['curves']
            for k in curves:
                curves[k] = curves[k][valid_curve_idx]

            patches = data['patches']
            for k in patches:
                patches[k] = patches[k][valid_patch_idx]

            data['patch_curve_similarity'] = data['patch_curve_similarity'][valid_patch_idx][:, valid_curve_idx]
            if 'patch_curve_connectivity' in data:
                data['patch_curve_connectivity'] = data['patch_curve_connectivity'][valid_patch_idx][:, valid_curve_idx]

            # 2) 过滤 patch_curve_similarity 中“全 0”的行(补丁)与列(曲线)
            pcs = data['patch_curve_similarity']                              # 形状: (n_patch, n_curve)
            row_keep = (pcs > th_valid_patch_curve_sim).any(dim=1)                                  # 哪些 patch 行保留
            col_keep = (pcs > th_valid_patch_curve_sim).any(dim=0)                                  # 哪些 curve 列保留

            # 紧随其后对 patches / curves 及矩阵做相同的二次过滤
            for k in patches:
                patches[k] = patches[k][row_keep]
            for k in curves:
                curves[k] = curves[k][col_keep]

            data['patch_curve_similarity'] = pcs[row_keep][:, col_keep]
            if 'patch_curve_connectivity' in data:
                data['patch_curve_connectivity'] = data['patch_curve_connectivity'][row_keep][:, col_keep]
            else:
                data['patch_curve_connectivity'] = data['patch_curve_similarity']>th_valid_patch_curve_sim
            return data
                
                
        def NMS_patch(data):
            patches = data['patches']
            
            patch_curve_similarity = data['patch_curve_similarity']
            # when deleting an item, the corresponding topo elements should be set as zeros
            
            patch_pts = patches['points']

            n_patch = patch_pts.shape[0]
            for i in range(n_patch - 1):
                for j in range(i + 1, n_patch):
                    tmp_chamfer_xy = chamfer_distance(patch_pts[i], patch_pts[j], direction='x_to_y')
                    tmp_chamfer_yx = chamfer_distance(patch_pts[i], patch_pts[j], direction='y_to_x')
                    if tmp_chamfer_xy < th_nms_dist_patch and tmp_chamfer_yx < th_nms_dist_patch:
                        # distance close enough
                        # print('merge patch {} {}'.format(i, j))
                        if patches['valid_prob'][i]>patches['valid_prob'][j]:
                            patches['valid_prob'][j] = 0
                            patch_curve_similarity[j] = 0.0
                        elif patches['valid_prob'][j]>patches['valid_prob'][i]:
                            patches['valid_prob'][i] = 0
                            patch_curve_similarity[i] = 0.0
        
        def NMS_patch_local_to_glb(data):
            patches = data['patches']
            patch_curve_similarity = data['patch_curve_similarity']
            patch_pts = patches['points']                 # Tensor [n_patch, m, 3]
            n_patch = patch_pts.shape[0]
            idx_all = torch.arange(n_patch, device=patch_pts.device)

            for i in range(n_patch):
                # 其余所有 patch 的点拼接为全局集合（集合差 ≠ 张量减法）
                mask = (idx_all != i)
                others = patch_pts[mask].reshape(-1, 3)  # [(n_patch-1)*m, 3]

                tmp_chamfer_local_to_glb = chamfer_distance(
                    patch_pts[i], others, direction='x_to_y'
                )
                if tmp_chamfer_local_to_glb < th_nms_dist_patch and patches['valid_prob'][i]<0.90:
                    patches['valid_prob'][i] = 0
                    patch_curve_similarity[i] = 0.0

        
        def NMS_curve(data):
            curves = data['curves']
            patches = data['patches']
            
            patch_curve_similarity = data['patch_curve_similarity']
            patch_curve_similarity_round = torch.round(data['patch_curve_similarity'])
            patch_curve_connectivity = data['patch_curve_connectivity']
            curve_pts = curves['points']
            n_curve = patch_curve_similarity.shape[1]
            
            patch_curve_geo_sim=get_patch_curve_similarity_geom(data)
            
            for i in range(n_curve - 1):
                for j in range(i + 1, n_curve):
                    c1,c2,_,_=global_scale_min_t(curve_pts[i],curve_pts[j])
                    tmp_chamfer=chamfer_distance(c1,c2)
                    if tmp_chamfer < th_nms_dist_curve:
                        i_con_patch=patch_curve_connectivity[:,i].nonzero().reshape(-1).tolist()
                        j_con_patch=patch_curve_connectivity[:,j].nonzero().reshape(-1).tolist()
                        i_con_patch.sort()
                        j_con_patch.sort()
                    
                        if len(i_con_patch)==len(j_con_patch):
                            if len(i_con_patch)==1 and i_con_patch[0]==j_con_patch[0]:
                                p_idx=i_con_patch[0]
                                if patch_curve_geo_sim[p_idx][i]>0.7 and patch_curve_geo_sim[p_idx][j]>0.7:
                                    if curves['valid_prob'][i]>curves['valid_prob'][j]:
                                        curves['valid_prob'][j] = 0
                                        patch_curve_similarity[:,j] = 0.0
                                    elif curves['valid_prob'][j]>curves['valid_prob'][i]:
                                        curves['valid_prob'][i] = 0
                                        patch_curve_similarity[:,i] = 0.0

                            elif len(i_con_patch)==2 and i_con_patch[0]==j_con_patch[0] and i_con_patch[1]==j_con_patch[1]:
                                if curves['valid_prob'][i]>curves['valid_prob'][j]:
                                    curves['valid_prob'][j] = 0
                                    patch_curve_similarity[:,j] = 0.0
                                elif curves['valid_prob'][j]>curves['valid_prob'][i]:
                                    curves['valid_prob'][i] = 0
                                    patch_curve_similarity[:,i] = 0.0
        
        def get_patch_curve_similarity_geom(data, flag_exp=True):
            all_curve_pts = data['curves']['points']   # torch.Tensor [Nc, Mc, D]
            all_patch_pts = data['patches']['points']  # torch.Tensor [Nf, Mp, D]
            
            boundary_pts = []
            for i in range(len(all_patch_pts)):
                bpt,_ = get_boundary(all_patch_pts[i])
                boundary_pts.append(bpt)

            nf = len(boundary_pts)
            nc = all_curve_pts.shape[0]
            sim = torch.zeros((nf, nc), dtype=all_curve_pts.dtype, device=all_curve_pts.device)


            for i in range(nf):
                for j in range(nc):
                    pts_diff = boundary_pts[i].unsqueeze(1) - all_curve_pts[j]
                    pts_dist = torch.linalg.norm(pts_diff, dim=-1)
                    sim[i, j] = torch.mean(pts_dist.min(dim=-1).values)
            if flag_exp:
                sim = torch.exp(-sim * sim / (d_patch_curve * d_patch_curve))
            
            return sim

        def extract_patch_curve_topo(data_input):
            curves = data_input['curves']
            patches = data_input['patches']
            patch_curve_similarity = data_input['patch_curve_similarity']
            curves_valid_prob = curves['valid_prob']
            patches_valid_prob = patches['valid_prob']

            patch_curve_similarity_geom = get_patch_curve_similarity_geom(data_input)
            
            # mask = (patch_curve_similarity_geom < 0.5) & (max_dist_endpoint < 0.1)
            # patch_curve_similarity[mask]=0.0
            
            for i in range(patch_curve_similarity.shape[1]):
                if patch_curve_similarity[:,i].max()<th_valid_patch_curve_sim:
                    patch_curve_similarity[:,i]=0
            
            for i in range(patch_curve_similarity.shape[0]):
                if patch_curve_similarity[i,:].max()<th_valid_patch_curve_sim:
                    patch_curve_similarity[i,:]=0
            
            patch_curve_connectivity=data['patch_curve_similarity']>th_valid_patch_curve_sim
            data_input['patch_curve_similarity']=patch_curve_similarity
            data['patch_curve_connectivity']=patch_curve_connectivity
        
        
        def patch_refine(data_input):
            """ 如果一个 patch 只邻接于 1(2) 条边, 这个 patch 是无效的 """
            patches = data_input['patches']
            patches_valid_prob = patches['valid_prob']
            patch_curve_connectivity=data['patch_curve_connectivity']
        
            for i in range(patch_curve_connectivity.shape[0]):
                if patch_curve_connectivity[i].sum()<=2:
                    patches_valid_prob[i]=0
                    
        def patch_curve_similarity_refine(data):
            """ 如果一个 curve 与大于等于三个 patch 邻接, 只取概率最大的两个"""
            patch_curve_similarity=data['patch_curve_similarity']
            patch_curve_connectivity=data['patch_curve_connectivity']
            curve_connect_patch_num=patch_curve_connectivity.sum(dim=0)
            for i in range(patch_curve_connectivity.shape[1]):
                if curve_connect_patch_num[i]>=3:
                    # 取最大的两个
                    valid_patches=torch.argsort(patch_curve_similarity[:,i],descending=True)[:2]
                    patch_curve_connectivity[:,i]=False
                    patch_curve_connectivity[valid_patches,i]=True
                    patch_curve_similarity[:,i]=0.0
        
        def curve_refine(data):
            curve_pts = data['curves']['points']   # torch.Tensor [Nc, Mc, D]
            patch_pts = data['patches']['points']  # torch.Tensor [Nf, Mp, D]
            curve_prob = data['curves']['valid_prob']
            patch_prob = data['patches']['valid_prob']
            
            patch_curve_similarity = data['patch_curve_similarity'] 

            valid_curve_ids = (curve_prob>0.7).nonzero().reshape(-1).tolist()
            candidate_curve_ids = (curve_prob<=0.7).nonzero().reshape(-1).tolist()
            
            if len(candidate_curve_ids)==0:
                return
            valid_curves=curve_pts[valid_curve_ids].reshape(-1,3)
            candidate_curves=curve_pts[candidate_curve_ids]
            candidate_curve_prob=curve_prob[candidate_curve_ids]
            
            boundary_pts = []
            for i in range(len(patch_pts)):
                bpt,_ = get_boundary(patch_pts[i])
                boundary_pts.append(bpt)
            
            boundary_ids=[len(it) for it in boundary_pts]
            
            boundary_pts=torch.concat(boundary_pts)
            num_boundary_pts=boundary_pts.shape[0]
            
            cost=chamfer_distance(boundary_pts,valid_curves,direction="x_to_y")*num_boundary_pts
            
            to_remove=[]
            for i in (-candidate_curve_prob).argsort():
                c_id=candidate_curve_ids[i]

                c_curve=candidate_curves[i]
                c_cost=chamfer_distance(boundary_pts,torch.cat([valid_curves, c_curve], dim=0) ,direction="x_to_y")*num_boundary_pts
                diff= cost - c_cost
                
                if diff>1.0:
                    to_remove.append(c_id)
            
            for c in to_remove:
                candidate_curve_ids.remove(c)
            curve_prob[candidate_curve_ids]=0.0
        
        
        def patch_curve_connectivity_refine(data):
            curve_pts = data['curves']['points']   
            patch_pts = data['patches']['points'] 
            curve_prob = data['curves']['valid_prob']
            patch_prob = data['patches']['valid_prob']
            patch_curve_similarity = data['patch_curve_similarity'] 
            patch_curve_connectivity = data['patch_curve_connectivity']
            
            patch_num=patch_curve_similarity.shape[0]
            
            for p_idx in range(patch_num):
                now_boundary,_=get_boundary(patch_pts[p_idx])
                now_boundary,trans_info=scale_point_cloud_to_aabb(now_boundary)
                num_boundary_pts=now_boundary.shape[0]
                nei_curve_ids=patch_curve_connectivity[p_idx].nonzero().reshape(-1).tolist()
                total_curvs_pts=apply_affine_from_info(curve_pts[nei_curve_ids].reshape(-1,3),trans_info)
                cost=chamfer_distance(now_boundary,total_curvs_pts,direction="x_to_y")*num_boundary_pts
                n_curve=len(nei_curve_ids)
                for i in range(n_curve - 1):
                    for j in range(i + 1, n_curve):
                        c_i=nei_curve_ids[i]
                        c_j=nei_curve_ids[j]
                        chamfer_dist = min(chamfer_distance(curve_pts[c_i], curve_pts[c_j],direction="x_to_y"),chamfer_distance(curve_pts[c_i], curve_pts[c_j],direction="y_to_x"))
                        chamfer_sim = math.exp(-chamfer_dist * chamfer_dist / (d_curve_curve * d_curve_curve))
                        cos_sim = cos_dist_curve(curve_pts[c_i], curve_pts[c_j])
                        chamfer_dist_max=min(chamfer_distance_max(curve_pts[c_i], curve_pts[c_j],direction="x_to_y"),chamfer_distance_max(curve_pts[c_i], curve_pts[c_j],direction="y_to_x"))
                        chamfer_max_sim = math.exp(-chamfer_dist_max * chamfer_dist_max / (d_curve_curve * d_curve_curve))
                        if chamfer_sim>0.8 and cos_sim>0.9 and chamfer_max_sim>0.9:
                            tmp_nei_curve_ids=nei_curve_ids.copy()
                            tmp_nei_curve_ids.remove(c_j)
                            c_cost=chamfer_distance(now_boundary,apply_affine_from_info(curve_pts[tmp_nei_curve_ids].reshape(-1,3),trans_info),direction="x_to_y")*num_boundary_pts
                            diff_j=c_cost-cost

                            tmp_nei_curve_ids=nei_curve_ids.copy()
                            tmp_nei_curve_ids.remove(c_i)
                            c_cost=chamfer_distance(now_boundary,apply_affine_from_info(curve_pts[tmp_nei_curve_ids].reshape(-1,3),trans_info),direction="x_to_y")*num_boundary_pts
                            diff_i=c_cost-cost

                            if diff_i>0.1 and diff_j>0.1:
                                continue
                            elif diff_i<diff_j:
                                patch_curve_similarity[p_idx][c_i]=0.0
                                patch_curve_connectivity[p_idx][c_i]=0.0
                            elif diff_j<diff_i:
                                patch_curve_similarity[p_idx][c_j]=0.0
                                patch_curve_connectivity[p_idx][c_j]=0.0

                    
        def filter_sub_curve(data):
            curves = data['curves']
            patches = data['patches']
            
            patch_curve_similarity = data['patch_curve_similarity']
            patch_curve_similarity_round = torch.round(data['patch_curve_similarity'])
            patch_curve_connectivity = data['patch_curve_connectivity']
            curve_pts = curves['points']
            n_curve = patch_curve_similarity.shape[1]
            
            for i in range(n_curve - 1):
                for j in range(i + 1, n_curve):
                    c1,c2,s1,s2=global_scale_min_t(curve_pts[i],curve_pts[j])
                    i_con_patch=patch_curve_connectivity[:,i].nonzero().reshape(-1).tolist()
                    j_con_patch=patch_curve_connectivity[:,j].nonzero().reshape(-1).tolist()
                    i_con_patch.sort()
                    j_con_patch.sort()
                    
                    if s1<s2 and (s1/s2)>0.25:
                        tmp_chamfer=chamfer_distance(c1,c2,direction="x_to_y")
                        if tmp_chamfer<0.05:
                            if set(i_con_patch)<=set(j_con_patch):
                                curves['valid_prob'][i] = 0
                                patch_curve_similarity[:,i] = 0.0
                    elif s2<s1 and (s2/s1)>0.25:
                        tmp_chamfer=chamfer_distance(c1,c2,direction="y_to_x")
                        if tmp_chamfer<0.05:
                            if set(j_con_patch)<=set(i_con_patch):
                                curves['valid_prob'][j] = 0
                                patch_curve_similarity[:,j] = 0.0
            
            
                    
        data={}
        data['patch_curve_similarity'] = patch_curve_similarity

        data['curves']={}
        data['curves']['points']=curve_points
        data['curves']['valid_prob']=curve_valid_prob
        data['curves']['features']=curve_features
        data['patches']={}
        data['patches']['points']=patch_points
        data['patches']['valid_prob']=patch_valid_prob
        data['patches']['features']=patch_features
        
        # 这里是一个初步的过滤
        filter_by_valid_th(data)
        
        
        # 那些重合的patch会被过滤掉
        NMS_patch(data)
        filter_by_valid_th(data)

        
        # 进一步过滤掉和其他patch重合的patch
        NMS_patch_local_to_glb(data)
        filter_by_valid_th(data)
        
        
        # 激进地过滤边，然后将那些被过滤的有效边加上
        curve_refine(data)
        filter_by_valid_th(data)

        # 理想情况下：
        #   2. 根据 patch_curve_sim 筛掉一些边或面
        extract_patch_curve_topo(data)
        filter_by_valid_th(data)
        
        
        # 如果一个 curve 与大于等于三个 patch 邻接, 只取概率最大的两个
        patch_curve_similarity_refine(data)
        filter_by_valid_th(data)
        
        NMS_curve(data)
        filter_by_valid_th(data)
        
        filter_sub_curve(data)
        filter_by_valid_th(data)
        
        # 如果一个 patch 只邻接于 1(2) 条边, 这个 patch 是无效的
        patch_refine(data)
        filter_by_valid_th(data)
        
        # # 过滤一部分错误的连接关系
        # for _ in range(3):
        #     patch_curve_connectivity_refine(data)
        #     filter_by_valid_th(data)
        
        assert data['curves']['valid_prob'].shape[0]==data['patch_curve_connectivity'].shape[1]
        assert data['patches']['valid_prob'].shape[0]==data['patch_curve_connectivity'].shape[0]
        return (
            data['patch_curve_similarity'],
            data['curves']['points'],
            data['curves']['valid_prob'],
            data['curves']['features'],
            data['patches']['points'],
            data['patches']['valid_prob'],
            data['patches']['features'],
            data['patch_curve_connectivity']
        )
    

    
    def get_edges_all(self,patch_curve_similarity,curve_points,curve_valid_prob,curve_features,patch_points,patch_points_scaled,patch_valid_prob,patch_features,patch_curve_connectivity):
        dim=768
        out={}
        flatten_pred={}
        panel_num=patch_valid_prob.shape[0]
        # dim_f=patch_features.shape[-1]
        if panel_num==0:
            out["flatten_pred"]={}
            out["patch_curve_similarity"]=patch_curve_similarity
            out["patch_curve_connectivity"]=patch_curve_connectivity
            
            out["curve_points"]=curve_points
            out["curve_valid_prob"]=curve_valid_prob
            
            out["patch_points"]=patch_points
            out["patch_valid_prob"]=patch_valid_prob
            out["patch_points_scaled"]=patch_points_scaled
            
            return out
        patch_tokens=patch_features.unsqueeze(1)

        edges_token_lst=[]
        
        for i in range(panel_num):
            connect_curve_ids=patch_curve_connectivity[i].nonzero(as_tuple=True)[0]
            edges_token_lst.append(curve_features[connect_curve_ids])
        
        edges_token, edges_mask, edges_lens = self._pad_and_masks(edges_token_lst)
        
        for i in range(12):
            # self-attn
            edges_token = self.edge_self_att[i](edges_token, key_padding_mask=edges_mask)

            # patch_tokens = self.patch_self_att[i](patch_tokens.reshape(1,-1,dim)).reshape(-1,1,dim)
            
            # with patch (cross-attn): Q=verts/edges, K=V=patch (长度=1，无需 mask)
            edges_token = self.edge_pc_cross_att[i](edges_token, patch_tokens)

            # p_fnn = self.patch_fnn[i](patch_tokens)
            # patch_tokens = self.patch_ffn_norm[i](patch_tokens + p_fnn)
            
            # FFN
            e_ffn = self.edge_fnn[i](edges_token)
            edges_token = self.edge_ffn_norm[i](edges_token + e_ffn)
        
        trans_pred = self.scale_decoder(patch_tokens).squeeze(1)  # [N, 1]
        
        for panel_idx in range(panel_num):
            Le_b = edges_lens[panel_idx].item()

            e_tok_b = edges_token[panel_idx, :Le_b, :]         # [Le_b, C]

            flatten_pred[panel_idx]={
                "edge_points":    self.edge_decoder(e_tok_b)["edge_points"],                  # 注意下文的 EdgeDecoder 事项
                # "trans_pred":    trans_pred[panel_idx],                                # [7]
                "scale_pred":    trans_pred[panel_idx],                                # [1]
            }
        
        out["flatten_pred"]=flatten_pred
        out["patch_curve_similarity"]=patch_curve_similarity
        out["patch_curve_connectivity"]=patch_curve_connectivity
        
        out["curve_points"]=curve_points
        out["curve_valid_prob"]=curve_valid_prob
        
        out["patch_points"]=patch_points
        out["patch_valid_prob"]=patch_valid_prob
        out["patch_points_scaled"]=patch_points_scaled
        return out
        
        
    @torch.no_grad()
    def infer(self, curve_pred, patch_pred, curve_features_batch, patch_features_batch,names=None):
        out_batch=[]
        bs=curve_features_batch.shape[0]
        for batch_idx in range(bs):
            curve_features=curve_features_batch[batch_idx]
            patch_features=patch_features_batch[batch_idx]

            pred_curve_logits=curve_pred["pred_curve_logits"][batch_idx]
            curve_points=curve_pred["pred_curve_points"][batch_idx]
            
            curve_valid_prob=torch.softmax(pred_curve_logits, dim=1)[:, [0]].reshape(-1)
            
            pred_patch_logits=patch_pred["pred_patch_logits"][batch_idx]
            patch_points=patch_pred["pred_patch_points"][batch_idx]
            patch_points_scaled=patch_pred['pred_patch_points_scaled'][batch_idx]
            
            patch_valid_prob=torch.softmax(pred_patch_logits, dim=1)[:, [0]].reshape(-1)
            
            
            curve_topo_embed_patch=curve_pred["curve_topo_embed_patch"][batch_idx]
            patch_topo_embed_curve=patch_pred["patch_topo_embed_curve"][batch_idx]
            patch_curve_similarity = torch.sigmoid(
                patch_topo_embed_curve @ curve_topo_embed_patch.T
            )
            

            patch_curve_similarity,curve_points,curve_valid_prob,curve_features,patch_points,patch_points_scaled,patch_valid_prob,patch_features,patch_curve_connectivity=\
                    self.extract_topo_step(patch_curve_similarity,curve_points,curve_valid_prob,curve_features,
                                patch_points,patch_points_scaled,patch_valid_prob,patch_features)
                
            flag=True
            while flag:
                flag=False
                patch_curve_similarity,curve_points,curve_valid_prob,curve_features,patch_points,patch_points_scaled,patch_valid_prob,patch_features,patch_curve_connectivity=\
                    self.extract_topo_step0(patch_curve_similarity,curve_points,curve_valid_prob,curve_features,
                                patch_points,patch_points_scaled,patch_valid_prob,patch_features)
                
                patch_curve_connectivity=patch_curve_similarity>self.th_valid_patch_curve_sim
                infer_out=self.get_edges_all(patch_curve_similarity,curve_points,curve_valid_prob,curve_features,patch_points,patch_points_scaled,patch_valid_prob,patch_features,patch_curve_connectivity)
                
                # filter by 2d panel
                flatten_pred=infer_out["flatten_pred"]
                for p_id in flatten_pred.keys():
                    edges=flatten_pred[p_id]["edge_points"]
                    keep_mask=detect_extra_edges_by_cost(edges)
                    if keep_mask is None: continue
                    connected_curves_ids=patch_curve_connectivity[p_id].nonzero().reshape(-1).tolist()
                    
                    for i in range(len(keep_mask)):
                        if not keep_mask[i]:
                            # print(f"datapoint {names[batch_idx]} patch {p_id} curve {connected_curves_ids[i]} is filtered")
                            patch_curve_connectivity[p_id][connected_curves_ids[i]]=0.0
                            patch_curve_similarity[p_id][connected_curves_ids[i]]=0.0
                            flag=True
            
            
            # ---------------------------------------------------------------just for debug-------------------------------------
            # patch_curve_similarity,curve_points,curve_valid_prob,curve_features,patch_points,patch_points_scaled,patch_valid_prob,patch_features,patch_curve_connectivity=\
            #     self.extract_topo_step1(patch_curve_similarity,curve_points,curve_valid_prob,curve_features,
            #                 patch_points,patch_points_scaled,patch_valid_prob,patch_features)
            # patch_curve_connectivity=patch_curve_similarity>self.th_valid_patch_curve_sim
            # -------------------------------------------------------------------------------------------------------------------
            
            
            infer_out=self.get_edges_all(patch_curve_similarity,curve_points,curve_valid_prob,curve_features,patch_points,patch_points_scaled,patch_valid_prob,patch_features,patch_curve_connectivity)
            
            out_batch.append(infer_out)
            

        return out_batch   
    
    
            
    def forward(self,curve_features, patch_features, curve_indices, patch_indices,PC_mat):
        bs=curve_features.shape[0]
        dim=768
        out_batch=[]
        for batch_idx in range(bs):
            out={}
            c_curve_f=curve_features[batch_idx]
            c_patch_f=patch_features[batch_idx]
            
            c_curve_ids=curve_indices[batch_idx]
            c_patch_ids=patch_indices[batch_idx]
            
            c_PC_mat=PC_mat[batch_idx]
            
            # --------------------------------------------------------------------------------
            panel_num=c_PC_mat.shape[0]
            
            # 1) 先把每个面板的索引一次性算好（避免在 6 层里重复 nonzero / index）
            panel_patch_ids, panel_edge_ids = self._gather_panel_indices(
                panel_num, c_patch_ids, c_curve_ids, c_PC_mat
            )
            
            # 2) 一次 gather -> list
            patch_tokens_list = [c_patch_f[idx].reshape(1, -1) for idx in panel_patch_ids]  # 每个 [1, C]
            edges_list        = [c_curve_f[idxs]  for idxs in panel_edge_ids]               # 每个 [Le_i, C]

  
            # 3) pad + mask
            patch_tokens = torch.stack(patch_tokens_list, dim=0)                            # [N, 1, C]
            edges_token, edges_mask, edges_lens = self._pad_and_masks(edges_list)
            
            for i in range(12):
                # self-attn
                edges_token = self.edge_self_att[i](edges_token, key_padding_mask=edges_mask)

                # patch_tokens = self.patch_self_att[i](patch_tokens.reshape(1,-1,dim)).reshape(-1,1,dim)
                
                # with patch (cross-attn): Q=verts/edges, K=V=patch (长度=1，无需 mask)
                edges_token = self.edge_pc_cross_att[i](edges_token, patch_tokens)

                # p_fnn = self.patch_fnn[i](patch_tokens)
                # patch_tokens = self.patch_ffn_norm[i](patch_tokens + p_fnn)
                
                
                # FFN
                e_ffn = self.edge_fnn[i](edges_token)
                edges_token = self.edge_ffn_norm[i](edges_token + e_ffn)
                
            # 5) 解码（按各自真实长度切回，再过解码器）———— 保证适配你当前 EdgeDecoder/verts_decoder
            # trans_pred = self.trans_decoder(patch_tokens).squeeze(1)  # [N, 7]
            trans_pred = self.scale_decoder(patch_tokens).squeeze(1)  # [N, 1]
    
            for panel_idx in range(panel_num):
                Le_b = edges_lens[panel_idx].item()

                e_tok_b = edges_token[panel_idx, :Le_b, :]         # [Le_b, C]

                out[panel_idx]={
                    "edges_pred":    self.edge_decoder(e_tok_b),                  # 注意下文的 EdgeDecoder 事项
                    # "trans_pred":    trans_pred[panel_idx],                                # [7]
                    "scale_pred":    trans_pred[panel_idx],                                # [1]
                }
            out_batch.append(out)
        
        
        return out_batch
       
if __name__=="__main__":
    import torch
    print(torch.cuda.is_available())