import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import rotation as rotation_tools

class CrossEntropyLossModule(nn.Module):
    def __init__(self):

        super(CrossEntropyLossModule, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        targets = targets.view(-1)       
        return self.loss_fn(logits, targets)


class L2LossModule(nn.Module):
    def __init__(self, reduction='mean'):
        """
        参数:
            reduction: 'mean'（默认）返回标量；'sum' 或 'none' 也可选
        """
        super(L2LossModule, self).__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        """
        输入:
            pred: Tensor, 形状为 (N, C)
            target: Tensor, 形状为 (N, C)

        输出:
            scalar 或 (N, C) 的 loss，根据 reduction 而定
        """
            
        return self.loss_fn(pred, target)

class EdgesParamsLoss(nn.Module):
    def __init__(self, args):
        super(EdgesParamsLoss, self).__init__()
        self.type_param_slices = {
            1: slice(0, 3),                 # circle
            2: slice(3, 5),                 # quadratic
            3: slice(5, 9)                  # cubic
        }

    def forward(self, types, pred, gt):
        parms = pred["parm"]                   

        total_loss = parms.new_tensor(0.0)
        valid = 0

        for i, t in enumerate(types):
            if t == 0:
                continue
            else:
                sl = self.type_param_slices[t]
                pred_i = parms[i, sl]                               
                gt_i = gt[i]
                if t == 1:
                    mse = F.mse_loss(pred_i[:1], gt_i[:1], reduction="mean")
                    bce = F.binary_cross_entropy_with_logits(pred_i[1:], gt_i[1:], reduction="mean")
                    total_loss = total_loss + (mse + 0.1*bce)
                else:
                    total_loss = total_loss + F.mse_loss(pred_i, gt_i, reduction="mean")
                valid += 1

        if valid == 0:
            # 非常神秘，不这么写会卡死
            return parms[..., :1].sum() * 0.0
        
        return total_loss / valid

class EdgePointsLoss(nn.Module):
    def __init__(self, args):
        super(EdgePointsLoss, self).__init__()
        self.args = args
        self.points_per_curve = args.points_per_edge

    def forward(self, pred, gt):
        distance_forward = (pred - gt).square().sum(-1).mean(-1).mean(-1)
        distance_backward = (torch.flip(pred, dims=(1,)) - gt).square().sum(-1).mean(-1).mean(-1)
        loss_geometry = torch.min(distance_forward, distance_backward)
        return loss_geometry

class TransLoss(nn.Module):
    def __init__(self, args):
        super(TransLoss, self).__init__()
        self.args = args
    
    def forward(self, pred, gt_translation, gt_rotation, gt_scale):
        """
        pred: [7] tensor, [tx, ty, tz, rx, ry, rz, s]
        gt_translation: [3]
        gt_rotation: [3]
        gt_scale: [1]
        """
        
        loss_t = F.mse_loss(pred[:3], gt_translation, reduction="mean")
        loss_s = (pred[6:]-gt_scale).pow(2).mean()

        R_pred = rotation_tools.euler_xyz_to_R_tensor(pred[3:6])
        R_gt   = rotation_tools.euler_xyz_to_R_tensor(gt_rotation)
        R_rel  = R_gt.T @ R_pred
        tr     = R_rel[0,0] + R_rel[1,1] + R_rel[2,2]
        # cos_th = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
        # theta  = torch.acos(cos_th)            # 真实旋转角
        # loss_r = (theta ** 2)                  # 或 theta 本身；若做均值就 .mean()
        
        # R_rel = R_gt.T @ R_pred   # (..., 3, 3)
        loss_r = (3.0 - tr)            # == 4 * sin^2(theta/2), ∈ [0, 4]
        
        # # loss_r = torch.norm(R_rel - torch.eye(3, device=R_rel.device), p='fro')**2         
        # if loss_r.isnan().item():
        #     print("Warning: loss_r is NaN, setting to 0")
        #     loss_r = torch.tensor(0.0, device=R_rel.device)
        return loss_t/100, loss_r, loss_s/100    
        

class FlattenLoss(nn.Module):
    def __init__(self, args):
        super(FlattenLoss, self).__init__()
        self.args = args
        self.cal_edges_classify_loss = CrossEntropyLossModule()
        self.cal_vertices_loss = L2LossModule(reduction='mean')
        self.cal_edges_params_loss = EdgesParamsLoss(args)
        self.cal_edges_points_loss = EdgePointsLoss(args)
        # self.cal_trans_loss = TransLoss(args)
        self.edge_type_to_idx={
            "line": 0,
            "circle": 1,
            "quadratic": 2,
            "cubic": 3,
            "none": 4
        }

    def forward(self, pred, gt_edge_points, gt_panel_scale):
        bs = len(pred)
        assert bs == len(gt_edge_points) == len(gt_panel_scale)

        edges_pred_all = []
        edges_gt_all = []
        scales_pred_all = []
        scales_gt_all = []

        for b in range(bs):
            cur_pred = pred[b]
            cur_gt_pts = gt_edge_points[b]
            cur_gt_scale = gt_panel_scale[b]

            for pid in sorted(cur_pred.keys()):
                cp = cur_pred[pid]

                # (E, 50, 2) 预测与GT
                e_pred = cp["edges_pred"]["edge_points"]
                e_gt   = cur_gt_pts[pid]

                # 标量 scale（有时是形如 [1] 的tensor，统一取标量）
                s_pred = cp["scale_pred"][0]
                s_gt   = cur_gt_scale[pid]

                edges_pred_all.append(e_pred)
                edges_gt_all.append(e_gt)
                scales_pred_all.append(s_pred)
                scales_gt_all.append(s_gt)


        # 拼接后一次性计算
        edges_pred_all = torch.cat(edges_pred_all, dim=0)  # (sum_E, 50, 2)
        edges_gt_all   = torch.cat(edges_gt_all,   dim=0)  # (sum_E, 50, 2)

        edge_loss = self.cal_edges_points_loss(edges_pred_all, edges_gt_all)

        scales_pred_all = torch.stack(scales_pred_all) # (N_panel,)
        scales_gt_all   = torch.stack(scales_gt_all)   # (N_panel,)
        scale_loss = (scales_pred_all - scales_gt_all).pow(2).mean()

        return {
            "edge_loss_geometry": edge_loss,
            "scale_loss": scale_loss,
        }