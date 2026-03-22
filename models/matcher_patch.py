# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
from typing import List
from utils.utils import knn_point
from config import ComplexStitchConfig
import numpy as np

TList = List[Tensor]
def chamfer_distance_patch(src_points, tgt_points):
  pairwise_distance = torch.cdist(src_points, tgt_points, p=2.0).square()
  assert(pairwise_distance.shape[0] > 0) #pairwise_distance.shape = [n_prediction, target_points_num, 100(10*10)]
  s2t = pairwise_distance.min(-1).values.mean(-1)
  t2s = pairwise_distance.min(-2).values.mean(-1)
  return (s2t + t2s) / 2.0


# @torch.jit.script  #not available for flag_batch_cd
def pairwise_shape_chamfer_patch(src_shapes, target_shapes: TList):
    pairwise_distance = []
    ll = torch.tensor([len(t) for t in target_shapes])
    assert(ll.min() > 0)
    for item in target_shapes:  #typically num_queries:100
      pairwise_distance.append(chamfer_distance_patch(item.unsqueeze(0), src_shapes)) 

    return torch.stack(pairwise_distance).transpose(0,1) #distance normalized to single point

@torch.jit.script
def emd_by_id(gt: Tensor, pred: Tensor, gtid: Tensor, points_per_patch_dim: int):
  #gt shape: N/1, 400, 3
  #pred shape: N, 400, 3
  gt_batch = gt[:, gtid, :].view(len(gt), -1, points_per_patch_dim * points_per_patch_dim, 3)
  pred_batch = pred.view(len(pred), 1, points_per_patch_dim * points_per_patch_dim, 3)
  dist = (gt_batch - pred_batch).square().sum(-1).mean(-1).min(-1).values
  return dist

class HungarianMatcher_Patch(nn.Module):
    def __init__(self, cost_class: float = 1, cost_position: float = 1, flag_patch_emd = False, flag_eval=False, val_th = 0.3, flag_patch_uv = False, dim_grid = 10):
        super().__init__()
        self.cost_class = cost_class
        self.cost_position = cost_position
        # self.batch_size = batch_size
        self.flag_eval = flag_eval
        self.flag_patch_uv = flag_patch_uv
        self.dim_grid = dim_grid
        self.val_th = val_th
        self.emd_idlist = []
        self.flag_patch_emd = False
        
        # if flag_patch_emd:
        #   self.flag_patch_emd = True
        #   base = torch.arange(dim_grid * dim_grid).view(dim_grid, dim_grid)
        #   for i in range(4):
        #     self.emd_idlist.append(torch.rot90(base, i, [0,1]).flatten())
          
        #   base_t = base.transpose(0,1)
        #   for i in range(4):
        #     self.emd_idlist.append(torch.rot90(base_t, i, [0,1]).flatten())
        #   self.emd_idlist = torch.cat(self.emd_idlist) #800
        
        # if flag_patch_uv:
        #   self.emd_idlist_u = []
        #   self.emd_idlist_v = []
        #   base = torch.arange(dim_grid * dim_grid).view(dim_grid,  dim_grid)
        #   #set idlist u
        #   for i in range(dim_grid):
        #     cur_base = base.roll(shifts=i, dims = 0)
        #     for i in range(0,4,2):
        #       self.emd_idlist_u.append(torch.rot90(cur_base, i, [0,1]).flatten())
            
        #     cur_base = cur_base.transpose(0,1)
        #     for i in range(1,4,2):
        #       self.emd_idlist_u.append(torch.rot90(cur_base, i, [0,1]).flatten())
          
        #   self.emd_idlist_u = torch.cat(self.emd_idlist_u)

    # def emd(self, src_points, tgt_points, uclosed, vclosed):
    #   #src is gt here
    #   if not self.flag_patch_uv:
    #     return emd_by_id(src_points, tgt_points, self.emd_idlist, self.dim_grid)
    #   if uclosed:
    #     return emd_by_id(src_points, tgt_points, self.emd_idlist_u, self.dim_grid)
    #   return emd_by_id(src_points, tgt_points, self.emd_idlist, self.dim_grid)
      
    # def pairwise_shape_emd(self, src_shapes, target_shapes, target_uclosed, target_vclosed):
    #   #assume that either u closed or v closed
    #   pairwise_distance = []
    #   assert(len(target_shapes) == len(target_uclosed) and len(target_shapes) == len(target_vclosed))

    #   for i in range(len(target_shapes)):
    #     pairwise_distance.append(self.emd(target_shapes[i].unsqueeze(0), src_shapes, target_uclosed[i], target_vclosed[i])) 
      
        

    #   return torch.stack(pairwise_distance).transpose(0,1) #distance normalized to single point
    
    @torch.no_grad()
    def forward(self, outputs, patch_points_batch):
        
        bs, num_queries = outputs["pred_patch_points"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_valid_prob = outputs["pred_patch_logits"].softmax(-1)  # [batch_size, num_queries, 2]
        assert(len(out_valid_prob.shape) == 3 and out_valid_prob.shape[2] == 2)
        out_patch_points_position = outputs["pred_patch_points"]#.flatten(0, 1)  # [batch_size, num_queries, 100, 3]
        
        indices = []
        for sample_batch_idx in range(bs):
          # Compute the classification cost. Contrary to the loss, we don't use the NLL,
          # but approximate it in 1 - proba[target class].
          # The 1 is a constant that doesn't change the matching, it can be ommitted.
          patch_points=patch_points_batch[sample_batch_idx]
          
          
          # if self.flag_patch_emd:
          #   uclosed_patch_gt = target_patches_list[sample_batch_idx]['u_closed']
          #   vclosed_patch_gt = target_patches_list[sample_batch_idx]['v_closed']
          
          if not self.flag_eval:
            cost_class = - (out_valid_prob[sample_batch_idx][:, torch.zeros(len(patch_points),dtype=torch.long)] + 1e-6).log()

            # Compute the chamfer distance between curves in batch
            if not self.flag_patch_emd:
              cost_patch_geometry = pairwise_shape_chamfer_patch(out_patch_points_position[sample_batch_idx], patch_points)
            # else:
            #   cost_patch_geometry = self.pairwise_shape_emd(out_patch_points_position[sample_batch_idx], patch_points, uclosed_patch_gt, vclosed_patch_gt)
            
            # 根据面积做一个缩放
            # cost_patch_geometry *= target_patches_list[sample_batch_idx]['patch_area_weighting'].view(1, -1)
            
            # Final cost matrix
            C = self.cost_position*cost_patch_geometry + self.cost_class * cost_class
            C = C.view(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
          else:
            valid_id = torch.where(out_valid_prob[sample_batch_idx][:,0] > self.val_th)
            if valid_id[0].shape[0] == 0:
              continue
            if not self.flag_patch_emd:
              cost_patch_geometry = pairwise_shape_chamfer_patch(out_patch_points_position[sample_batch_idx][valid_id], patch_points)
            # else:
            #   cost_patch_geometry = self.pairwise_shape_emd(out_patch_points_position[sample_batch_idx][valid_id], target_patches_list[sample_batch_idx]['patch_points'], uclosed_patch_gt, vclosed_patch_gt)
            
            # cost_patch_geometry *= target_patches_list[sample_batch_idx]['patch_area_weighting'].view(1, -1)
            # Final cost matrix
            C = self.cost_position*cost_patch_geometry
            if valid_id[0].shape[0] == 0:
                tmp = np.array([], dtype=np.int64)
                indices.append((tmp,tmp))
            else:
                C = C.view(valid_id[0].shape[0], -1).cpu()
                (pred_id, tar_id) = linear_sum_assignment(C)
                pred_id = valid_id[0][pred_id]
                indices.append((pred_id, tar_id))

        if len(indices) != 0:
          return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
          return []


def build_matcher_patch(args:ComplexStitchConfig,flag_eval = False):
  # TODO: 这里为什么需要 batch size?
  if not flag_eval:
    return HungarianMatcher_Patch(cost_class=args.class_loss_coef, 
                                  cost_position=args.patch_geometry_loss_coef, 
                                  # flag_patch_emd = args.patch_emd,                # ?
                                  # flag_patch_uv=args.patch_uv,               # ?  
                                  dim_grid = args.points_per_patch_dim
                                  )
  else:
    return HungarianMatcher_Patch(cost_class=0.0, cost_position=1.0, using_prob_in_matching=False, single_dir_patch_chamfer=False,flag_batch_cd=args.batch_cd, flag_eval = True, val_th = args.val_th, flag_patch_emd = args.patch_emd, flag_patch_uv=args.patch_uv, dim_grid = args.points_per_patch_dim)
