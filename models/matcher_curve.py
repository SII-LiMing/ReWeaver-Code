# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
import numpy as np
#from pytorch3d.loss import chamfer_distance
from typing import List, Dict
from config import ComplexStitchConfig
TList = List[Tensor]
TDict = Dict[str, Tensor]

def curve_distance(src_points, tgt_points):
  distance_forward = (src_points - tgt_points).square().sum(-1).mean(-1).view(-1,1)
  distance_backward = (torch.flip(src_points, dims=(1,)) - tgt_points).square().sum(-1).mean(-1).view(-1,1)
  return torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values


@torch.jit.script
def pairwise_shape_chamfer(src_shapes, target_shapes):
  pairwise_distance = []
  for i in range(target_shapes.shape[0]):  #typically num_queries:100
    pairwise_distance.append(curve_distance(target_shapes[i].unsqueeze(0), src_shapes)) #, 
  return torch.stack(pairwise_distance).transpose(0,1)#.sqrt() #distance normalized to single point



class HungarianMatcher_Curve(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, batch_size: int, cost_class: float = 1, cost_position: float = 1,  flag_eval: bool = False, val_th: float = 0.5):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_position = cost_position
        self.flag_eval = flag_eval
        self.val_th = val_th

    @torch.no_grad()
    def forward(self, outputs:TDict, target_curves_list:TList):
        """ Performs the matching
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        # pred_curve_points
        bs, num_queries = outputs["pred_curve_points"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        out_valid_prob = outputs["pred_curve_logits"].softmax(-1)  # [batch_size, num_queries, 2], valid or not
        assert(len(out_valid_prob.shape) == 3 and out_valid_prob.shape[2] == 2)
        out_curve_points_position = outputs["pred_curve_points"]#.flatten(0, 1)  # [batch_size, num_queries, 100, 3]
        batch_size=len(target_curves_list)
        indices = []
        for sample_batch_idx in range(batch_size):
          # Compute the classification cost. Contrary to the loss, we don't use the NLL,
          # but approximate it in 1 - proba[target class].
          # The 1 is a constant that doesn't change the matching, it can be ommitted.
          tgt_ids = torch.zeros(target_curves_list[sample_batch_idx].shape[0], dtype=torch.long)          
          if not self.flag_eval: 
            cost_class = -(out_valid_prob[sample_batch_idx][:, torch.zeros_like(tgt_ids)] + 1e-6).log()
            
            cost_curve_geometry = pairwise_shape_chamfer(out_curve_points_position[sample_batch_idx], target_curves_list[sample_batch_idx])
            # 将长度作为权重，可以考虑
            # cost_curve_geometry *= target_curves_list[sample_batch_idx]['curve_length_weighting'].view(1,-1)

            # Final cost matrix
            C = self.cost_position*cost_curve_geometry + self.cost_class * cost_class
            C = C.view(num_queries, -1).cpu()

            res_ass = linear_sum_assignment(C)
            indices.append(res_ass)
            # for i, j in res_ass:
            #   print(i," ",j)
          else:
            # Compute the chamfer distance between curves in batch

            valid_id = torch.where(out_valid_prob[sample_batch_idx][:,0] > self.val_th)

            # C = C.view(num_queries, -1).cpu()
            if valid_id[0].shape[0] == 0:
                tmp = np.array([], dtype=np.int64)
                indices.append((tmp,tmp))
            else:
                cost_curve_geometry = pairwise_shape_chamfer(out_curve_points_position[sample_batch_idx][valid_id], target_curves_list[sample_batch_idx])


                # cost_curve_geometry *= target_curves_list[sample_batch_idx]['curve_length_weighting'].view(1,-1)

                # Final cost matrix
                C = self.cost_position*cost_curve_geometry
                C = C.view(valid_id[0].shape[0], -1).cpu()
                (pred_id, tar_id) = linear_sum_assignment(C)
                pred_id = valid_id[0][pred_id]
                indices.append((pred_id, tar_id))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



def build_matcher_curve(batch_size,args:ComplexStitchConfig=None, flag_eval = False):
  
  if not flag_eval:
    return HungarianMatcher_Curve(batch_size, cost_class=args.class_loss_coef, cost_position=args.curve_geometry_loss_coef)
  else:
    return HungarianMatcher_Curve(batch_size, cost_class=0.0, cost_position=1, flag_eval = True, val_th = args.val_th)
