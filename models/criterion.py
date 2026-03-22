import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import trimesh

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [100.0 * torch.ones([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class SetCriterion_Curve(nn.Module):
    """ This class computes the loss for DETR-Curve.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth curve and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and curve geometry)
    """
    def __init__(self, matcher, eos_coef): #num_classes
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = 4 #'Circle' 'BSpline' 'Line' 'Ellipse'
        self.matcher = matcher
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2) #non-empty, empty
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    
    def loss_valid_labels(self, outputs, targets, indices, num_corners, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_curve_logits' in outputs
        src_logits = outputs['pred_curve_logits']

        idx = self._get_src_permutation_idx(indices)
        #target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_o = torch.cat([torch.zeros(J.shape, device=src_logits.device) for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = 0 #target_classes_o, for corner points having matching gt, target label set to 0

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device))
        losses = {'curve_loss_ce': loss_ce}

        if log:
            losses['valid_class_accuracy'] = accuracy(src_logits[idx], target_classes_o)[0]
            losses['valid_class_accuracy_overall'] = accuracy(src_logits.view(-1,2), target_classes.view(-1))[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_curves):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_curve_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_geometry(self, outputs, targets, indices, num_curves):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_curve_points' in outputs
        idx = self._get_src_permutation_idx(indices) #only src
        src_curve_points = outputs['pred_curve_points'][idx]
        target_curve_points = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        # target_curve_length_weight = torch.cat([t['curve_length_weighting'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # assert(target_curve_length_weight.shape[0] == target_curve_points.shape[0])
        assert(src_curve_points.shape == target_curve_points.shape)
        
        #open curve
        distance_forward = (src_curve_points - target_curve_points).square().sum(-1).mean(-1).view(-1,1)
        distance_backward = (torch.flip(src_curve_points, dims=(1,)) - target_curve_points).square().sum(-1).mean(-1).view(-1,1)
        loss_geometry = torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values

        # assert(loss_geometry.shape == target_curve_length_weight.shape)
        # loss_geometry *= target_curve_length_weight
        losses = {}
        losses['curve_loss_geometry'] = loss_geometry.sum() / num_curves        
        return losses

    def get_cd(self, outputs, targets, indices, num_curves):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_curve_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        pred_logits = outputs['pred_curve_logits'][0]
        device = pred_logits.device
        if indices[0][0].shape[0] != 0:
          src_curve_points = outputs['pred_curve_points'][idx]
          target_curve_points = torch.cat([t['curve_points'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_curve_points.device)
          target_curve_length_weight = torch.cat([t['curve_length_weighting'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_curve_points.device)
          assert(target_curve_length_weight.shape[0] == target_curve_points.shape[0])
          is_target_curve_closed = torch.cat([t['is_closed'][i] for t, (_, i) in zip(targets, indices)])
          assert(src_curve_points.shape == target_curve_points.shape)
          if(True):
            #compute chamfer distance
            pairwise_distance = torch.cdist(src_curve_points, target_curve_points, p=2.0) #in shape [batch_size, src_curve_number, tgt_curve_number)
            s2t = pairwise_distance.min(-1).values.mean(-1)
            t2s = pairwise_distance.min(-2).values.mean(-1)        
            loss_geometry = (s2t + t2s) / 2.0
          else:
            #open curve
            distance_forward = (src_curve_points - target_curve_points).square().sum(-1).mean(-1).view(-1,1).sqrt()
            distance_backward = (torch.flip(src_curve_points, dims=(1,)) - target_curve_points).square().sum(-1).mean(-1).view(-1,1).sqrt()
            loss_geometry = torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values
            #print(loss_geometry.shape)
            #print("src_curve_points.shape = ", src_curve_points.shape)
            for i in range(is_target_curve_closed.shape[0]):
              if(is_target_curve_closed[i]):
                tgt_possible_curves = cyclic_curve_points(target_curve_points[i].unsqueeze(0)) #[66, 34, 3]
                loss_geometry[i] = (tgt_possible_curves - src_curve_points[i:i+1]).square().sum(-1).mean(-1).min()
          assert(loss_geometry.shape == target_curve_length_weight.shape)
          losses = {}
          losses['cd'] = loss_geometry.sum() / max(1, loss_geometry.shape[0])   
          #corner
          close_indices = torch.where(loss_geometry < args.dist_th)
          pred_labels = pred_logits.softmax(-1)
          pred_valid_id = torch.where(pred_labels[:, 0]>args.val_th)
          losses['precision'] = torch.tensor(close_indices[0].shape[0] / max(1, pred_valid_id[0].shape[0]), device=device)
          losses['recall'] = torch.tensor(close_indices[0].shape[0] / num_curves, device = device)
          
          #for classification
          src_logits = outputs['pred_curve_type']
          idx = self._get_src_permutation_idx(indices)
          target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
          losses['class_accuracy'] = accuracy(src_logits[idx], target_classes)[0]
        else:
          losses = {}
          losses['cd'] = torch.tensor(0.0, device = device)
          losses['precision'] = torch.tensor(0.0, device = device)
          losses['recall'] = torch.tensor(0.0, device = device)
          losses['class_accuracy'] = torch.tensor(0.0, device = device)
        losses['fscore'] = 2 * losses['precision'] * losses['recall'] / (losses['precision'] + losses['recall'] + 1e-6)
        return losses

    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_corners):
        loss_map = {
            'labels': self.loss_valid_labels,
            'cardinality': self.loss_cardinality,    # no_grad 
            'geometry': self.loss_geometry,
            'cd': self.get_cd,                       # for eval
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss == 'geometry':
          return loss_map[loss](outputs, targets, indices, num_corners)
        else:
          return loss_map[loss](outputs, targets, indices, num_corners)
          

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_curves = sum(t.shape[0] for t in targets)
        # num_curves = torch.as_tensor([num_curves], dtype=torch.float, device=next(iter(outputs.values())).device)
        # torch.distributed.all_reduce(num_curves)
        # num_curves = torch.clamp(num_curves / dist.get_world_size(), min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in ["labels","geometry"]:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_curves))
        return losses, indices

class SetCriterion_Patch(nn.Module):
    """ This class computes the loss for DETR-Patch.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth patch and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and patch geometry)
    """
    def __init__(self, matcher, eos_coef): #num_classes
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = 6 # Cylinder, Torus, BSpline, Plane, Cone, Sphere
        self.matcher = matcher
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2) #non-empty, empty
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        # if args.patch_emd:
        #   self.emd_idlist = []
        #   base = torch.arange(points_per_patch_dim * points_per_patch_dim).view(points_per_patch_dim, points_per_patch_dim)
        #   for i in range(4):
        #     self.emd_idlist.append(torch.rot90(base, i, [0,1]).flatten())
          
        #   base_t = base.transpose(0,1)
        #   for i in range(4):
        #     self.emd_idlist.append(torch.rot90(base_t, i, [0,1]).flatten())

        #   self.emd_idlist = torch.cat(self.emd_idlist)
        # if args.patch_uv:
        #   self.emd_idlist_u = []
        #   self.emd_idlist_v = []
        #   base = torch.arange(points_per_patch_dim * points_per_patch_dim).view(points_per_patch_dim, points_per_patch_dim)
        #   #set idlist u
        #   for i in range(points_per_patch_dim):
        #     cur_base = base.roll(shifts=i, dims = 0)
        #     for i in range(0,4,2):
        #       self.emd_idlist_u.append(torch.rot90(cur_base, i, [0,1]).flatten())
            
        #     cur_base = cur_base.transpose(0,1)
        #     for i in range(1,4,2):
        #       self.emd_idlist_u.append(torch.rot90(cur_base, i, [0,1]).flatten())
          
        #   self.emd_idlist_u = torch.cat(self.emd_idlist_u)

        # if args.eval_param:
        #   cp_distance = ComputePrimitiveDistance(reduce = True)
        #   self.routines = {
        #     5: cp_distance.distance_from_sphere,
        #     0: cp_distance.distance_from_cylinder,
        #     4: cp_distance.distance_from_cone,
        #     3: cp_distance.distance_from_plane,
        #   }
        #   self.sqrt = True


    def loss_valid_labels(self, outputs, targets, indices, num_patches, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_patch_logits' in outputs
        src_logits = outputs['pred_patch_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([torch.zeros(J.shape, device=src_logits.device) for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = 0 #target_classes_o, for corner points having matching gt, target label set to 0

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device))
        losses = {'patch_loss_ce': loss_ce}

        if log:
          losses['valid_class_accuracy'] = accuracy(src_logits[idx], target_classes_o)[0]
          losses['valid_class_accuracy_overall'] = accuracy(src_logits.view(-1,2), target_classes.view(-1))[0]
        return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_patches):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_patch_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_geometry(self, outputs, targets, indices, num_patches):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        
        assert 'pred_patch_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_patch_points = outputs['pred_patch_points'][idx]

        target_patch_points_list = [[t[j] for j in i.numpy().tolist()] for t, (_, i) in zip(targets, indices)]#torch.cat(, dim=0)
        
        # target_patch_area_weighting = torch.cat([t['patch_area_weighting'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # assert(target_patch_area_weighting.shape[0] == src_patch_points.shape[0])
        target_patch_points = target_patch_points_list[0]
        for i in range(1, len(target_patch_points_list)):
          target_patch_points += target_patch_points_list[i]


        assert(len(src_patch_points) == len(target_patch_points))
        # assert(target_patch_area_weighting.shape[0] == len(target_patch_points))
        #compute chamfer distance
        loss_geometry = []
        loss_patch_lap = []
        loss_output_normal_diff = []
        loss_output_normal_tangent = []
        for patch_idx in range(len(target_patch_points)):
          patch_distance = torch.cdist(src_patch_points[patch_idx], target_patch_points[patch_idx], p=2.0).square() 
          assert(len(patch_distance.shape) == 2)
          # loss_geometry.append(target_patch_area_weighting[patch_idx]*(patch_distance.min(0).values.mean() + 0.2*patch_distance.min(-1).values.mean()) / 1.2)
          loss_geometry.append((patch_distance.min(0).values.mean() + 0.2*patch_distance.min(-1).values.mean()) / 1.2)
          
          # if args.patch_lap:
          #   x_minus = src_patch_points[patch_idx][outputs['mask_x_minus']]
          #   x_plus = src_patch_points[patch_idx][outputs['mask_x_plus']]
          #   y_minus = src_patch_points[patch_idx][outputs['mask_y_minus']]
          #   y_plus = src_patch_points[patch_idx][outputs['mask_y_plus']]
          #   loss_patch_lap.append((src_patch_points[patch_idx] - (x_minus + x_plus + y_minus + y_plus) / 4.0).norm(dim = -1).mean())
          # if args.patch_lapboundary:
          #   loss_patch_lap.append(torch.mm(outputs['mat_lapboundary'], src_patch_points[patch_idx]).norm(dim = -1).mean())
            
        losses = {}
        losses['patch_loss_geometry'] = sum(loss_geometry) / num_patches
        # if args.patch_lap or args.patch_lapboundary:
        #   losses['loss_patch_lap'] = sum(loss_patch_lap) / num_patches
        return losses
    
    def loss_single_cd(self, outputs, targets, indices, num_patches):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_patch_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_patch_points = outputs['pred_patch_points'][idx]
        # print(len(targets))
        target_patch_points_list = [[t['patch_pcs'][j] for j in i.numpy().tolist()] for t, (_, i) in zip(targets, indices)]#torch.cat(, dim=0)
        target_patch_points = target_patch_points_list[0]
        for i in range(1, len(target_patch_points_list)):
          target_patch_points += target_patch_points_list[i]
        if True:
          #batch computation          
          target_point_clouds_length = torch.tensor([len(p) for p in target_patch_points], device=src_patch_points.device)
          flag_equasize = False
          if len(target_point_clouds_length.unique()) == 1:
            flag_equasize = True
          
          target_patch_points_batch = list_to_padded(target_patch_points, (target_point_clouds_length.max(), 3), equisized = flag_equasize)
          target_nn = knn_points(target_patch_points_batch, src_patch_points, lengths1=target_point_clouds_length)
          target_cd = target_nn.dists[...,0]
          loss_geometry_batch = (target_cd.sum(-1) / target_point_clouds_length).mean()
          losses = {}
          losses['loss_single_cd'] = loss_geometry_batch
          return losses
 
    def get_cd(self, outputs, targets, indices, num_patches):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_patch_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_patch_points = outputs['pred_patch_points'][idx]
        print('indices shape: ', indices[0][0].shape)
        target_patch_points_list = [[t['patch_points'][j] for j in i.numpy().tolist()] for t, (_, i) in zip(targets, indices)]#torch.cat(, dim=0)
        target_patch_area_weighting = torch.cat([t['patch_area_weighting'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_patch_points.device)
        assert(target_patch_area_weighting.shape[0] == src_patch_points.shape[0])
        target_patch_points = target_patch_points_list[0]
        for i in range(1, len(target_patch_points_list)):
          target_patch_points += target_patch_points_list[i]
        assert(len(src_patch_points) == len(target_patch_points))
        assert(target_patch_area_weighting.shape[0] == len(target_patch_points))
        loss_geometry = []
        for patch_idx in range(len(target_patch_points)):
          patch_distance = torch.cdist(src_patch_points[patch_idx], target_patch_points[patch_idx].to(src_patch_points.device), p=2.0) #in shape [src_patch_points, tgt_patch_points]
          assert(len(patch_distance.shape) == 2)
          if(args.single_dir_patch_chamfer): #default: false
            loss_geometry.append(target_patch_area_weighting[patch_idx]*patch_distance.min(0).values.mean())
          else:
            loss_geometry.append((patch_distance.min(0).values.mean() + patch_distance.min(-1).values.mean()) / 2.0)
        
        losses = {}
        losses['cd'] = sum(loss_geometry) / len(loss_geometry)

        if args.eval_res_cov:
          target_patch_pc_list = [[t['patch_pcs'][j] for j in i.numpy().tolist()] for t, (_, i) in zip(targets, indices)]#torch.cat(, dim=0)
          src_logits = outputs['closed_patch_logits'][idx]
          src_uclosed = src_logits[:,0] < src_logits[:,1]
          target_classes = torch.cat([t["u_closed"][J] for t, (_, J) in zip(targets, indices)])

          if args.eval_param:
            src_with_param = outputs['pred_patch_with_param'][idx]
            src_type_logits = outputs['pred_patch_type'][idx]
            src_param = outputs['pred_patch_param'][idx]

          target_patch_pcs = target_patch_pc_list[0]
          for i in range(1, len(target_patch_pc_list)):
            target_patch_pcs += target_patch_pc_list[i]
          loss_res = []
          loss_res_filter = []
          patch_idx_filtered = []
          for patch_idx in range(len(target_patch_pcs)):
            if args.eval_param and src_with_param[patch_idx] > 0.5:
              para_dist = self.routines[torch.argmax(src_type_logits[patch_idx]).item()](target_patch_pcs[patch_idx], src_param[patch_idx], self.sqrt)
            #spline
            pts = src_patch_points[patch_idx].detach().cpu().numpy()
            
            #prediction
            pts, faces = get_patch_mesh_pts_faces(pts, args.points_per_patch_dim, args.points_per_patch_dim, src_uclosed[patch_idx],0, True, 0.05)
            mesh = trimesh.Trimesh(vertices = pts, faces = faces)
            
            #mesh version
            (closest_points,distances,triangle_id) = mesh.nearest.on_surface(target_patch_pcs[patch_idx].detach().cpu().numpy()) #here distance is squared norm
            if args.eval_param and src_with_param[patch_idx] > 0.5:
              loss_res.append(min(para_dist.item(), distances.mean()))
            else:
              loss_res.append(distances.mean())

            if loss_res[-1] < args.th_res:
              loss_res_filter.append(loss_res[-1])
              patch_idx_filtered.append(indices[0][0][patch_idx].item())

          #append unmatched id
          matched_id_set = set(indices[0][0].tolist())
          unmatched_id_set = set(range(outputs['pred_patch_points'][0].shape[0])) - matched_id_set
          patch_idx_filtered += list(unmatched_id_set)

          losses['res'] = torch.tensor(sum(loss_res) / len(loss_res))
          if len(loss_res_filter) > 0:
            losses['res_filter'] = torch.tensor(sum(loss_res_filter) / len(loss_res_filter))
          else:
            losses['res_filter'] = torch.tensor(0.0)
          
          losses['n_patch'] = torch.tensor(len(loss_res))
          losses['n_patch_filter'] = torch.tensor(len(loss_res_filter))
          losses['patch_idx_filter'] = patch_idx_filtered


        loss_geometry = torch.tensor(loss_geometry)
        close_indices = torch.where(loss_geometry < args.dist_th)
        pred_logits = outputs['pred_patch_logits'][0]
        device = pred_logits.device
        pred_labels = pred_logits.softmax(-1)
        pred_valid_id = torch.where(pred_labels[:, 0]>args.val_th)
        losses['precision'] = torch.tensor(close_indices[0].shape[0] / max(1, pred_valid_id[0].shape[0]), device=device)
        losses['recall'] = torch.tensor(close_indices[0].shape[0] / num_patches, device = device)
        losses['fscore'] = 2 * losses['precision'] * losses['recall'] / (losses['precision'] + losses['recall'] + 1e-6)

        assert 'pred_patch_type' in outputs
        src_logits = outputs['pred_patch_type']

        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)

        losses['type_class_accuracy'] = accuracy(src_logits[idx], target_classes)[0]
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # goto
    def get_loss(self, loss, outputs, targets, indices, num_patches, **kwargs):
        loss_map = {
            'labels': self.loss_valid_labels,
            'cardinality': self.loss_cardinality,
            'geometry': self.loss_geometry,
            'single_cd': self.loss_single_cd,
            'cd': self.get_cd,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_patches, **kwargs)

    def forward(self, outputs, patch_points):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, patch_points)
        if len(indices) == 0:
          return {}, []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_patches = sum(len(p) for p in patch_points)
        # num_patches = torch.as_tensor([num_patches], dtype=torch.float, device=next(iter(outputs.values())).device)
        # torch.distributed.all_reduce(num_patches)
        # num_patches = torch.clamp(num_patches /  dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        # for loss in ['labels','cardinality','geometry','single_cd','cd']:
        for loss in ['labels','geometry']:
          losses.update(self.get_loss(loss, outputs, patch_points, indices, num_patches))
        return losses, indices
  
def Curve_Corner_Matching(corner_predictions, curve_predictions,corners_gt,curves_gt, EV_mat, corner_indices, curve_indices,use_geo_loss=False, flag_round = False):
    #for samples in each batch seperately
    topo_correspondence_loss = []
    topo_geometry_loss = []
    topo_correspondence_acc = []
    
    bs=len(EV_mat)
    device = corner_predictions['pred_corner_logits'].device
    for i in range(bs):
      corner_predictions_topo_embed = corner_predictions['corner_topo_embed_curve'][i] #in shape [100, 192]
      curve_predictions_topo_embed = curve_predictions['curve_topo_embed_corner'][i] #in shape[100, 192]
    
      cur_corner_indices = corner_indices[i] #a tuple
      cur_curve_indices = curve_indices[i] #a tuple
      
      valid_corner_predictions_topo_embed = corner_predictions_topo_embed[cur_corner_indices[0]]
      valid_curve_predictions_topo_embed = curve_predictions_topo_embed[cur_curve_indices[0]]
      
      cur_EV_mat = EV_mat[i]
      cur_EV_mat = cur_EV_mat[cur_curve_indices[1]][:, cur_corner_indices[1]]
      
      curve_corner_similarity = torch.sigmoid(torch.mm(valid_curve_predictions_topo_embed, valid_corner_predictions_topo_embed.transpose(0,1))) #in shape [valid_open_curves, valid_corners]
      
      # curve_predictions['pred_curve_points']: (B,num_query,num_sampled_point,3)
      # curve_endpoints_position: (num_query,2,3)
      curve_endpoints_position = curve_predictions['pred_curve_points'][i][:,[0, -1],:]
      curve_endpoints_position = curve_endpoints_position[cur_curve_indices[0]]

      corner_position = corner_predictions['pred_corner_position'][i][cur_corner_indices[0]] #to mapped space
      
      
      # whether use_geo_loss:
      if True:
        num_curve=len(curve_endpoints_position)
        for curve_idx in range(num_curve):
          _, top2_indices = torch.topk(curve_corner_similarity[curve_idx], k=2)
          pred_end_point = corner_position[top2_indices]
          cur_curve_endpoints_position = curve_endpoints_position[curve_idx]
          diff1 = (pred_end_point - cur_curve_endpoints_position).square().sum(-1).mean(-1)
          diff2 = (pred_end_point - cur_curve_endpoints_position[[1, 0], :]).square().sum(-1).mean(-1)
          topo_geometry_loss.append(torch.min(diff1, diff2))
      else:
        topo_geometry_loss.append(torch.zeros(1, device = corner_predictions['pred_corner_logits'].device)[0])
      if cur_corner_indices[0].shape[0] != 0:
        topo_correspondence_loss.append(F.binary_cross_entropy(curve_corner_similarity.view(-1), cur_EV_mat.view(-1)))
        topo_correspondence_acc.append(100.0 * (1.0 - (torch.round(curve_corner_similarity)-cur_EV_mat).abs().mean()) )
      

    if(len(topo_geometry_loss) != 0):
      return sum(topo_geometry_loss) / len(topo_geometry_loss), sum(topo_correspondence_loss) / len(topo_correspondence_loss), sum(topo_correspondence_acc) / len(topo_correspondence_acc)
    else:
      return torch.tensor(0, device=device), torch.tensor(0, device=device), torch.tensor(100.0, device=device)
  
  

def Patch_Curve_Matching(curve_predictions, patch_predictions, curves_gt, patches_gt,PC_mat, curve_indices, patch_indices):
    #for samples in each batch seperately
    assert(len(patches_gt) == len(curves_gt))
    
    topo_correspondence_loss = []
    topo_correspondence_acc = []
    
    for i in range(len(patches_gt)):
      #compute pairwise dot product
      curve_predictions_topo_embed = curve_predictions['curve_topo_embed_patch'][i] 
      patch_predictions_topo_embed = patch_predictions['patch_topo_embed_curve'][i] 
      
      #select matched curve and corners
      cur_curve_indices = curve_indices[i] #a tuple
      cur_patch_indices = patch_indices[i] #a tuple
      if cur_curve_indices[0].shape[0] == 0:
        continue
      if cur_patch_indices[0].shape[0] == 0:
        continue
      
      valid_patch_predictions_topo_embed = patch_predictions_topo_embed[cur_patch_indices[0]]
      valid_curve_predictions_topo_embed = curve_predictions_topo_embed[cur_curve_indices[0]]
      
      cur_PC_mat = PC_mat[i]
      cur_PC_mat = cur_PC_mat[cur_patch_indices[1]][:, cur_curve_indices[1]]

      patch_curve_similarity = torch.sigmoid(torch.mm(valid_patch_predictions_topo_embed, valid_curve_predictions_topo_embed.transpose(0,1))) #in shape [valid_open_curves, valid_corners]

      topo_correspondence_loss.append(F.binary_cross_entropy(patch_curve_similarity.view(-1), cur_PC_mat.view(-1)))

      topo_correspondence_acc.append(100.0 * (1.0 - (torch.round(patch_curve_similarity)-cur_PC_mat).abs().mean()) )
      

    if(len(topo_correspondence_loss) != 0):
      return sum(topo_correspondence_loss) / len(topo_correspondence_loss), sum(topo_correspondence_acc) / len(topo_correspondence_acc)
    else:
      return torch.tensor(0, device=curve_predictions['pred_curve_logits'].device), torch.tensor(100.0, device=curve_predictions['pred_curve_logits'].device)

# def degree2_loss_soft(patch_corner_sim, patch_curve_sim, curve_corner_sim):
#   """
#   patch_corner_sim: (P, C)  ∈ [0,1]
#   patch_curve_sim : (P, E)  ∈ [0,1]
#   curve_corner_sim: (E, C)  ∈ [0,1]
#   """
#   P, C = patch_corner_sim.shape
#   _, E = patch_curve_sim.shape
#   S = curve_corner_sim  # (E, C)

#   losses = []
#   for p in range(P):
#     w_c = patch_corner_sim[p]      # (C,)
#     w_e = patch_curve_sim[p]       # (E,)
#     # “软子矩阵”的行/列和（只累计属于该 patch 的 corner/curve）
#     # 行和：对列求和，并用行权重 w_e 选择性约束
#     row_sum = (S * w_c.unsqueeze(0)).mean(dim=1)        # (E,)
#     row_target = 2.0 * w_e/C                             # 仅对被选中的行（w_e≈1）逼近 2
#     loss_rows = ((row_sum - row_target)**2 * w_e).sum() / (w_e.sum().clamp_min(1.0).detach())

#     # 列和：对行求和，并用列权重 w_c 选择性约束
#     col_sum = (S * w_e.unsqueeze(1)).mean(dim=0)        # (C,)
#     col_target = 2.0 * w_c/E
#     loss_cols = ((col_sum - col_target)**2 * w_c).sum()/ (w_c.sum().clamp_min(1.0).detach())
    
#     cnt_c = w_c.sum()
#     cnt_e = w_e.sum()
#     loss_balance = (cnt_c - cnt_e)**2 / (C+E)
  
#     losses.append(loss_rows + loss_cols + loss_balance)

#   losses = torch.stack(losses)
  
#   return losses.mean()
  
def Patch_Corner_Matching(corner_predictions, curve_predictions, patch_predictions, corners_gt, curves_gt, patches_gt,PC_mat,CK_mat,PK_mat, corner_indices, curve_indices, patch_indices): 
    #for samples in each batch seperately
    assert(len(patches_gt) == len(curves_gt))
    
    topo_correspondence_loss = []
    topo_correspondence_acc = []
    #topo loss
    curve_point_loss = []
    patch_close_loss = []
    curve_patch_loss = []
    zero_corners_examples = 0
    for i in range(len(patches_gt)):
      # num_gt_patches=len(patches_gt[i])
      # num_gt_corners = len(corners_gt[i])
      
      #no curves exists thus we do not have to compute
      if(corners_gt[i].shape[0] == 0 or corner_indices[i][0].shape[0]==0):
        zero_corners_examples += 1
        continue
      #compute pairwise dot product
      corner_predictions_topo_embed_patch = corner_predictions['corner_topo_embed_patch'][i] #in shape [100, 256]
      patch_predictions_topo_embed_corner = patch_predictions['patch_topo_embed_corner'][i] #in shape [100, 256]
      corner_predictions_topo_embed_curve = corner_predictions['corner_topo_embed_curve'][i] #in shape [100, 256]
      patch_predictions_topo_embed_curve = patch_predictions['patch_topo_embed_curve'][i] #in shape [100, 256]
      curve_predictions_topo_embed_corner = curve_predictions['curve_topo_embed_corner'][i] #in shape [100, 256]
      curve_predictions_topo_embed_patch = curve_predictions['curve_topo_embed_patch'][i] #in shape [100, 256]
     
      #select matched curve and corners
      cur_corner_indices = corner_indices[i] #a tuple
      cur_curve_indices = curve_indices[i] #a tuple
      cur_patch_indices = patch_indices[i] #a tuple

      valid_patch_predictions_topo_embed_corner = patch_predictions_topo_embed_corner[cur_patch_indices[0]]
      valid_patch_predictions_topo_embed_curve = patch_predictions_topo_embed_curve[cur_patch_indices[0]]
      
      valid_corner_predictions_topo_embed_patch = corner_predictions_topo_embed_patch[cur_corner_indices[0]]
      valid_corner_predictions_topo_embed_curve = corner_predictions_topo_embed_curve[cur_corner_indices[0]]

      valid_curve_predictions_topo_embed_corner = curve_predictions_topo_embed_corner[cur_curve_indices[0]]
      valid_curve_predictions_topo_embed_patch = curve_predictions_topo_embed_patch[cur_curve_indices[0]]


      patch_corner_similarity = torch.sigmoid(torch.mm(valid_patch_predictions_topo_embed_corner, valid_corner_predictions_topo_embed_patch.transpose(0,1))) 
      curve_corner_similarity = torch.sigmoid(torch.mm(valid_curve_predictions_topo_embed_corner, valid_corner_predictions_topo_embed_curve.transpose(0,1)))
      patch_curve_similarity = torch.sigmoid(torch.mm(valid_patch_predictions_topo_embed_curve, valid_curve_predictions_topo_embed_patch.transpose(0,1)))
      
      
      cur_PK_mat=PK_mat[i]
      cur_PK_mat= cur_PK_mat[cur_patch_indices[1]][:, cur_corner_indices[1]]

      # PK topo loss/acc
      topo_correspondence_loss.append(F.binary_cross_entropy(patch_corner_similarity.view(-1), cur_PK_mat.view(-1)))
      topo_correspondence_acc.append(100.0 * (1.0 - (torch.round(patch_corner_similarity)-cur_PK_mat).abs().mean()))
      
      # 每个curve有两个端点
      curve_point_loss.append((torch.sum(curve_corner_similarity, dim=1) - 2).norm().mean()/math.sqrt(curve_corner_similarity.shape[1]))

      # PC*CK=2PK
      pc_cc_m = torch.mm(patch_curve_similarity, curve_corner_similarity)
      assert(pc_cc_m.shape == patch_corner_similarity.shape)
      patch_close_loss.append((pc_cc_m - 2 * patch_corner_similarity).norm() / math.sqrt(patch_corner_similarity.shape[0] * patch_corner_similarity.shape[1]))
      
      curve_patch_loss.append((torch.sum(patch_curve_similarity, dim=0) - 2).norm().mean()/math.sqrt(patch_curve_similarity.shape[0]))
      # panel_topo_loss.append(degree2_loss_soft(patch_corner_similarity, patch_curve_similarity, curve_corner_similarity))
    
    return [sum(topo_correspondence_loss) / len(topo_correspondence_loss),
            sum(curve_point_loss) / len(curve_point_loss),
            sum(patch_close_loss) / len(patch_close_loss),
            sum(curve_patch_loss) / len(curve_patch_loss),
            # torch.tensor(0.0,device=patch_close_loss[0].device,dtype=patch_close_loss[0].dtype),
            sum(topo_correspondence_acc) / len(topo_correspondence_acc),]