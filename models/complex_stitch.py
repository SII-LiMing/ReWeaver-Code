import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from config import ComplexStitchConfig
from models.core_module import MLP,MLP_hn
from models.pc_encoder import PctComplexStitch
from models.transformer_multipath import build_transformer_tripath,build_transformer_bipath

class CornerPredictor(nn.Module):
    def __init__(self,args:ComplexStitchConfig=None):
        hidden_dim=768 
        super().__init__()
        self.empty_prediction_embed = nn.Linear(hidden_dim, 2)#num_classes + 1, empty or non-empty
        self.corner_position_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        
        self.corner_topo_embed_curve = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 1) #topo embed dim:256
        self.corner_topo_embed_patch = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 1)
        
    def forward(self, hs):                 
        outputs_corner_coord = self.corner_position_embed(hs).tanh() # [-1,1] 
        outputs_class = self.empty_prediction_embed(hs) #to be consistent with curve, we treat 0 as non-empty and 1 as empty

        output_corner_topo_embedding_curve = self.corner_topo_embed_curve(hs)
        output_corner_topo_embedding_patch = self.corner_topo_embed_patch(hs)
        out = {'pred_corner_logits': outputs_class, 'pred_corner_position': outputs_corner_coord, 'corner_topo_embed_curve': output_corner_topo_embedding_curve,'corner_topo_embed_patch': output_corner_topo_embedding_patch} 
        return out


class CurvePredictor(nn.Module):
    """ This is the DETR module that performs geometric primitive detection - Curves """
    def __init__(self,args:ComplexStitchConfig=None): #num_classes is not used for corner detection #backbone
        
        hidden_dim=768
        hn_mlp_dim=64
        self.points_per_curve = args.points_per_curve
        
        super().__init__()
        self.valid_curve_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.curve_start_point_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        
        self.curve_shape_embed = MLP_hn(1, hn_mlp_dim, 3, 3, hidden_dim)
        curve_pe_x_tensor = (torch.arange(self.points_per_curve, dtype=torch.float32) / (self.points_per_curve - 1)).view(self.points_per_curve,1)
        self.register_buffer("curve_pe", curve_pe_x_tensor)
        
        self.curve_topo_embed_patch = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 1)
        
         
    def forward(self, hs):
        
        outputs_start_point_coord = self.curve_start_point_embed(hs).tanh() # [-1,1]
        sampled_points_feature = self.curve_pe.view(1,1,self.points_per_curve, 1).repeat(hs.shape[0], hs.shape[1], 1,1)
        

        sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,self.points_per_curve,1) + self.curve_shape_embed(sampled_points_feature, hs)

        is_curve_valid_pred = self.valid_curve_embed(hs)
        

        output_curve_topo_embedding_patch = self.curve_topo_embed_patch(hs)
        
        out = {'pred_curve_logits': is_curve_valid_pred, 'pred_curve_points': sampled_points, 'curve_topo_embed_patch': output_curve_topo_embedding_patch}

        return out


class PatchPredictor(nn.Module):
    def __init__(self,args:ComplexStitchConfig): #num_classes is not used for corner detection #backbone
        super().__init__()
        
        self.args = args
        self.num_queries = args.n_patch_queries
        hidden_dim=768
        patch_embedding_mlp_layers = 3
        hn_mlp_dim=64
        
        self.points_per_patch_dim=args.points_per_patch_dim
        points_per_patch_dim=self.points_per_patch_dim
        
        self.valid_patch_embed = MLP(hidden_dim, hidden_dim, 2, patch_embedding_mlp_layers)
        self.patch_center_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        output_dim = 3

        self.patch_shape_embed = MLP_hn(2, hn_mlp_dim, output_dim, 3, hidden_dim)
        patch_pe_x_tensor = (torch.arange(points_per_patch_dim, dtype=torch.float32) / (points_per_patch_dim - 1)).view(points_per_patch_dim,1)
        patch_pe_y_tensor = (torch.arange(points_per_patch_dim, dtype=torch.float32) / (points_per_patch_dim - 1)).view(points_per_patch_dim,1)
        self.register_buffer("patch_pe_x", patch_pe_x_tensor)
        self.register_buffer("patch_pe_y", patch_pe_y_tensor)
        self.patch_topo_embed_curve = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 1)

          
    def forward(self, hs):
        
        batch_size = hs.shape[0]
        patch_pe_x=self.patch_pe_x
        patch_pe_y=self.patch_pe_y
        points_per_patch_dim=self.points_per_patch_dim
        
        outputs_start_point_coord = self.patch_center_embed(hs).tanh()
        
        sampled_points_feature = [torch.cat([patch_pe_x[i], patch_pe_y[j]], dim = -1) for i in range(points_per_patch_dim) for j in range(points_per_patch_dim)]
        sampled_points_feature = torch.cat(sampled_points_feature).view(1,1,points_per_patch_dim*points_per_patch_dim, 2).repeat( batch_size, self.args.n_patch_queries, 1,1)
        
        sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,points_per_patch_dim*points_per_patch_dim,1) + self.patch_shape_embed(sampled_points_feature, hs)[...,:3]

        is_patch_valid_pred = self.valid_patch_embed(hs)

        output_patch_topo_embedding_curve = self.patch_topo_embed_curve(hs)
        
        
        out = {'pred_patch_logits': is_patch_valid_pred,  
               'pred_patch_points': sampled_points, 
               'patch_topo_embed_curve': output_patch_topo_embedding_curve}   

        return out
    
    @torch.no_grad()
    def forward_scaled_points(self, hs):
        device, dtype = hs.device, hs.dtype
        base_dim = self.points_per_patch_dim
        batch_size, query_num = hs.shape[0], hs.shape[1]
        
        base_pe_x = (torch.arange(base_dim, dtype=torch.float32) / (base_dim - 1)).view(base_dim,1).to(device)
        base_pe_y = (torch.arange(base_dim, dtype=torch.float32) / (base_dim - 1)).view(base_dim,1).to(device)
        base_grid_hw2 = [torch.cat([base_pe_x[i], base_pe_y[j]], dim = -1) for i in range(base_dim) for j in range(base_dim)]
        base_grid_hw2 = torch.cat(base_grid_hw2).view(1,1,base_dim*base_dim, 2).repeat( batch_size, self.args.n_patch_queries, 1,1)
        
        outputs_start_point_coord = self.patch_center_embed(hs).tanh()
        sampled_points_ori_res = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,base_dim*base_dim,1) + self.patch_shape_embed(base_grid_hw2, hs)[...,:3]
        
        K=60

        results: list[list[torch.Tensor]] = []

        for b in range(batch_size):
            row_list: list[torch.Tensor] = []
            for q in range(query_num):
                start_point_coord = outputs_start_point_coord[b][q]
                
                pts_preview = sampled_points_ori_res[b][q].reshape(base_dim,base_dim,3)

                # 2) 相邻边平均长度（u: 水平/W 方向, v: 垂直/H 方向）
                du = pts_preview[:, 1:, :] - pts_preview[:, :-1, :]  # [H, W-1, 3]
                dv = pts_preview[1:, :, :] - pts_preview[:-1, :, :]  # [H-1, W, 3]
                Lu = du.norm(dim=-1).mean().item()
                Lv = dv.norm(dim=-1).mean().item()

                dim_u = int(round(K * Lu * base_dim))
                dim_v = int(round(K * Lv * base_dim))

                # 3) 用自适应分辨率重采样
                pe_x = (torch.arange(dim_u, dtype=torch.float32) / (dim_u - 1)).view(dim_u,1).to(device)
                pe_y = (torch.arange(dim_v, dtype=torch.float32) / (dim_v - 1)).view(dim_v,1).to(device)
                grid_hw2 = [torch.cat([pe_x[i], pe_y[j]], dim = -1) for i in range(dim_u) for j in range(dim_v)]
                grid_hw2 = torch.cat(grid_hw2).view(1,1,dim_u*dim_v, 2)

                
                
                c_hs=hs[b][q]
                pts_adapt = start_point_coord.reshape(1,1,1,-1).repeat(1,1,dim_u*dim_v,1) + self.patch_shape_embed(grid_hw2, c_hs.reshape(1,1,-1))[...,:3]
                pts_adapt = pts_adapt.squeeze(0).squeeze(0)
                
                row_list.append(pts_adapt)

            results.append(row_list)

        return {'pred_patch_points_scaled': results}


class ComplexStitchModel(nn.Module):
    def __init__(self,args:ComplexStitchConfig):
        super().__init__()
        
        self.args = args
        
        # self.pc_enoder = PctComplexStitch(args)
        self.bi_transformer=build_transformer_bipath(args)
        hidden_dim = self.bi_transformer.d_model
        
        self.curve_model=CurvePredictor(args)
        self.patch_model = PatchPredictor(args)
        
        
        self.primitive_type_embed = nn.Embedding(2, hidden_dim)
        self.query_embed_curve = nn.Embedding(args.n_curve_queries, hidden_dim)
        self.query_embed_patch = nn.Embedding(args.n_patch_queries, hidden_dim)
    
    def forward(self, img_tokens):
        query_lst = [self.query_embed_curve.weight, self.query_embed_patch.weight]
        
        curve_features,patch_features=self.bi_transformer(img_tokens, query_lst, self.primitive_type_embed.weight)
        
        # pred_curve_logits: B,200,2
        # pred_curve_points: B,200,34,3
        curve_predictions = self.curve_model(curve_features)
        
        
        patch_predictions = self.patch_model(patch_features)
        
        return curve_predictions,patch_predictions,curve_features,patch_features
    
    def get_scaled_points(self, patch_features):

        return self.patch_model.forward_scaled_points(patch_features)