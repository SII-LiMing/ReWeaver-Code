import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from config import ComplexStitchConfig

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == 'lrelu':
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayerMultipath(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="lrelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_pc = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #only impl for no depoule version
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) #element


        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, tgt,query_pos: Optional[Tensor] = None):
        
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos) # query pose
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        # I don't know why the original code use this 
        # tgt2 = self.norm2(tgt)
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
        #                            key=memory,
        #                            value=memory)[0]
        # tgt = tgt + self.dropout2(tgt2)
                
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
    
    def forward_pre_stage1(self, tgt, query_pos):
        assert query_pos is not None
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos) # query pose
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        return tgt2

    def forward_pre_stage2(self, tgt2, 
                    query_pos: Optional[Tensor] = None,
                    key_cross: Optional[Tensor] = None,
                    val_cross: Optional[Tensor] = None,
                    ):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                key=key_cross,value=val_cross)[0]
        return tgt2

    def forward_pre_stage3(self, tgt2,
                           query_pos: Optional[Tensor] = None,
                           key_cross: Optional[Tensor] = None,
                           val_cross: Optional[Tensor] = None):
        
        tgt2 = self.multihead_attn_pc(query=self.with_pos_embed(tgt2, query_pos),
                                key=key_cross,
                                value=val_cross)[0]
        return tgt2


    def forward(self, tgt, val_cross: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                stage = -1,
                key_cross: Optional[Tensor] = None):
        
        if stage == 1:
            return self.forward_pre_stage1(tgt, query_pos)
        elif stage == 2:
            return self.forward_pre_stage2(tgt,query_pos,key_cross, val_cross)
        elif stage == 3:
            return self.forward_pre_stage3(tgt,query_pos,val_cross, val_cross)
        elif stage == 4:
            tgt = self.norm3(tgt)
            tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            return tgt
            
        
        self.forward_pre(tgt, query_pos)


class TransformerDecoderMultipath(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, n_path = 3):
        super().__init__()
        layers = _get_clones(decoder_layer, num_layers) #will be released after the copy
        self.layers_list = _get_clones(layers, n_path)

        self.num_layers = num_layers
        self.norm_list = _get_clones(norm, n_path)
        self.n_path = n_path

    def forward(self, tgt_list, memory,query_pos_list,primitive_type_embed):
        
        # query_pos_list: [n_path, [B, L, D]]      L means sequence length
        # tgt_list: [n_path, [B, L, D]]
        # memory: [B, L, D]
        
        # for nn.MultiheadAttention
        tgt_list=[it.transpose(0,1) for it in tgt_list] # [n_path, [L, B, D]]
        query_pos_list=[it.transpose(0,1) for it in query_pos_list] # [n_path, [L, B, D]]
        memory=memory.transpose(0,1) # [L, B, D]
        
        assert(len(query_pos_list) == self.n_path)
        assert(len(tgt_list) == self.n_path)

        output_list = tgt_list
        intermediate_list = []
        for i in range(self.n_path):
            intermediate_list.append([])         

        for j in range(self.num_layers):
            #stage1: self attention of each path
            for i in range(self.n_path):
                output_selfatt_res=self.layers_list[i][j](output_list[i], query_pos=query_pos_list[i], stage = 1)
                
                output_list[i] = output_list[i] + output_selfatt_res
            
            # LayerNorm
            output_normalize = []
            for i in range(self.n_path):
                output_normalize.append(self.layers_list[i][j].norm2(output_list[i]))
                
            #stage2: cross attention between different paths
            val_cross = [] #only for 3 types, without pritimive embedding
            key_cross = []
            for iter1 in range(self.n_path):
                idlist = list(range(self.n_path))
                idlist.remove(iter1)
                val_list = []
                key_list = []
                assert(len(idlist) == self.n_path - 1)
                for id in idlist:
                    val_list.append(output_normalize[id])
                    key_list.append(output_normalize[id] + primitive_type_embed[id]+query_pos_list[id])
                val_cross.append(torch.cat(val_list, dim=0))
                key_cross.append(torch.cat(key_list, dim=0))
        
            output_stage2_res = []
            for i in range(self.n_path):
                output_stage2_res.append(self.layers_list[i][j](output_normalize[i], val_cross=val_cross[i],
                                                                query_pos=query_pos_list[i], stage = 2, key_cross = key_cross[i]))
                
                output_list[i] = output_stage2_res[i] + output_list[i]


            #stage 3 cross attention between each path and pc features:
            output_stage2_res_pc = []
            for i in range(self.n_path):
                output_stage2_res_pc.append(self.layers_list[i][j](output_normalize[i], val_cross=memory,query_pos=query_pos_list[i], stage = 3))
                output_list[i] = output_stage2_res_pc[i] + output_list[i]
                
            #stage 4: finnal feedforward
            output_stage3_res = []
            for i in range(self.n_path):
                output_stage3_res.append(self.layers_list[i][j](output_list[i], stage = 4))
                output_list[i] = output_stage3_res[i] + output_list[i]     

        
        # LayerNorm
        for i in range(self.n_path):
            if self.norm_list[0] is not None:
                output_list[i] = self.norm_list[i](output_list[i])

        

        return output_list          # [B,L,D]


class TransformerMultipath(nn.Module):

    def __init__(self, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="lrelu",   n_path=3):
        super().__init__()
        
        
        decoder_layer = TransformerDecoderLayerMultipath(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderMultipath(decoder_layer, num_decoder_layers, decoder_norm, n_path = n_path)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.n_path = n_path

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed_list, primitive_type_embed):
        
        B, N, C = src.shape
        query_embed_list_new = []
        for i in range(self.n_path):
            query_embed_list_new.append(query_embed_list[i].unsqueeze(0).repeat(B, 1, 1))

        tgt_list = []
        for i in range(self.n_path):
            tgt_list.append(torch.zeros_like(query_embed_list_new[i]))

        # src: B, N, D
        # query_embed_list_new: [n_path [B, L, D]]
        # tgt_list: [n_path [B, L, D]]
        
        hs_list = self.decoder(tgt_list, src, query_pos_list = query_embed_list_new,primitive_type_embed = primitive_type_embed)

        for i in range(self.n_path):
            hs_list[i] = hs_list[i].transpose(0,1)  # [n_path,[L, B, D]] -> [n_path,[B, L, D]]

        return hs_list
    
    
def build_transformer_tripath(args:ComplexStitchConfig=None):
    return TransformerMultipath(
        d_model=args.d_model,
        dropout=0,
        nhead=8,
        dim_feedforward=2048,
        num_decoder_layers=6,
        n_path = 3,
    )

def build_transformer_bipath(args:ComplexStitchConfig=None):
    return TransformerMultipath(
        d_model=args.d_model,
        dropout=0,
        nhead=8,
        dim_feedforward=2048,
        num_decoder_layers=12,
        n_path = 2,
    )
    
