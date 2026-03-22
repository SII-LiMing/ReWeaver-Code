import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points
from config import Args,ComplexStitchConfig
import math
from utils.utils import knn_point
 


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



def sample_and_group(npoint, nsample, xyz, points):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint 
    xyz = xyz.contiguous()


    fps_idx=sample_farthest_points(points=xyz,K=npoint)[1]
    
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    idx = knn_point(nsample, xyz, new_xyz)

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)       # b,n,d,s
        x = x.reshape(-1, d, s)         # bn,d,s
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)        #b,d,n
        return x

class PctComplexStitch(nn.Module):
    def __init__(self, args:ComplexStitchConfig):
        super(PctComplexStitch, self).__init__()
        
        self.args = args
        
        self.n_pct_feature=self.args.n_pct_feature
        self.n_pct_sample=self.args.n_pct_sample
        self.d_model=args.d_model
        assert self.d_model<=1280
        
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.gather_local_0 = Local_op(in_channels=512, out_channels=256)
        self.gather_local_1 = Local_op(in_channels=512, out_channels=256)

        self.pt_last = Point_Transformer_Last(256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, self.d_model, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(self.d_model),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.conv3 = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.d_model)
        self.conv4 = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)

    def forward(self, xyz):
        """
            xyz: B,N,3
        """
        
        x = xyz.permute(0, 2, 1)                # B,3,N
        batch_size, _, _ = x.size()             
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)                  # B,N,D
        
        # new_feature: B,2*n_feature,n_sample,D*2
        new_xyz, new_feature = sample_and_group(npoint=self.n_pct_feature*2, nsample=self.n_pct_sample, xyz=xyz, points=x) 
        feature_0 = self.gather_local_0(new_feature)    # B,D,n_feature
        feature = feature_0.permute(0, 2, 1)            # B,n_feature,D
        
        new_xyz, new_feature = sample_and_group(npoint=self.n_pct_feature, nsample=self.n_pct_sample, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)    # B,D,N''
    
        # pt_last: 4*D->D
        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)                      
        x=F.relu(self.bn3(self.conv3(x)))   
        x=self.conv4(x)                         # B,D,N'' 
        x = x.permute(0,2,1)                    # B,N'',D=B,1024,384
        return x


class PctFlatten(nn.Module):
    def __init__(self, args:Args):
        super(PctFlatten, self).__init__()
        
        self.args = args
        d_model=args.d_model
        assert d_model<=1280
        
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False)

        self.gather_local = Local_op(in_channels=512, out_channels=256)

        self.pt_last = Point_Transformer_Last(256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, d_model, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(d_model),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(d_model, d_model, kernel_size=1)
        
        # self.input_transform = TransformNet(k=3)

    def forward(self, xyz):
        """
            xyz: B,N,3
        """
        # n_pc_feature=self.args.n_pc_feature
        n_pc_feature=xyz.shape[0]//32+1
        N=xyz.shape[1]
        n_pc_feature=N//50+1
        n_pc_sample=50
        x = xyz.permute(0, 2, 1)                # B,3,N
        
        # transform = self.input_transform(x)          # B,3,3
        # x = torch.bmm(transform, x)                  # B,3,N
        # xyz = x.permute(0, 2, 1).contiguous()        # B,N,3  --> 重新更新 xyz


        batch_size, _, _ = x.size()             
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)                  # B,N,D
        
        # new_feature: B,2*n_feature,n_sample,D*2
        new_xyz, new_feature = sample_and_group(npoint=n_pc_feature, nsample=n_pc_sample, xyz=xyz, points=x) 
        feature_1 = self.gather_local(new_feature)
        # pt_last: 4*D->D
        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)                      
        x=F.relu(self.conv3(x))   
        x=self.conv4(x)                         # B,D,N'' 
        x = x.permute(0,2,1)                    # B,N'',D=B,1024,384
        return x
    
class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

if __name__=="__main__":
    import torch
    print(torch.cuda.is_available())