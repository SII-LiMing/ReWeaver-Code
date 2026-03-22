import json
import os
import random
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from utils import rotation as rotation_tools
from utils.utils import get_pattern_json_with_3d_vertices,pc_normalize
from config import parse_args,DatasetConfig,Statistics
from collections import Counter
from PIL import Image
import torch
from skimage.transform import resize
from utils.img_utils import center_human

def get_final_trans(mean_pc, scale_pc, trans_2_to_3, rot_2_to_3, mean_2d, scale_2d):

    A = np.eye(4)
    A[:3, :3] = (1.0 / scale_pc) * np.eye(3)
    A[:3, 3]  = - mean_pc / scale_pc

    Binv = np.eye(4)
    Binv[:3, :3] = scale_2d * np.eye(3)
    Binv[:3, 3]  = np.array([mean_2d[0], mean_2d[1], 0.0])

    R = rotation_tools.euler_xyz_to_R(rot_2_to_3)

    C = np.eye(4)
    C[:3, :3] = R
    C[:3, 3]  = trans_2_to_3

    to_world_scale = np.eye(4)
    to_world_scale[:3, :3] = scale_pc * np.eye(3)
    T = to_world_scale @ A @ C @ Binv
    
    M = T[:3, :3]
    col_norms = np.linalg.norm(M, axis=0)
    final_scale = col_norms.mean()
    R = M / final_scale
    final_rot= rotation_tools.R_to_euler(R,return_rad=True)
    final_trans = T[:3, 3]
    
    return final_scale,final_trans,final_rot

def is_single_cycle_incidence(M: np.ndarray) -> bool:
    M = M.astype(bool)
    E, V = M.shape

    # 1) 每条边连接两个点
    if not np.all(M.sum(axis=1) == 2):
        return False

    # 2) 参与点度数=2（未参与的点度数=0）
    deg = M.sum(axis=0)
    active = deg > 0
    if not np.all(deg[active] == 2):
        return False

    # 3) 边数 = 参与点数
    E_active = np.count_nonzero(M.sum(axis=1) > 0)  # 这里其实就是 E
    V_active = np.count_nonzero(active)
    if E_active != V_active:
        return False

    # 4) 连通性（在参与点的子图上做）
    # 顶点邻接: A = (M^T M) 的对角清零后看是否 > 0
    A = (M.T @ M).astype(int)
    np.fill_diagonal(A, 0)
    A = (A > 0)
    A = A[np.ix_(active, active)]

    # BFS 连通
    seen = np.zeros(V_active, dtype=bool)
    stack = [0]; seen[0] = True
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(A[u]):
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return seen.all()

class TestDataSet_GCD(Dataset):
        def __init__(self, args: DatasetConfig, statistics: Statistics,device="cpu"):
            self.root = Path(args.root)
            self.type = args.data_type
            self.texture_type = args.texture_type
            self.device = device
            self.samples = []
            self.img_mean = np.array(statistics.img_mean).reshape(1,3,1,1)
            self.img_std = np.array(statistics.img_std).reshape(1,3,1,1)
            sample_file = Path(args.samples)
            
            self.edge_type_to_idx={
                "line": 0,
                "circle": 1,
                "quadratic": 2,
                "cubic": 3,
                "none": 4
            }
            
            if sample_file.exists() and sample_file.is_file():
                # 如果提供了sample_file，优先从文件读取
                with open(sample_file, "r") as f:
                    lines = f.readlines()
                samples = [line.strip() for line in lines if line.strip()]
                samples = [self.root / Path(sample) for sample in samples]
                self.samples = samples
            else:
                # 否则从 root 扫描
                self.samples = list(self.root.iterdir())
            # if self.type=="test":
                # self.samples=[it for it in self.samples if it.name.startswith("rand_00") or it.name.startswith("rand_10") or it.name.startswith("rand_AA") or it.name.startswith("rand_ZZ")]
                # self.samples=[it for it in self.samples if it.name.startswith("rand_NF6XNTMEX5")]
                # self.samples=[it for it in self.samples if it.name.startswith("rand_0BRE5SKBPT")]
            # self.samples=self.samples[:1000]
            
        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample_dir = self.samples[idx]
            name=sample_dir.name
            
            # load image
            imgs = []
            
            if self.texture_type == "ori":
                img_dir = sample_dir / "render_output_ori_texture" / "rgb"
            elif self.texture_type == "tileable":
                img_dir = sample_dir / "render_output" / "rgb"
            else:
                raise("texture type not found!")
        
            img_files = sorted(img_dir.iterdir())  # 保证视角顺序一致
            # img_files = img_files[1:2]
            for img_path in img_files:
                img_arr = Image.open(img_path).convert("RGB")
                img_arr = np.array(img_arr,dtype=np.float32)  # H x W x C

                img_arr = np.transpose(img_arr, (2, 0, 1))  # C x H x W
                img_arr = (img_arr/255.).clip(0., 1.)
                imgs.append(img_arr)

            imgs = np.stack(imgs, axis=0)  # V x C x H x W
            imgs = (imgs - self.img_mean) / self.img_std

        

            return {
                    # str
                    "name": name,
                    
                    # View x Channel x W x H
                    "images": imgs,               
                }
        



class TestDataSet_4D_Dress(Dataset):
        def __init__(self, args: DatasetConfig, statistics: Statistics,device="cpu"):
            self.root = Path(args.root)
            self.type = args.data_type
            self.texture_type = args.texture_type
            self.device = device
            self.samples = []
            self.img_mean = np.array(statistics.img_mean).reshape(1,3,1,1)
            self.img_std = np.array(statistics.img_std).reshape(1,3,1,1)
            sample_file = Path(args.samples)
            self.edge_type_to_idx={
                "line": 0,
                "circle": 1,
                "quadratic": 2,
                "cubic": 3,
                "none": 4
            }
            
            if sample_file.exists() and sample_file.is_file():
                # 如果提供了sample_file，优先从文件读取
                with open(sample_file, "r") as f:
                    lines = f.readlines()
                samples = [line.strip() for line in lines if line.strip()]
                samples = [self.root / Path(sample) for sample in samples]
                self.samples = samples
            else:
                # 否则从 root 扫描
                self.samples = list(self.root.iterdir())
            # if self.type=="test":
                # self.samples=[it for it in self.samples if it.name.startswith("rand_00") or it.name.startswith("rand_10") or it.name.startswith("rand_AA") or it.name.startswith("rand_ZZ")]
                # self.samples=[it for it in self.samples if it.name.startswith("rand_NF6XNTMEX5")]
                # self.samples=[it for it in self.samples if it.name.startswith("rand_0BRE5SKBPT")]
            # self.samples=self.samples[:1000]
            
        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample_dir = self.samples[idx]
            name=sample_dir.name
            
            imgs=[]
            masks=[]
            img_path_lst=list((sample_dir/"view").iterdir())
            img_path_lst=sorted(img_path_lst)
            if (sample_dir/"mask").exists():
                mask_path_lst=list((sample_dir/"mask").iterdir())
                mask_path_lst=sorted(mask_path_lst)
                for msk_path in mask_path_lst:
                    mask = Image.open(msk_path)
                    mask_arr = np.array(mask,dtype=np.float32)  # H x W x C
                    masks.append(mask_arr)

            
            for i,img_path in enumerate(img_path_lst):
                img = Image.open(img_path).convert("RGB")
                # arr = np.array(img,dtype=np.float32)  # H x W x C
                img_arr = np.array(img)  # H x W x C
                
                if len(mask_path_lst)==len(img_path_lst):
                    img_arr[masks[i]==0]=255
                img_to_save=Image.fromarray(img_arr)
                img_to_save.save(f"test_{i}.png")
                
                img_arr = np.array(img_arr,dtype=np.float32)
                img_arr = np.transpose(img_arr, (2, 0, 1))  # C x H x W
                img_arr = (img_arr/255.).clip(0., 1.)
                
                c, h, w = img_arr.shape
                start_h = (h - w) // 2
                img_arr = img_arr[:, start_h:start_h+w, :]
                img_arr = resize(img_arr, (3, 518, 518), anti_aliasing=True).astype(np.float32)
                
                img_arr = center_human(img_arr)
                vis_arr = (img_arr.clip(0, 1) * 255).astype(np.uint8)

                vis_arr = np.transpose(vis_arr, (1, 2, 0))

                # 3. 保存
                img_to_check = Image.fromarray(vis_arr)
                img_to_check.save(f"test_{i}_resized.png")
                imgs.append(img_arr)
            
            imgs = np.stack(imgs, axis=0)  # V x C x H x W
            imgs = (imgs - self.img_mean) / self.img_std
            return {
                    # str
                    "name": name,
                    
                    # View x Channel x W x H
                    "images": imgs,               
                }
            
class GCD_DataSet(Dataset):
    def __init__(self, args: DatasetConfig, statistics: Statistics,device="cpu"):
        self.root = Path(args.root)
        self.type = args.data_type
        self.texture_type = args.texture_type
        self.device = device
        self.samples = []
        self.img_mean = np.array(statistics.img_mean).reshape(1,3,1,1)
        self.img_std = np.array(statistics.img_std).reshape(1,3,1,1)
        sample_file = Path(args.samples)
        
        self.edge_type_to_idx={
            "line": 0,
            "circle": 1,
            "quadratic": 2,
            "cubic": 3,
            "none": 4
        }
        
        if sample_file.exists() and sample_file.is_file():
            # 如果提供了sample_file，优先从文件读取
            with open(sample_file, "r") as f:
                lines = f.readlines()
            samples = [line.strip() for line in lines if line.strip()]
            samples = [self.root / Path(sample) for sample in samples]
            self.samples = samples
        else:
            # 否则从 root 扫描
            self.samples = list(self.root.iterdir())
        
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        name=sample_dir.name
        
        # load image
        imgs = []
        
        
        if self.texture_type == "ori":
            img_dir = sample_dir / "render_output_ori_texture" / "rgb"
        elif self.texture_type == "tileable":
            img_dir = sample_dir / "render_output" / "rgb"
        else:
            raise("texture type not found!")
    
        img_files = sorted(img_dir.iterdir())  # 保证视角顺序一致
        for img_path in img_files:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img,dtype=np.float32)  # H x W x C

            arr = np.transpose(arr, (2, 0, 1))  # C x H x W
            arr = (arr/255.).clip(0., 1.)
            imgs.append(arr)

        imgs = np.stack(imgs, axis=0)  # V x C x H x W
        imgs = (imgs - self.img_mean) / self.img_std

        panels_dic=json.load(open(sample_dir/f"{name}_2d_panel.json","r"))
        panel_order=panels_dic["panel_order"]
        panels=panels_dic["panels"]        

        panels = {
            int(k): item 
            for k, item in panels.items()
        }
        
        panels_mean=[]
        panels_scale=[]
        panels_edge_pts=[]
        for i in sorted(panels.keys()):
            panels_mean.append(panels[i]['mean'])
            panels_scale.append(panels[i]['scale'])
            panels_edge_pts.append(np.array(panels[i]['edge_points'],dtype=np.float32))
            
        panel_mean=np.array(panels_mean)
        panel_scale=np.array(panels_scale)
        
        
        geo_3d_npz=np.load(sample_dir/f"{name}_3d_geo.npz")
        pc_sampled=geo_3d_npz["pc_sampled"]
        pc_labels=geo_3d_npz["pc_labels"]
        PC_mat=geo_3d_npz["PC_mat"]
        pc_scale=geo_3d_npz["pc_scale"]    
        pc_mean=geo_3d_npz["pc_mean"]
        curves_sampled=geo_3d_npz["curves_sampled"]
        
        # build patch points
        pc_labels_unique = np.unique(pc_labels)
        pc_labels_unique = pc_labels_unique[pc_labels_unique != -1]  # remove stitch label
        patch_points=[]
        for i in pc_labels_unique:
            patch_points.append(pc_sampled[pc_labels == i])

        return {
                # str
                "name": name,
                
                # View x Channel x W x H
                "images": imgs,               
                
                # List[ point_num x 3 ]
                "patch_points": patch_points,   
                
                # curve_num x point_per_curve x 3
                "curves_sampled": curves_sampled,
                
                # patch_num x curve_num
                "PC_mat": PC_mat,
                
                # List[ edge_num x point_per_edge x 2 ]
                "edge_points": panels_edge_pts,      
                
                "panel_scale": panel_scale,
                "panel_mean": panel_mean,
                "pc_mean": pc_mean,
                "pc_scale": pc_scale
            }

          
if __name__ == '__main__':
    args=parse_args("configs/config_img_input.yaml")
    train_dataset=GCD_DataSet(args.train_data,device="cuda:1")
    from tqdm import tqdm
    
    count=0
    for it in tqdm(train_dataset):
        count+=1
    
    print(count)
    import ipdb;ipdb.set_trace()
    # num_lst=[]
    # name_lst=[]
    # for it in test_data:
    #     # import ipdb
    #     # ipdb.set_trace()
    #     num=it["labels"].sum()
    #     name=it["name"]
    #     num_lst.append(num.item())
    #     name_lst.append(name)
        
    # # sort name_lst by num_lst
    # num_lst = np.array(num_lst)
    # name_lst = np.array(name_lst)
    # name_lst = name_lst[np.argsort(num_lst)]
    # # write name_lst to file: meta_data_train.txt
    # with open("meta_data_test.txt","w") as f:
    #     for name in name_lst:
    #         f.write(name+"\n")
    # print("Done")
    
    
    # root = ""  # 填你的根目录
    # sample_file = ""  # 填你的保存路径
    # samples = []

    # # 列出root下所有条目
    # it_lst = os.listdir(root)
    # for it in it_lst:
    #     now_dir = os.path.join(root, it)
    #     samples.append(now_dir)

    # print(f"Total samples found: {len(samples)}")

    # # 确保样本数足够
    # num_samples = min(5000, len(samples))

    # # 随机抽取
    # selected_samples = random.sample(samples, num_samples)

    # # 写入sample_file
    # with open(sample_file, "w") as f:
    #     for path in selected_samples:
    #         f.write(path + "\n")

    # print(f"Saved {num_samples} samples to {sample_file}")