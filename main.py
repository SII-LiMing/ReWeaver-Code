import os
import shutil
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

# ddp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models.complex_stitch import ComplexStitchModel
from models.flatten import FlattenModel
from models.matcher_curve import build_matcher_curve
from models.matcher_patch import build_matcher_patch
from models.criterion import SetCriterion_Curve,SetCriterion_Patch,Patch_Curve_Matching
from models.flatten_loss import FlattenLoss

from vggtencoder.aggregator import Aggregator

from utils.utils import setup_seed
from config import parse_args,Args
from data import GCD_DataSet,TestDataSet_GCD,TestDataSet_4D_Dress
from loss_manager import LossManager
from logger import WandbLogger,IOStreamLogger,TensorBoardLogger



import math
from torch import optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR



def pause_one_rank(device_sync=True, tag=""):
    """
    只让 rank0 进入交互式断点，其他 rank 在 barrier 处挂起等待。
    所有进程必须在同一代码行按相同顺序调用它（否则会死锁）。
    """
    if device_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    rank = dist.get_rank()

    if rank == 0:
        print(f"[rank 0] >>> pause point {tag} (others waiting at barrier). Entering PDB...")
        import pdb; pdb.set_trace()  # 在此处调试
        dist.barrier()               # 调试完继续，放行其他进程
    else:
        dist.barrier()               # 等 rank0 放行

def custom_collate_fn(batch):
    """
    保留 batch 为 list 格式，不进行 torch.stack
    适用于每个样本 shape 不一样的情况。
    """
    return batch  # 或者 return tuple(zip(*batch)) 按需解构

def init_saved_dir(args:Args):
    (args.save_dir / "weights").mkdir(parents=True, exist_ok=True)
    (args.save_dir / "pred").mkdir(parents=True, exist_ok=True)
    (args.save_dir / "backup").mkdir(parents=True, exist_ok=True)
    (args.save_dir/"weights"/"complex_stitch").mkdir(parents=True, exist_ok=True)
    (args.save_dir/"weights"/"flatten").mkdir(parents=True, exist_ok=True)
    (args.save_dir/"weights"/"img_encoder").mkdir(parents=True, exist_ok=True)
    
    for e in range(args.save_pred_freq,args.epochs+1,args.save_pred_freq):
        (args.save_dir / "pred" / "train" / str(e)).mkdir(parents=True, exist_ok=True)
        (args.save_dir / "pred" / "val" / str(e)).mkdir(parents=True, exist_ok=True)
    (args.save_dir / "pred" / "test").mkdir(parents=True, exist_ok=True)
    
    # 复制当前目录下所有 .py 文件
    for py_file in glob.glob("*.py"):
        src_path = Path(py_file)
        dst_path = args.save_dir / "backup" / f"{src_path.name}.backup"
        shutil.copy(src_path, dst_path)

    # 递归复制 models 文件夹下的 .py 文件（包括子目录）
    for py_file in Path("models").rglob("*.py"):
        relative_path = py_file.relative_to("models")
        backup_path = args.save_dir / "backup" / "models" / relative_path.with_suffix(".py.backup")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(py_file, backup_path)


def nested_to_tensor(obj, dtype=torch.float32,device="cpu"):
    for panel_id,panel in obj.items():
        for k,v in panel.items():
            if k in ["translation","rotation","scale","vertices","vertices_3d","vertices_permuted","edge_points"]:
                panel[k] = torch.tensor(v, dtype=dtype,device=device)
            if k == "edge_params":
                panel[k] = [torch.tensor(item, dtype=dtype,device=device) for item in v]
    return obj       

def detach_cpu_numpy(obj):
    """
    递归地将 obj 中所有的 torch.Tensor 转为 numpy.ndarray（.detach().cpu().numpy()）
    支持 dict / list / tuple / set / 其他类型
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: detach_cpu_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_cpu_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_cpu_numpy(v) for v in obj)
    elif isinstance(obj, set):
        return {detach_cpu_numpy(v) for v in obj}
    else:
        return obj


def build_optimizer_and_scheduler(
    params,
    base_lr: float,
    weight_decay: float,
    total_steps: int,
    warmup_ratio: float = 0.05,
    warmup_start_factor: float = 1e-6,
    eta_min: float = 1e-6,
    adamw: bool = True,
):
    """
    构建单个 optimizer + scheduler（按 iteration 调度）
    - Warmup: LinearLR  (从 base_lr * warmup_start_factor 线性到 base_lr)
    - Cosine: CosineAnnealingLR (从 base_lr 余弦衰减到 eta_min)
    """
    if adamw:
        opt = optim.AdamW(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        opt = optim.Adam(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))
    decay_steps  = max(1, total_steps - warmup_steps)

    # 1) 线性 warmup：把 lr 从 base_lr * warmup_start_factor 拉到 base_lr
    sched_warmup = LinearLR(
        opt,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # 2) 余弦衰减到 eta_min
    sched_cosine = CosineAnnealingLR(
        opt,
        T_max=decay_steps,
        eta_min=eta_min,
    )

    # 3) 顺序组合：先 warmup，再 cosine
    sched = SequentialLR(
        opt,
        schedulers=[sched_warmup, sched_cosine],
        milestones=[warmup_steps],  # 在第 warmup_steps 步切换到 cosine
    )

    return opt, sched, warmup_steps, decay_steps


def setup_optim_sched_all(
    img_encoder_model,
    complex_stitch_model,
    flatten_model,
    *,
    steps_per_epoch: int,
    epochs: int,
    lr_img_encoder: float,
    lr_complex: float,
    lr_flatten: float,
    weight_decay: float,
):
    total_steps = max(1, steps_per_epoch * epochs)

    # 经验下限：encoder 更保守；head 模块略高
    opt_img, sch_img, w_img, d_img = build_optimizer_and_scheduler(
        img_encoder_model.parameters(),
        base_lr=lr_img_encoder,           # e.g., 1e-5 ~ 2e-5
        weight_decay=weight_decay,
        total_steps=total_steps,
        warmup_ratio=0.05,                # 5% warmup
        warmup_start_factor=1e-6,         # 从极小因子起步
        eta_min=max(1e-6, lr_img_encoder * 0.1),  # encoder 末期不低于起点的 10%
        adamw=True,
    )

    opt_cpx, sch_cpx, w_cpx, d_cpx = build_optimizer_and_scheduler(
        complex_stitch_model.parameters(),
        base_lr=lr_complex,               # e.g., 1e-4
        weight_decay=weight_decay,
        total_steps=total_steps,
        warmup_ratio=0.05,
        warmup_start_factor=1e-6,
        eta_min=max(5e-6, lr_complex * 0.05),     # 末期 ≈ 5% base
        adamw=True,
    )

    opt_flt, sch_flt, w_flt, d_flt = build_optimizer_and_scheduler(
        flatten_model.parameters(),
        base_lr=lr_flatten,               # e.g., 1e-4
        weight_decay=weight_decay,
        total_steps=total_steps,
        warmup_ratio=0.05,
        warmup_start_factor=1e-6,
        eta_min=max(5e-6, lr_flatten * 0.05),
        adamw=True,
    )

    return (
        (opt_img, sch_img),
        (opt_cpx, sch_cpx),
        (opt_flt, sch_flt),
        dict(
            total_steps=total_steps,
            warmup_steps=w_img,  # 三者相同
            decay_steps=d_img,   # 三者相同
        ),
    )
    
    
def train(rank,world_size,args:Args):
    torch.cuda.set_device(rank)
    setup_seed(args.seed + rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{rank}")
    complex_stitch_args = args.complex_stitch_config
    flatten_args = args.flatten_config
    img_enc_args = args.img_enc
    
    complex_stitch_model=ComplexStitchModel(complex_stitch_args).to(device)
    flatten_model=FlattenModel(flatten_args).to(device)
    img_encoder_model=Aggregator(img_enc_args).to(device)
    
    # if isinstance(args.complex_model_path,str) and len(args.complex_model_path)>0:
    #     assert os.path.exists(args.complex_model_path)
    #     state_dict = torch.load(args.complex_model_path)
    #     complex_stitch_model.load_state_dict(state_dict)
    
    # if isinstance(args.flatten_model_path,str) and len(args.flatten_model_path)>0:
    #     assert os.path.exists(args.flatten_model_path)
    #     state_dict = torch.load(args.flatten_model_path)
    #     flatten_model.load_state_dict(state_dict)
    
    # if isinstance(args.img_encoder_model_path,str) and len(args.img_encoder_model_path)>0:
    #     assert os.path.exists(args.img_encoder_model_path)
    #     state_dict = torch.load(args.img_encoder_model_path)
    #     img_encoder_model.load_state_dict(state_dict)
        
     
    
        
    train_dataset=GCD_DataSet(args.train_data,args.statistics,device=device)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.train_data.batch_size,
                               num_workers=0, drop_last=True,
                              collate_fn=custom_collate_fn, sampler=train_sampler)
    
    eval_dataset=GCD_DataSet(args.eval_data,args.statistics,device=device)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank,shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_data.batch_size,shuffle=False,
                              num_workers=0, drop_last=False,
                             collate_fn=custom_collate_fn,sampler=eval_sampler)

    
    if rank==0:
        run_log_path = args.save_dir / f'run.log'
        io = IOStreamLogger(str(run_log_path))
        usd_wandb=args.wandb.use_wandb
        if usd_wandb:
            wandb_logger=WandbLogger(args.exp_name,args)
        else:
            wandb_logger=None
        
        usd_tb=args.tensorboard.use_tb
        if usd_tb:
            tb_logger=TensorBoardLogger(args.exp_name,args.tensorboard)
        else:
            tb_logger=None
        io.cprint(str(args))
        io.cprint(str(complex_stitch_model),log_file_only=True)
    
    
        io.cprint(f"Load {len(train_dataset)} samples from {train_dataset.root}")
        io.cprint(f"Load {len(eval_dataset)} samples from {eval_dataset.root}")

    
    steps_per_epoch = len(train_loader)
    (opt_img, sch_img), (opt_cpx, sch_cpx), (opt_flt, sch_flt), info = setup_optim_sched_all(
        img_encoder_model,
        complex_stitch_model,
        flatten_model,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        lr_img_encoder=args.lr_img_encoder,
        lr_complex=args.lr_complex,
        lr_flatten=args.lr_flatten,
        weight_decay=args.weight_decay,
    )
    
    if isinstance(args.resume_path,str) and len(args.resume_path)>0:
        ckpt = torch.load(args.resume_path, map_location=device)
        img_encoder_model.load_state_dict(ckpt["models"]["img"])        # 去掉 .module
        complex_stitch_model.load_state_dict(ckpt["models"]["complex"])
        flatten_model.load_state_dict(ckpt["models"]["flatten"])
        opt_img.load_state_dict(ckpt["optimizers"]["img"])
        opt_cpx.load_state_dict(ckpt["optimizers"]["complex"])
        opt_flt.load_state_dict(ckpt["optimizers"]["flatten"])
        sch_img.load_state_dict(ckpt["schedulers"]["img"])
        sch_cpx.load_state_dict(ckpt["schedulers"]["complex"])
        sch_flt.load_state_dict(ckpt["schedulers"]["flatten"])
        start_epoch = ckpt["epoch"] + 1
    else:
        start_epoch = 1   # ✅ 建议从 1 开始
    
        
    img_encoder_model = DDP(
        img_encoder_model,
        device_ids=[rank],
    )
    complex_stitch_model = DDP(
        complex_stitch_model,
        device_ids=[rank],
    )
    flatten_model = DDP(
        flatten_model,
        device_ids=[rank],
    )
    
    train_loss_manager=LossManager()
    eval_loss_manager=LossManager(eval=True)
    
    curve_weight_dict = {'curve_loss_ce': complex_stitch_args.class_loss_coef, 'curve_loss_geometry': complex_stitch_args.curve_geometry_loss_coef}
    patch_weight_dict = {'patch_loss_ce': complex_stitch_args.class_loss_coef, 'patch_loss_geometry': complex_stitch_args.patch_geometry_loss_coef}
    panel_weight_dict = {"edge_loss_geometry": flatten_args.edge_geometry_loss_coef,"scale_loss": flatten_args.scale_loss_coef}
    matching_weight_dict = {"patch_curve_matching_loss_topo": complex_stitch_args.patch_curve_topo_loss_coef,}
    
    train_loss_manager.add_loss_terms(patch_weight_dict|curve_weight_dict|panel_weight_dict|matching_weight_dict)
    eval_loss_manager.add_loss_terms(patch_weight_dict|curve_weight_dict|panel_weight_dict|matching_weight_dict)
    
    
    
    matcher_curve=build_matcher_curve(args.train_data.batch_size, complex_stitch_args)
    curve_eos_coef_cal = complex_stitch_args.curve_avg_count / (complex_stitch_args.n_curve_queries - complex_stitch_args.curve_avg_count) * complex_stitch_args.global_invalid_weight
    curve_loss_criterion = SetCriterion_Curve(matcher_curve,curve_eos_coef_cal)
    
    
    matcher_patch=build_matcher_patch(complex_stitch_args)
    patch_eos_coef_cal = complex_stitch_args.patch_avg_count / (complex_stitch_args.n_patch_queries - complex_stitch_args.patch_avg_count) * complex_stitch_args.global_invalid_weight
    patch_loss_criterion=SetCriterion_Patch(matcher_patch, patch_eos_coef_cal)
    
    flatten_losss_criterion = FlattenLoss(flatten_args)
    
    
    for epoch in tqdm(range(start_epoch,args.epochs+1),desc="training(epoch)...", disable=(rank != 0)):
        ####################
        # Train
        ####################
        complex_stitch_model.train()
        img_encoder_model.train()
        flatten_model.train()
        train_loss_manager.reset_accumulate()
        train_sampler.set_epoch(epoch)

        for data in tqdm(train_loader, disable=(rank != 0)):
            # st=time.time()
            name=[]
            imgs=[]
            patch_points=[]
            curves=[]
            PC_mat=[]
            edge_points=[]
            panel_scale=[]
            for it in data:
                name.append(it['name'])
                imgs.append(torch.tensor(it['images'],device=device,dtype=torch.float32))
                
                patch_points.append([torch.tensor(p,device=device,dtype=torch.float32)
                                     for p in it['patch_points']])
                curves.append(torch.tensor(it['curves_sampled'],device=device,dtype=torch.float32))
                PC_mat.append(torch.tensor(it['PC_mat'],device=device,dtype=torch.float32))
                edge_points.append([torch.tensor(e,device=device,dtype=torch.float32)
                                    for e in it['edge_points']])
                panel_scale.append(torch.tensor(it["panel_scale"],device=device,dtype=torch.float32))

            imgs=torch.stack(imgs,dim=0)
            
            # print(f"   data load:{time.time()-st}")
            opt_img.zero_grad()
            opt_cpx.zero_grad()
            opt_flt.zero_grad()
            
            # st=time.time()
            # time consuming...
            img_tokens,_=img_encoder_model(imgs)
            B,S,N,D=img_tokens.shape
            img_tokens=img_tokens.reshape(B,S*N,D)
            # print(f"  image encoder {time.time()-st}")
            
            # st=time.time()
            curve_predictions,patch_predictions, curve_features, patch_features= complex_stitch_model(img_tokens)
            # print(f"  3d pred: {time.time()-st}")
            
            # st=time.time()
            curve_loss_dict, curve_matching_indices = curve_loss_criterion(curve_predictions, curves)
            patch_loss_dict, patch_matching_indices = patch_loss_criterion(patch_predictions, patch_points)
            train_loss_manager.update(curve_loss_dict|patch_loss_dict)
            # print(f"  matching: {time.time()-st}")

            patch_curve_matching_loss_topo,patch_curve_topo_acc =\
                Patch_Curve_Matching(curve_predictions,patch_predictions,curves,patch_points,PC_mat,
                                curve_matching_indices,patch_matching_indices)

            train_loss_manager.update({"patch_curve_matching_loss_topo": patch_curve_matching_loss_topo,})

            # st=time.time()
            flatten_pred=flatten_model(curve_features, patch_features,\
                curve_matching_indices, patch_matching_indices,PC_mat)
            
            panel_loss_dict=flatten_losss_criterion(flatten_pred,edge_points,panel_scale)
            train_loss_manager.update(panel_loss_dict)
            # print(f"  flatten: {time.time()-st}")
            
            train_loss_manager.step()
            opt_img.step()
            opt_cpx.step()
            opt_flt.step()
            sch_img.step()
            sch_cpx.step()
            sch_flt.step()
            
        train_loss_manager.reduce_loss_dict()
        if rank == 0 and (epoch % args.save_weight_freq == 0 or epoch == 1):
            (args.save_dir / "weights" / "complex_stitch").mkdir(parents=True, exist_ok=True)
            (args.save_dir / "weights" / "flatten").mkdir(parents=True, exist_ok=True)
            (args.save_dir / "weights" / "img_encoder").mkdir(parents=True, exist_ok=True)

            # 1) 仍然分别保存纯权重（和你原来兼容）
            torch.save(complex_stitch_model.module.state_dict(),
                    str(args.save_dir / "weights" / "complex_stitch" / f"{epoch}.pth"))
            torch.save(flatten_model.module.state_dict(),
                    str(args.save_dir / "weights" / "flatten" / f"{epoch}.pth"))
            torch.save(img_encoder_model.module.state_dict(),
                    str(args.save_dir / "weights" / "img_encoder" / f"{epoch}.pth"))

            # 2) 额外保存“可断点恢复”的统一 checkpoint（含优化器/调度器/epoch）
            ckpt = {
                "epoch": epoch,
                "models": {
                    "img":     img_encoder_model.module.state_dict(),
                    "complex": complex_stitch_model.module.state_dict(),
                    "flatten": flatten_model.module.state_dict(),
                },
                "optimizers": {
                    "img":     opt_img.state_dict(),
                    "complex": opt_cpx.state_dict(),
                    "flatten": opt_flt.state_dict(),
                },
                "schedulers": {
                    "img":     sch_img.state_dict(),
                    "complex": sch_cpx.state_dict(),
                    "flatten": sch_flt.state_dict(),
                },
            }
            torch.save(ckpt, str(args.save_dir / "weights" / f"ckpt_e{epoch}.pt"))
            torch.save(ckpt, str(args.save_dir / "weights" / "last.pt"))  # 覆盖式保存最新


        
        
        # ####################
        # # Eval
        # ####################
        complex_stitch_model.eval()
        flatten_model.eval()
        # img_encoder_model.eval()
        eval_loss_manager.reset_accumulate()
        with torch.no_grad(): 
            for data in tqdm(eval_loader,desc="eval...", disable=(rank != 0)):
                name=[]
                imgs=[]
                patch_points=[]
                curves=[]
                PC_mat=[]
                edge_points=[]
                panel_scale=[]
                for it in data:
                    name.append(it['name'])
                    imgs.append(torch.tensor(it['images'],device=device,dtype=torch.float32))
                    
                    patch_points.append([torch.tensor(p,device=device,dtype=torch.float32)
                                        for p in it['patch_points']])
                    curves.append(torch.tensor(it['curves_sampled'],device=device,dtype=torch.float32))
                    PC_mat.append(torch.tensor(it['PC_mat'],device=device,dtype=torch.float32))
                    edge_points.append([torch.tensor(e,device=device,dtype=torch.float32)
                                        for e in it['edge_points']])
                    panel_scale.append(torch.tensor(it["panel_scale"],device=device,dtype=torch.float32))

                imgs=torch.stack(imgs,dim=0)
                img_tokens,_=img_encoder_model(imgs)

                B,S,N,D=img_tokens.shape
                img_tokens=img_tokens.reshape(B,S*N,D)
                
                curve_predictions,patch_predictions, curve_features, patch_features= complex_stitch_model(img_tokens)
                    
                curve_loss_dict, curve_matching_indices = curve_loss_criterion(curve_predictions, curves)
                patch_loss_dict, patch_matching_indices = patch_loss_criterion(patch_predictions, patch_points)
                eval_loss_manager.update(curve_loss_dict|patch_loss_dict)
                
                patch_curve_matching_loss_topo,patch_curve_topo_acc =\
                    Patch_Curve_Matching(curve_predictions,patch_predictions,curves,patch_points,PC_mat,
                                    curve_matching_indices,patch_matching_indices)
                eval_loss_manager.update({
                                    "patch_curve_matching_loss_topo": patch_curve_matching_loss_topo,})
                
                
                flatten_pred=flatten_model(curve_features, patch_features,\
                curve_matching_indices, patch_matching_indices, PC_mat,)
                
                panel_loss_dict=flatten_losss_criterion(flatten_pred,edge_points,panel_scale)
                eval_loss_manager.update(panel_loss_dict)
                    
                
                eval_loss_manager.step()
                # save pred
                if epoch%args.save_pred_freq==0:
                    save_dic_batch= {**curve_predictions, **patch_predictions}
                    batch_size=len(flatten_pred)
                    for i in range(batch_size):
                        cur_name = name[i]
                        save_path = args.save_dir / "pred" / "val" /str(epoch)/ f"{cur_name}.npz"
                        # 构造当前样本的保存字典：取 batch 第 i 项并转为 numpy
                        save_dict = {}
                        for k, v in save_dic_batch.items():
                            save_dict[k] = v[i].detach().cpu().numpy()  # (B, ...) → 第 i 项
                        cur_flatten_pred = flatten_pred[i]
                        save_dict["flatten_pred"] = detach_cpu_numpy(cur_flatten_pred)
                        # 保存为 .npz 文件
                        np.savez_compressed(save_path, **save_dict)
                
        eval_loss_manager.reduce_loss_dict()
        
        
        if rank==0:
            out_msg=train_loss_manager.get_log()|eval_loss_manager.get_log()
            # out_msg=train_loss_manager.get_log()

            if usd_wandb:
                wandb_logger.log(out_msg)
            if usd_tb:
                tb_logger.log(out_msg)
            
            io.cprint(str(out_msg))
    
    if rank==0:
        io.cprint("Training finished.")
        io.close()
    
    if rank ==0 and dist.is_initialized():
        dist.destroy_process_group()
        
        

def test(rank,world_size,args:Args):
    torch.cuda.set_device(rank)
    setup_seed(args.seed + rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    complex_stitch_args = args.complex_stitch_config
    flatten_args = args.flatten_config
    img_enc_args = args.img_enc
    
    complex_stitch_model=ComplexStitchModel(complex_stitch_args).to(device)
    flatten_model=FlattenModel(flatten_args).to(device)
    img_encoder_model=Aggregator(img_enc_args).to(device)
    
    if isinstance(args.complex_model_path,str) and len(args.complex_model_path)>0:
        assert os.path.exists(args.complex_model_path)
        state_dict = torch.load(args.complex_model_path)
        complex_stitch_model.load_state_dict(state_dict)
    
    if isinstance(args.flatten_model_path,str) and len(args.flatten_model_path)>0:
        assert os.path.exists(args.flatten_model_path)
        state_dict = torch.load(args.flatten_model_path)
        flatten_model.load_state_dict(state_dict)

    if isinstance(args.img_encoder_model_path,str) and len(args.img_encoder_model_path)>0:
        assert os.path.exists(args.img_encoder_model_path)
        state_dict = torch.load(args.img_encoder_model_path)
        img_encoder_model.load_state_dict(state_dict)
    
    
    img_encoder_model = DDP(
        img_encoder_model,
        device_ids=[rank],
    ) 
    complex_stitch_model = DDP(
        complex_stitch_model,
        device_ids=[rank],
    )
    flatten_model = DDP(
        flatten_model,
        device_ids=[rank],
        # find_unused_parameters=True
    )
    
    if args.test_data.data_type=="test_gcd":
        test_dataset=TestDataSet_GCD(args.test_data,args.statistics,device=device)
    elif args.test_data.data_type=="test_4d_dress":
        test_dataset=TestDataSet_4D_Dress(args.test_data,args.statistics,device=device)
        
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank,shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_data.batch_size,shuffle=False,
                              num_workers=0, drop_last=False,
                             collate_fn=custom_collate_fn,sampler=test_sampler)
    
    if rank == 0:
        run_log_path = args.save_dir / f'run.log'
        io = IOStreamLogger(str(run_log_path))
        io.cprint(str(args))
        io.cprint(str(complex_stitch_model),log_file_only=True)
        io.cprint(f"Load {len(test_dataset)} samples from {test_dataset.root}")
    
    complex_stitch_model.eval()
    with torch.no_grad(): 
        for data in tqdm(test_loader,desc="test...", disable=(rank != 0)):     
            name=[]
            imgs=[]
            for it in data:
                name.append(it['name'])
                imgs.append(torch.tensor(it['images'],device=device,dtype=torch.float32))
            
            B=len(name)
            exist_flag_lst=torch.zeros((B))   
            for i in range(B):
                cur_name = name[i]
                cur_save_dir = args.save_dir / "pred" /"test"/ f"{cur_name}"
                cur_save_dir.mkdir(exist_ok=True)
                
                save_path = cur_save_dir/f"{cur_name}.npz"
                exist_flag_lst[i]=save_path.exists()
            # if exist_flag_lst.all():
            #     continue
            
            imgs=torch.stack(imgs,dim=0)
            img_tokens,_=img_encoder_model(imgs)
            B,S,N,D=img_tokens.shape
            img_tokens=img_tokens.reshape(B,S*N,D)
            
            curve_predictions,patch_predictions, curve_features, patch_features= complex_stitch_model(img_tokens)
            
            pred_patch_points_scaled=complex_stitch_model.module.get_scaled_points(patch_features)
             
            patch_predictions['pred_patch_points_scaled']=pred_patch_points_scaled['pred_patch_points_scaled']
            
            all_pred=flatten_model.module.infer(curve_predictions, patch_predictions, 
                                                    curve_features, patch_features,names=name)
            
            for i in range(B):
                cur_name = name[i]
                cur_save_dir = args.save_dir / "pred" /"test"/ f"{cur_name}"
                cur_save_dir.mkdir(exist_ok=True)
                
                save_path = cur_save_dir/f"{cur_name}.npz"
                cur_pred = all_pred[i]
                save_dict = detach_cpu_numpy(cur_pred)
                
                save_dict['patch_points_scaled']=np.array(save_dict['patch_points_scaled'],dtype=object)
                # 保存为 .npz 文件
                np.savez_compressed(save_path, **save_dict)
    
    if rank == 0:
        io.cprint("Testing finished.")
        io.close()
    
    if rank ==0 and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    config_path="configs/test_ori_4d_dress.yaml"
    # config_path="configs/test_ori_gcd.yaml"
    
    args=parse_args(config_path)

    init_saved_dir(args)
    
    world_size = torch.cuda.device_count()
    rank=int(os.environ["RANK"])
    local_rank=int(os.environ["LOCAL_RANK"])
    args.rank=rank
    args.local_rank=local_rank
    
    if not args.eval:
        train(local_rank,world_size,args)
    else:
        test(local_rank,world_size,args)
    