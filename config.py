import dataclasses
from omegaconf import OmegaConf
import tyro
from dataclasses import field
from dataclasses import is_dataclass
from pathlib import Path
import torch
import argparse

@dataclasses.dataclass
class DatasetConfig:
    root: str = ""
    batch_size: int = 32
    data_type: str = "train"
    samples: str = ""
    texture_type: str = ""

@dataclasses.dataclass
class WandbConfig:
    use_wandb: bool = False
    project_name: str = ""
    log_dir: str = "experiments"
    
@dataclasses.dataclass
class TensorboardConfig:
    use_tb: bool = False
    project_name: str = ""
    log_dir: str = "experiments"
    

@dataclasses.dataclass
class Statistics:
    img_mean: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    img_std: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

@dataclasses.dataclass
class ComplexStitchConfig:
    d_model: int = 768
    
    topo_embed_dim: int=1024
    points_per_curve: int=50    
    points_per_patch_dim: int=20        

    
    n_curve_queries: int = 200
    n_patch_queries: int = 70
    
    curve_avg_count: int = 35
    patch_avg_count: int = 9
    
    val_th: float = 0.5
    class_loss_coef: float = 1
    curve_geometry_loss_coef: float = 300
    patch_geometry_loss_coef: float = 300
    
    global_invalid_weight: float=1
    
    patch_curve_topo_loss_coef: float=1

@dataclasses.dataclass
class FlattenConfig:
    edge_classify_loss_coef: float = 1
    edge_geometry_loss_coef: float = 300
    
    points_per_edge: int = 50
    scale_loss_coef: float = 1

@dataclasses.dataclass
class ImageEncoderConfig:
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    patch_embed: str = "dinov2_vits14_reg"
    aa_block_size: int = 1
    qk_norm: bool = True
    rope_freq: int = 100
    init_values: float = 0.01
    
    dino_path: str = "/inspire/hdd/global_user/liming-253108120187/GarmentRecon/Model_final/vggtencoder/weights/dinov2_vits14_reg4_pretrain.pth"

@dataclasses.dataclass
class Args:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    tensorboard: TensorboardConfig = field(default_factory=TensorboardConfig)
    train_data: DatasetConfig = field(default_factory=DatasetConfig)
    eval_data: DatasetConfig = field(default_factory=DatasetConfig)
    test_data: DatasetConfig = field(default_factory=DatasetConfig)
    
    complex_stitch_config: ComplexStitchConfig = field(default_factory=ComplexStitchConfig)
    flatten_config: FlattenConfig = field(default_factory=FlattenConfig)
    img_enc: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    statistics: Statistics = field(default_factory=Statistics)
    
    exp_name: str = "exp"
    save_dir: str|int = "ckpts"
    
    device: str = "cuda"
    seed: int = 42
    
    eval: bool = False
    complex_model_path: str = ""
    flatten_model_path: str = ""
    img_encoder_model_path: str = ""
    resume_path: str = ""
    
    # training
    epochs: int = 250
    lr_complex: float = 0.001
    lr_flatten: float = 0.001
    lr_img_encoder: float = 0.001
    weight_decay: float = 1e-4
    save_weight_freq: int = 50
    save_pred_freq: int = 50
    warmup: int = 10
    

def parse_args(yaml_path: str) -> Args:
    """从 YAML 加载默认配置"""
    yaml_conf = OmegaConf.load(yaml_path)
    default_cfg = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Args), yaml_conf)
    )
    
    args: Args = tyro.cli(Args, default=default_cfg)
    

    args.device=torch.device(args.device)
    
    args.save_dir = Path(args.save_dir) / args.exp_name
    args.save_dir.mkdir(parents=True, exist_ok=True)
    return args



def args_to_dict(obj):
    """
    将 dataclass 实例（可以嵌套）递归转换为字典。
    """
    if is_dataclass(obj):
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = args_to_dict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [args_to_dict(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: args_to_dict(v) for k, v in obj.items()}
    else:
        return obj


if __name__ == "__main__":
    import torch

    args = parse_args("config/config.yaml")
    dic_args = args_to_dict(args)
    
    
    import ipdb
    ipdb.set_trace()
    
    
