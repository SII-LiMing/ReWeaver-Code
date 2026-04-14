# ReWeaver-Code

Official training and inference code for **ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction**.

- Project page: https://sii-liming.github.io/ReWeaver/
- Dataset: https://huggingface.co/datasets/SII-LiMing/ReWeaver-GCD-TS
- Paper page PDF entry: https://sii-liming.github.io/ReWeaver/static/pdfs/paper.pdf

## Overview

ReWeaver reconstructs structured 3D garments and 2D sewing patterns from sparse multi-view RGB images. This repository contains:

- the VGGT-style multi-view image encoder,
- the 3D geometry/topology prediction modules,
- the 2D flattening module,
- training and evaluation entrypoints,
- dataset loaders for GCD-style data and 4D-Dress-style test data.

## Repository Structure

```text
ReWeaver-Code/
├── main.py                # training / evaluation entrypoint
├── config.py              # dataclass-based config parsing
├── data.py                # dataset loaders
├── configs/               # experiment configs
├── models/                # core model, matcher, loss, flatten modules
├── vggtencoder/           # image encoder implementation
├── utils/                 # geometry / IO / image utilities
└── bash_scripts/          # environment notes and launch examples
```

## Environment

The current codebase was developed with:

- Python `3.12.9`
- CUDA toolkit `12.4`
- PyTorch + `torchrun` for distributed training/evaluation

### Conda Setup

```bash
conda create -n reweaver python=3.12.9
conda activate reweaver
```

### Python Dependencies

There is no pinned `requirements.txt` in the repository yet, so installation currently follows the packages used by the codebase and `bash_scripts/env.sh`:

```bash
pip install ipdb tqdm tyro omegaconf trimesh wandb scipy tensorboard matplotlib pillow scikit-image
pip install torch torchvision torchaudio
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

If your local PyTorch/CUDA combination requires a specific wheel source, install `torch`, `torchvision`, and `torchaudio` using the matching command from the official PyTorch site first, then install the remaining packages.

## Data Preparation

### Supported Data Modes

The current code supports three data entry modes through `DatasetConfig.data_type`:

- `train` / `eval`: GCD-style training and validation data via `GCD_DataSet`
- `test_gcd`: GCD-style test data via `TestDataSet_GCD`
- `test_4d_dress`: sparse-view external data via `TestDataSet_4D_Dress`

### Expected GCD-Style Directory Layout

Each sample directory is expected to contain:

- RGB images under either:
  - `render_output/rgb/` when `texture_type: tileable`
  - `render_output_ori_texture/rgb/` when `texture_type: ori`
- a 2D pattern annotation file:
  - `<sample_name>_2d_panel.json`

The loader scans `train_data.root`, `eval_data.root`, or `test_data.root` and treats each direct child directory as one sample unless `samples` is set to a text file listing relative sample paths.

### 4D-Dress-Style Test Layout

For `test_4d_dress`, each sample directory is expected to contain:

- `view/` for input images
- optionally `mask/` for foreground masks

## Configuration

Experiment configuration is defined by YAML files in [`configs/`](configs/).

Examples already included in the repo:

- [`configs/train_tileable.yaml`](configs/train_tileable.yaml)
- [`configs/train_ori.yaml`](configs/train_ori.yaml)
- [`configs/test_ori_gcd.yaml`](configs/test_ori_gcd.yaml)
- [`configs/test_ori_4d_dress.yaml`](configs/test_ori_4d_dress.yaml)

Important config fields:

- `train_data.root`, `eval_data.root`, `test_data.root`: dataset locations
- `texture_type`: `tileable` or `ori`
- `save_dir`, `exp_name`: output directory root and experiment name
- `eval`: `False` for training, `True` for testing
- `resume_path`: full training checkpoint path
- `complex_model_path`, `flatten_model_path`, `img_encoder_model_path`: model weights used in evaluation mode
- `img_enc.dino_path`: path to the DINOv2 initialization weights

## Important Entry-Point Note

`main.py` currently does **not** accept a `--config` argument directly. Instead, the config file path is hardcoded at the bottom of [`main.py`](main.py):

```python
if __name__ == "__main__":
    config_path="configs/test_ori_4d_dress.yaml"
    # config_path="configs/test_ori_gcd.yaml"
```

Before running an experiment, edit this `config_path` to the YAML file you want to use.

## Training

1. Edit the target YAML in `configs/`:
   - set dataset roots,
   - set `save_dir` and `exp_name`,
   - set `img_enc.dino_path`,
   - make sure `eval: False`.
2. In [`main.py`](main.py), set `config_path` to your training YAML.
3. Launch with `torchrun`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py
```

The same example is also reflected in [`bash_scripts/run.sh`](bash_scripts/run.sh).

## Evaluation / Inference

1. Edit the test YAML:
   - set `test_data.root`,
   - set `complex_model_path`,
   - set `flatten_model_path`,
   - set `img_encoder_model_path`,
   - set `eval: True`.
2. In [`main.py`](main.py), point `config_path` to the test YAML.
3. Launch with `torchrun`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py
```

In evaluation mode, predictions are saved as compressed `.npz` files under:

```text
<save_dir>/<exp_name>/pred/test/<sample_name>/<sample_name>.npz
```

## Outputs

At startup, the code creates:

```text
<save_dir>/<exp_name>/
├── backup/               # backup copies of Python source files
├── pred/                 # saved predictions
└── weights/
    ├── complex_stitch/
    ├── flatten/
    └── img_encoder/
```

The training script also prepares periodic prediction directories for train/val snapshots based on `save_pred_freq`.

## Logging

The repository includes three logging backends:

- Weights & Biases
- TensorBoard
- plain text IO logger

Logging behavior is controlled in the YAML configs through:

- `wandb.use_wandb`
- `tensorboard.use_tb`

## Current Limitations / Notes

- There is no frozen environment file yet.
- Running a new experiment requires editing the hardcoded `config_path` in `main.py`.
- Configs in this repository still contain author-local absolute paths and should be rewritten for your machine before use.
- The repo currently ships code and configs only; pretrained weights are referenced by path but are not included here.

## Citation

```bibtex
@inproceedings{li2026reweaver,
  title={ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction},
  author={Li, Ming and Shan, Hui and Zheng, Kai and Shen, Chentao and Liu, Siyu and Fu, Yanwei and Chen, Zhen and Huang, Xiangru},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026},
  url={https://sii-liming.github.io/ReWeaver/}
}
```
