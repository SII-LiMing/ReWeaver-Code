# ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction

<div align="Middle">
  <a href="https://sii-liming.github.io/ReWeaver/static/pdfs/paper.pdf" target="_blank"><img src="https://img.shields.io/badge/Paper-b5212f.svg?logo=adobeacrobatreader&logoColor=white" height="22px"></a>
  <a href="https://sii-liming.github.io/ReWeaver/" target="_blank"><img src="https://img.shields.io/badge/Project%20Page-2563eb.svg" height="22px"></a>
  <a href="https://huggingface.co/datasets/SII-LiMing/ReWeaver-GCD-TS" target="_blank"><img src="https://img.shields.io/badge/Data-HuggingFace-yellow" height="22px"></a>
</div>

## 🌟 Overview

**ReWeaver** reconstructs structured 3D garments and 2D sewing patterns from sparse multi-view RGB images. The method jointly predicts 3D curves, 3D patches, patch-curve connectivity, and flattened 2D panel edges, producing garment assets that are better aligned with downstream simulation and editing pipelines.

This repository contains:

- the VGGT-style multi-view image encoder,
- the 3D geometry and topology prediction modules,
- the 2D flattening module,
- training and evaluation code,
- data loaders for GCD-style garment data.

---

## 🎯 TODO List

- [x] 🚀 Training / evaluation code
- [x] 📂 Config files
- [x] 📄 Project page
- [x] 🤗 GCD-TS dataset link
- [ ] 📦 Clean environment file
- [ ] 🏋️ Pretrained model weights

---

## Get Started with ReWeaver

### 🛠️ Preparation

1. **💻 Environment.**  
   The current codebase was developed with Python `3.12.9` and CUDA `12.4`.

```bash
git clone https://github.com/SII-LiMing/ReWeaver-Code.git
cd ReWeaver-Code

conda create -n reweaver python=3.12.9
conda activate reweaver

pip install ipdb tqdm tyro omegaconf trimesh wandb scipy tensorboard matplotlib pillow scikit-image
pip install torch torchvision torchaudio
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

2. **📥 Pretrained initialization weights.**  
   The image encoder config expects a DINOv2 initialization checkpoint via `img_enc.dino_path` in the YAML config. Before running experiments, update this path to a valid local checkpoint on your machine.

3. **⚙️ Config files.**  
   Main configs are stored in [`configs/`](configs/). Before training or testing, you should update:

- `train_data.root`, `eval_data.root`, `test_data.root`
- `save_dir`, `exp_name`
- `img_enc.dino_path`
- `complex_model_path`, `flatten_model_path`, `img_encoder_model_path` for evaluation

---

### 📂 Training Dataset

We train ReWeaver on **GCD-TS**, a textured multi-view garment dataset introduced in the paper. The public dataset link is:

- 🤗 https://huggingface.co/datasets/SII-LiMing/ReWeaver-GCD-TS

For the current codebase, each GCD-style sample directory is expected to contain:

- RGB images under:
  - `render_output/rgb/` when `texture_type: tileable`
  - `render_output_ori_texture/rgb/` when `texture_type: ori`
- a panel annotation file:
  - `<sample_name>_2d_panel.json`

The loader scans each direct child folder under `train_data.root`, `eval_data.root`, or `test_data.root` unless `samples` is set to a text file listing selected samples.

---

### 🚀 Training

The current training / evaluation entrypoint is [`main.py`](main.py).  
One important caveat is that the config path is currently hardcoded at the bottom of the file. In practice, you should change it to the YAML you want to run, for example:

```python
if __name__ == "__main__":
    config_path="configs/train_tileable.yaml"
```

Before launching a run, edit `config_path` to the YAML you want to use, for example:

- [`configs/train_tileable.yaml`](configs/train_tileable.yaml)
- [`configs/train_ori.yaml`](configs/train_ori.yaml)

Then launch distributed training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py
```

The repo also includes a minimal launch example in [`bash_scripts/run.sh`](bash_scripts/run.sh).

---

### 🧪 Evaluation

For evaluation, set in the chosen YAML:

- `eval: True`
- `test_data.root`
- `complex_model_path`
- `flatten_model_path`
- `img_encoder_model_path`

Recommended test configs in this repo:

- [`configs/test_ori_gcd.yaml`](configs/test_ori_gcd.yaml)

Then run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py
```

Predictions are saved as compressed `.npz` files under:

```text
<save_dir>/<exp_name>/pred/test/<sample_name>/<sample_name>.npz
```

---

## 🗂️ Repository Structure

```text
ReWeaver-Code/
├── main.py
├── config.py
├── data.py
├── configs/
├── models/
├── utils/
├── vggtencoder/
└── bash_scripts/
```

---

## 📝 Notes

- `main.py` does not currently accept `--config`; you need to edit the hardcoded `config_path`.
- Existing YAML files still contain author-local absolute paths and should be changed before running on a new machine.
- The training script creates:

```text
<save_dir>/<exp_name>/
├── backup/
├── pred/
└── weights/
    ├── complex_stitch/
    ├── flatten/
    └── img_encoder/
```

- Logging backends supported by the code are:
  - Weights & Biases
  - TensorBoard
  - plain text logger

---

## 💖 Acknowledgement

This repository builds on open-source tools and prior work from the community. Many thanks to the authors for sharing their code and resources.

- 🔗 [DINOv2](https://github.com/facebookresearch/dinov2)
- 🔗 [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- 🔗 [VGGT-related encoder design in this repo](vggtencoder/)

---

## 📜 Citation

If you find this repository useful in your research, please cite:

```bibtex
@inproceedings{li2026reweaver,
  title={ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction},
  author={Li, Ming and Shan, Hui and Zheng, Kai and Shen, Chentao and Liu, Siyu and Fu, Yanwei and Chen, Zhen and Huang, Xiangru},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026},
  url={https://sii-liming.github.io/ReWeaver/}
}
```
