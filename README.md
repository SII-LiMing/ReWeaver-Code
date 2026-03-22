# ReWeaver
## Installation

### 1. Create a Conda Environment
First, create a new Conda environment with Python 3.12.9:

```bash
conda create -n ReWeaver python=3.12.9
conda activate ReWeaver
```

### 2. Install Required Packages
After activating the environment, install the necessary dependencies:

```bash
pip install ipdb
pip3 install torch torchvision torchaudio
pip install tqdm
pip install tyro
pip install omegaconf
pip install trimesh
pip install wandb
pip install scipy
pip install tensorboard
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```


## Configuration Files

- The main configurations for the model are stored in the `configs/` directory. This includes training parameters, dataset paths, and pretrained weights.
- Modify the configuration files in this directory to fit your training and testing environment.


## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 main.py
```
## Testing


```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 main.py --eval
```


