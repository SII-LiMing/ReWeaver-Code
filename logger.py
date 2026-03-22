import os
import matplotlib.figure
from omegaconf import OmegaConf
from config import WandbConfig,args_to_dict
import logging
from typing import Union
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from config import TensorboardConfig

class IOStreamLogger:
    def __init__(self, path, level=logging.INFO):
        self.logger = logging.getLogger("IOStreamLogger")
        self.logger.setLevel(level)

        if not self.logger.handlers:
            self.file_handler = logging.FileHandler(path, mode='a')
            self.file_handler.setLevel(level)

            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(level)

            formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m-%d %H:%M:%S')
            self.file_handler.setFormatter(formatter)
            self.console_handler.setFormatter(formatter)

            self.logger.addHandler(self.file_handler)
            self.logger.addHandler(self.console_handler)

    def cprint(self, text, log_file_only=False):
        if log_file_only:
            # 暂时移除 console handler
            self.logger.removeHandler(self.console_handler)
            self.logger.info(text)
            # 重新添加 console handler
            self.logger.addHandler(self.console_handler)
        else:
            self.logger.info(text)

    def close(self):
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()


class WandbLogger:
    """
    Log using `Weights and Biases`.
    """
    def __init__(self,exp_name, args:WandbConfig):        
        import wandb
        self._wandb = wandb
        self.wandb_args=args
        self.exp_name=exp_name
        self.exp_args=args_to_dict(args)
        # Initialize a W&B run
        if self._wandb.run is None:
            self._wandb.init(
                project=self.wandb_args.project_name,
                config=self.exp_args,
                dir=self.wandb_args.log_dir,
                name=self.exp_name
            )
    
    
    def log(self, data):
        assert isinstance(data, dict)
        log_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray) or isinstance(v, Image.Image) or isinstance(v,matplotlib.figure.Figure) :
                log_data[k] = self._wandb.Image(v)
            else:
                log_data[k] = v
        self._wandb.log(log_data,commit=True)
        
    def log_metrics(self, metrics,step=None): 
        """
        Log train/validation metrics onto W&B.

        metrics: dictionary of metrics to be logged
        """
        if step is not None:
            self._wandb.log(metrics,step=step,commit=True)
        else:
            self._wandb.log(metrics,commit=True)

    def log_pic(self, images: Union[np.ndarray, Image.Image, dict],image_name="image", step=None):
        """
        Log image(s) to Weights & Biases.

        Args:
            images: single image (np.ndarray or PIL.Image) or dict of images {'name': img}
            step: optional logging step
        """
        if isinstance(images, dict):
            data = {k: self._wandb.Image(v) for k, v in images.items()}
        else:
            data = {image_name: self._wandb.Image(images)}

        if step is not None:
            self._wandb.log(data, step=step)
        else:
            self._wandb.log(data)
        

class TensorBoardLogger:
    """
    Log using TensorBoard (SummaryWriter).
    """
    def __init__(self,exp_name, args:TensorboardConfig):
        self.log_dir = args.log_dir
        self.exp_name = exp_name
        self.writer = SummaryWriter(log_dir=f"{self.log_dir}/{self.exp_name}")
        self.global_step = 0

    def log(self, data: dict, step: int = None):
        step = step if step is not None else self.global_step

        for k, v in data.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step)

            elif isinstance(v, np.ndarray):
                if v.ndim == 3:  # CHW
                    self.writer.add_image(k, v, global_step=step)
                elif v.ndim == 2:  # HW
                    self.writer.add_image(k, v[np.newaxis, ...], global_step=step)

            elif isinstance(v, Image.Image):
                tensor = T.ToTensor()(v)
                self.writer.add_image(k, tensor, global_step=step)

            elif isinstance(v, plt.Figure):
                buf = io.BytesIO()
                v.savefig(buf, format='png')
                buf.seek(0)
                image = Image.open(buf)
                self.writer.add_image(k, T.ToTensor()(image), global_step=step)
                buf.close()

        self.global_step += 1
        self.writer.flush()
        

if __name__=="__main__":
    io=IOStreamLogger("test")
    io.cprint(str(
        {"111":22,
         "bbb":33}
    ))


