import os

from classification_module import ClassificationLightningModule
import hydra
from omegaconf import DictConfig

# import torch
# from torch.nn import functional as F
# from torch.nn import Conv2d, Linear, MaxPool2d
# from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
# from torchvision import transforms
# from torchvision import models
# import torchvision

import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from torch_lr_finder import LRFinder
# from torchvision import datasets

import argparse
from argparse import Namespace

# from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize
# from albumentations.pytorch import ToTensor

# from albumentations import *
# from albumentations.pytorch import ToTensor

# from warmup_scheduler import GradualWarmupScheduler
# from onecyclelr import OneCycleLR 

import helpers
import yaml
import time

#parser = argparse.ArgumentParser()
#parser.add_argument('--config', metavar='FILE', type=str,
#                               help=' config file ')
#args = parser.parse_args()
#print(args.config)
#config = yaml.load(open(args.config))
#hparams = Namespace(**config)
#
#
## hparams = #parser.parse_args()
## 
#print(hparams)
@hydra.main(config_path="config/config.yaml")
def train_app(cfg):
    print(cfg.pretty())
    hparams = Namespace(**cfg)

    datetime_str = time.strftime("%Y-%m-%d_%H:%M")
    hparams.data_dir = os.environ["DATASET_FOLDER"] +"/"+ hparams.data_dir
    hparams.save_path = hparams.data_dir+"/logs/"+datetime_str+"/"+hparams.model_name
    hparams.model_dir = hparams.data_dir+"/"+hparams.model_dir+"/"+hparams.model_name
    
    model = ClassificationLightningModule(hparams=hparams)
    #model.load_from_checkpoint("/datahdd/raphael/datasets/ntu/ntu_no_border/cross_subject/logs/2020-02-15_11:20/efficientnet/lightning_logs/version_0/checkpoints/_ckpt_epoch_17.ckpt")
    
    # most basic trainer, uses good defaults
    trainer = pl.Trainer(min_epochs=hparams.min_epochs,
                 max_epochs=hparams.max_epochs,
                 default_save_path=hparams.save_path,
                 amp_level=hparams.amp_level, use_amp=hparams.use_amp,
                 gpus=hparams.gpus,
                 gradient_clip_val=hparams.gradient_clip_val,
                 checkpoint_callback=ModelCheckpoint(filepath=hparams.model_dir,
                     save_top_k=hparams.checkpoint_save_top_k,
                     mode=hparams.checkpoint_mode,
                     verbose=hparams.checkpoint_verbose,
                     monitor=hparams.checkpoint_monitor)
                 )
    
    trainer.fit(model)

if __name__ == "__main__":
    train_app()
