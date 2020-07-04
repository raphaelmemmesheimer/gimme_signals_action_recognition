import os

from classification_module import ClassificationLightningModule
import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import Namespace

import yaml
import time

@hydra.main(config_path="config/config.yaml")
def train_app(cfg):
    print(cfg.pretty())
    hparams = Namespace(**cfg)

    datetime_str = time.strftime("%Y-%m-%d_%H:%M")
    hparams.data_dir = os.environ["DATASET_FOLDER"] +"/"+ hparams.data_dir
    hparams.save_path = hparams.data_dir+"/logs/"+datetime_str+"/"+hparams.model_name
    hparams.model_dir = hparams.data_dir+"/"+hparams.model_dir+"/"+hparams.model_name
    
    model = ClassificationLightningModule(hparams=hparams)

    trainer = pl.Trainer(min_epochs=hparams.min_epochs,
                 max_epochs=hparams.max_epochs,
                 default_root_dir=hparams.save_path,
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
