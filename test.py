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
def test_app(cfg):
    print(cfg.pretty())
    hparams = Namespace(**cfg)

    model = ClassificationLightningModule.load_from_checkpoint(hparams.model_dir)
    trainer = Trainer()
    trainer.test(model)

if __name__ == "__main__":
    test_app()
