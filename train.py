import argparse
from typing import Union

import lightning as L
import torch
import wandb
# from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger

import params
import utils as ut
from callback_utils import model_checkpoint, early_stopping, tqdm_progress_bar
from lightning_module.sp_data_module import SPDataModule
from lightning_module.sp_module import SPModule


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--process', type=str, default='train', choices=['train', 'test', 'full'])
    arg_parser.add_argument('--env', type=str, default='local')
    arg_parser.add_argument('--log-dir', type=str, default='/logs')
    arg_parser.add_argument('--devices', type=Union[list[int], str, int], default='auto')
    # accelerator can be 'cpu', 'gpu', 'tpu' or use 'auto' instead of do not want to specify
    arg_parser.add_argument('--accelerator', type=str, default='auto')

    return arg_parser.parse_args()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    # CLI parsing arguments
    # args = parse_arguments()
    log_dir = params.KAGGLE_DIR if params.ENV == 'kaggle' else ut.abspath(params.LOG_DIR)
    # logger = TensorBoardLogger(save_dir=log_dir, name='tensorboard')
    logger = WandbLogger(save_dir=log_dir, name='wandb', project='SPPredictor')
    logger.experiment.config["batch_size"] = params.BATCH_SIZE

    sp_module = SPModule()
    sp_data_module = SPDataModule()
    trainer = L.Trainer(
        devices=params.DEVICES,
        accelerator=params.ACCELERATOR,
        max_epochs=params.EPOCHS,
        val_check_interval=1.0,
        # logger=logger,
        logger=False,
        enable_checkpointing=False,
        callbacks=[early_stopping, tqdm_progress_bar]
        # callbacks=[model_checkpoint, early_stopping, tqdm_progress_bar]
    )

    trainer.fit(sp_module, datamodule=sp_data_module)

    wandb.finish(quiet=True)
