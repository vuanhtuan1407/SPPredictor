import argparse

import lightning as L
import torch
import wandb
# from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger

import params
from lightning_module.sp_data_module import SPDataModule
from lightning_module.sp_module import SPModule
from typing_ext import union_devices


# import platform


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--env', type=str, default='local')
    arg_parser.add_argument('--log_dir', type=str, default='logs')
    arg_parser.add_argument('--devices', type=union_devices, default='auto')
    # accelerator can be 'cpu', 'gpu', 'tpu' or use 'auto' instead of do not want to specify
    arg_parser.add_argument('--accelerator', type=str, default='auto')
    arg_parser.add_argument('--lr', type=float, default=1e-6)
    arg_parser.add_argument('--epochs', type=int, default=3)
    arg_parser.add_argument('--batch_size', type=int, default=8)
    arg_parser.add_argument('--model_type', type=str, default='cnn')
    arg_parser.add_argument('--data_type', type=str, default='aa')
    arg_parser.add_argument('--conf_type', type=str, default='default')
    arg_parser.add_argument('--num_workers', type=int, default=1)

    return arg_parser.parse_args()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    # # CLI parsing arguments
    # args = parse_arguments()
    logger = WandbLogger(save_dir=params.LOG_DIR, project='SPPredictor')
    if params.USE_ORGANISM:
        logger.experiment.name = f'{params.MODEL_TYPE}_{params.DATA_TYPE}_use_organism'
    else:
        logger.experiment.name = f'{params.MODEL_TYPE}_{params.DATA_TYPE}'
    logger.experiment.config['batch_size'] = params.BATCH_SIZE

    sp_module = SPModule(
        model_type=params.MODEL_TYPE,
        data_type=params.DATA_TYPE,
        conf_type=params.CONF_TYPE,
        use_organism=params.USE_ORGANISM,
        batch_size=params.BATCH_SIZE,
        lr=params.LEARNING_RATE,
    )

    sp_data_module = SPDataModule(
        data_type=params.DATA_TYPE,
        batch_size=params.BATCH_SIZE,
        num_workers=params.NUM_WORKERS,
    )

    trainer = L.Trainer(
        devices=params.DEVICES,
        accelerator=params.ACCELERATOR,
        max_epochs=params.EPOCHS,
        logger=False,
        enable_checkpointing=False,
        val_check_interval=1.0,
        # callbacks=[model_checkpoint, early_stopping],
    )

    trainer.fit(sp_module, datamodule=sp_data_module)

    wandb.finish(quiet=True)
