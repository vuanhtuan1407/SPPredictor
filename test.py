import argparse
import os.path
from typing import Optional

import lightning as L
import torch

import params
import utils as ut
from lightning_module.sp_data_module import SPDataModule
from lightning_module.sp_module import SPModule
from typing_ext import union_devices


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--checkpoint', type=Optional[str], default=None)
    arg_parser.add_argument('--env', type=str, default='kaggle')
    arg_parser.add_argument('--devices', type=union_devices, default='auto')
    # accelerator can be 'cpu', 'gpu', 'tpu' or use 'auto' instead of do not want to specify
    arg_parser.add_argument('--accelerator', type=str, default='auto')
    arg_parser.add_argument('--model_type', type=str, default='cnn')
    arg_parser.add_argument('--data_type', type=str, default='aa')
    arg_parser.add_argument('--conf_type', type=str, default='conf')
    arg_parser.add_argument('--epochs', type=int, default=3)
    arg_parser.add_argument('--num_workers', type=int, default=1)
    arg_parser.add_argument('--batch_size', type=int, default=8)

    return arg_parser.parse_args()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    # args = parse_arguments()
    checkpoint = ut.abspath(
        f'checkpoints/{params.MODEL_TYPE}_{params.DATA_TYPE}_epoch={params.EPOCHS}_{params.CONF_TYPE}_{params.ENV}.ckpt'
    )
    if checkpoint is not None:
        checkpoint = ut.abspath(
            f'checkpoints/{params.MODEL_TYPE}_{params.DATA_TYPE}_epoch={params.EPOCHS}_{params.CONF_TYPE}_{params.ENV}-v{params.CHECKPOINT_VER}.ckpt'
        )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("Path does not exist. Check checkpoint path again")

    sp_module = SPModule.load_from_checkpoint(checkpoint_path=checkpoint)
    sp_data_module = SPDataModule.load_from_checkpoint(checkpoint_path=checkpoint)

    trainer = L.Trainer(
        devices=params.DEVICES,
        accelerator=params.ACCELERATOR,
        logger=False
    )

    trainer.test(sp_module, datamodule=sp_data_module)
