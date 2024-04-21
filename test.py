import argparse
import os.path
from typing import Optional

import lightning as L
import torch

import utils as ut
from lightning_module.sp_data_module import SPDataModule
from lightning_module.sp_module import SPModule
from typing_ext import union_devices


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--checkpoint', type=Optional[str], default=None)
    arg_parser.add_argument('--devices', type=union_devices, default='auto')
    # accelerator can be 'cpu', 'gpu', 'tpu' or use 'auto' instead of do not want to specify
    arg_parser.add_argument('--accelerator', type=str, default='auto')
    arg_parser.add_argument('--data_type', type=str, default='aa')
    arg_parser.add_argument('--num_workers', type=int, default=1)
    arg_parser.add_argument('--batch_size', type=int, default=8)

    return arg_parser.parse_args()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parse_arguments()
    if args.checkpoint is None:
        raise ValueError('Please specify a checkpoint path')
    checkpoint = ut.abspath(f'checkpoints/{args.checkpoint}')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("Path does not exist. Check checkpoint path again")
    sp_module = SPModule.load_from_checkpoint(checkpoint_path=checkpoint)
    sp_data_module = SPDataModule(
        data_type=args.data_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = L.Trainer(
        devices=args.devices,
        accelerator=args.accelerator,
        logger=False
    )

    trainer.test(sp_module, datamodule=sp_data_module)
