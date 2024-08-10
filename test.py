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
    # accelerator can be 'cpu', 'gpu', 'tpu' or use 'auto' instead if you do not want to specify
    arg_parser.add_argument('--accelerator', type=str, default='auto')
    arg_parser.add_argument('--model_type', type=str, default='cnn')
    arg_parser.add_argument('--data_type', type=str, default='aa')
    arg_parser.add_argument('--conf_type', type=str, default='conf')
    arg_parser.add_argument('--epochs', type=int, default=3)
    arg_parser.add_argument('--num_workers', type=int, default=1)
    arg_parser.add_argument('--batch_size', type=int, default=8)

    return arg_parser.parse_args()


def map_location(storage, location):
    return storage


def test(checkpoint):
    torch.set_float32_matmul_precision('medium')

    # args = parse_arguments()
    checkpoint = ut.abspath(f'checkpoints/{checkpoint}')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("Path does not exist. Check checkpoint path again")

    sp_module = SPModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location=map_location)
    sp_data_module = SPDataModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location=map_location)

    trainer = L.Trainer(
        devices=params.DEVICES,
        accelerator=params.ACCELERATOR,
        logger=False,
        enable_checkpointing=False
    )

    trainer.test(sp_module, datamodule=sp_data_module)


if __name__ == '__main__':
    # torch.set_float32_matmul_precision('medium')
    #
    # # args = parse_arguments()
    # checkpoint = ut.abspath(f'checkpoints/{params.CHECKPOINT}')
    # if not os.path.exists(checkpoint):
    #     raise FileNotFoundError("Path does not exist. Check checkpoint path again")
    #
    # sp_module = SPModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location=map_location)
    # sp_data_module = SPDataModule.load_from_checkpoint(checkpoint_path=checkpoint, map_location=map_location)
    #
    # trainer = L.Trainer(
    #     devices=params.DEVICES,
    #     accelerator=params.ACCELERATOR,
    #     logger=False,
    #     enable_checkpointing=False
    # )
    #
    # trainer.test(sp_module, datamodule=sp_data_module)

    test(checkpoint=params.CHECKPOINT)

    # checkpoints = [
    #     "cnn-aa-default-0_epochs=100",
    #     # "transformer-aa-lite-0_epochs=100",
    #     # "transformer-aa-lite-1_epochs=100",
    #     "transformer-aa-default-0_epochs=100",
    #     "lstm-aa-default-0_epochs=100",
    #     "st_bilstm-aa-default-0_epochs=100",
    #     # "bert-aa-default-0_epochs=100",
    #     # "bert_pretrained-aa-default-0_epochs=100",
    #     # "bert_pretrained-aa-default-0_epochs=100_v1",
    #     "cnn_trans-aa-lite-0_epochs=100",
    #     "cnn-smiles-default-0_epochs=100",
    #     # "cnn_trans-smiles-lite-0_epochs=100",
    #     "gconv-graph-heavy-0_epochs=100",
    #     "gconv_trans-graph-default-0_epochs=100",
    #     "cnn-aa-default-1_epochs=100",
    #     "transformer-aa-default-1_epochs=100",
    #     "lstm-aa-lite-1_epochs=100",
    #     "st_bilstm-aa-lite-1_epochs=100",
    #     # "bert-aa-default-1_epochs=100",
    #     # "bert_pretrained-aa-default-1_epochs=100",
    #     # "bert_pretrained-aa-default-1_epochs=100_v1",
    #     "cnn_trans-aa-lite-1_epochs=100",
    #     "cnn-smiles-default-1_epochs=100",
    #     "transformer-smiles-lite-1_epochs=100",
    #     "cnn_trans-smiles-lite-1_epochs=100",
    #     "gconv-graph-heavy-1_epochs=100",
    #     "gconv_trans-graph-default-1_epochs=100",
    #
    # ]
    #
    # for checkpoint in checkpoints:
    #     full_checkpoint = checkpoint + '.ckpt'
    #     test(full_checkpoint)
