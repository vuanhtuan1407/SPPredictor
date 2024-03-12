import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

import params
from callback_utils import model_checkpoint, early_stopping
from lightning_module.sp_data_module import SPDataModule
from lightning_module.sp_module import SPModule


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--env', type=str, default='local')

    return arg_parser.parse_args()


if __name__ == '__main__':
    # CLI arguments
    args = parse_arguments()
    logger = TensorBoardLogger(save_dir=str(Path(params.ROOT_DIR) / params.DEFAULT_LOG_DIR), name='tensorboard_logs')
    if args.env == 'kaggle':
        logger = False

    # Training and validation
    sp_module = SPModule(model_type=params.MODEL)
    sp_data_module = SPDataModule(dataset_type=params.DATA)
    trainer = L.Trainer(
        devices='auto',
        accelerator='auto',
        max_epochs=params.EPOCHS,
        val_check_interval=1.0,
        logger=logger,
        callbacks=[model_checkpoint, early_stopping]
    )

    trainer.fit(sp_module, datamodule=sp_data_module)
