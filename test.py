import os.path

import lightning as L
import torch

import params
import utils as ut
from lightning_module.sp_data_module import SPDataModule
from lightning_module.sp_module import SPModule

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    checkpoint = ut.abspath(
        f'checkpoints/{params.MODEL}_{params.DATA}_epoch={params.EPOCHS}_{params.CONF_TYPE}_{params.ENV}-v1.ckpt'
    )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("Path does not exist. Check args again")
    sp_module = SPModule.load_from_checkpoint(checkpoint_path=checkpoint)
    sp_data_module = SPDataModule()

    trainer = L.Trainer(
        devices=params.DEVICES,
        accelerator=params.ACCELERATOR,
        max_epochs=params.EPOCHS,
        val_check_interval=1.0,
        logger=False
    )

    trainer.test(sp_module, datamodule=sp_data_module)
