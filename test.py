import os.path
from pathlib import Path

import lightning as L

import params
import utils as ut
from lightning_module.sp_data_module import SPDataModule
from lightning_module.sp_module import SPModule

if __name__ == '__main__':
    checkpoint = str(
        Path(ut.ROOT_DIR) / f'checkpoints/{params.MODEL}_{params.DATA}_epoch={params.EPOCHS}_{params.ENV}.ckpt')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("Path does not exist. Check args again")
    sp_module = SPModule.load_from_checkpoint(checkpoint=checkpoint)
    sp_data_module = SPDataModule()

    trainer = L.Trainer(
        devices=params.DEVICES,
        accelerator=params.ACCELERATOR,
        max_epochs=params.EPOCHS,
        val_check_interval=1.0,
        logger=False
    )

    trainer.test(sp_module, datamodule=sp_data_module)
