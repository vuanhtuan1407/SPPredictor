from pathlib import Path

import lightning as L

import params
from lightning_module.sp_data_module import SPDataModule
from lightning_module.sp_module import SPModule

if __name__ == '__main__':
    sp_data_module = SPDataModule(dataset_type=params.DATA)
    sp_module = SPModule.load_from_checkpoint(
        str(Path(params.ROOT_DIR) / f'checkpoints/{params.MODEL}_epoch=3_kaggle.ckpt'))
    trainer = L.Trainer(
        devices='auto',
        accelerator='auto',
        max_epochs=params.EPOCHS,
        val_check_interval=1.0,
        logger=False
    )

    trainer.test(sp_module, datamodule=sp_data_module)
