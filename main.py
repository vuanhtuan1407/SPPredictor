from lightning.pytorch.cli import LightningCLI

from lightning_module.sp_data_module import SPDataModule
from lightning_module.sp_module import SPModule

if __name__ == '__main__':
    cli = LightningCLI(SPModule, SPDataModule)
