from typing import Optional

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

import data.data_utils as dut
import params
import utils as ut
from data.sp_dataset import SPDataset


class SPDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.test_set = None
        self.val_set = None
        self.train_set = None
        self.batch_size = params.BATCH_SIZE

    def prepare_data(self) -> None:
        dut.extract_raw_dataset_by_partition(raw_path=ut.abspath(params.TRAIN_PATH))
        dut.extract_raw_dataset_by_partition(raw_path=ut.abspath(params.BENCHMARK_PATH), benchmark=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # print(f"\nSetting up: Using {self.dtype}\n")
            train_path = ['data/sp_data/train_set_partition_0.json', 'data/sp_data/train_set_partition_1.json']
            val_path = ['data/sp_data/test_set_partition_0.json', 'data/sp_data/test_set_partition_1.json']
            self.train_set = SPDataset(json_paths=ut.abspaths(train_path), dtype=params.DATA)
            self.val_set = SPDataset(json_paths=ut.abspaths(val_path), dtype=params.DATA)
        elif stage == "test":
            test_path = ['data/sp_data/train_set_partition_2.json', 'data/sp_data/test_set_partition_2.json']
            self.test_set = SPDataset(json_paths=ut.abspaths(test_path), dtype=params.DATA)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=1, persistent_workers=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    spdata_module = SPDataModule()
    spdata_module.prepare_data()
