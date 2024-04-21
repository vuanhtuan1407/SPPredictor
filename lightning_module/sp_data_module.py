from typing import Optional

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

import params
import utils as ut
from data.sp_dataset import SPDataset


class SPDataModule(L.LightningDataModule):
    def __init__(self, data_type: str, batch_size: int, num_workers: int):
        super().__init__()
        self.test_set = None
        self.val_set = None
        self.train_set = None

        self.data_type = data_type
        self.batch_size = batch_size
        self.num_workers = num_workers

    # def prepare_data(self) -> None:
    #     dut.extract_raw_dataset_by_partition(raw_path=ut.abspath(params.TRAIN_PATH))
    #     dut.extract_raw_dataset_by_partition(raw_path=ut.abspath(params.BENCHMARK_PATH), benchmark=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # print(f"\nSetting up: Using {self.data_type}\n")
            # train_paths = [f'data/sp_data/train_set_partition_0_{params.ORGANISM}.json',
            #                f'data/sp_data/train_set_partition_1_{params.ORGANISM}.json']
            # val_paths = [f'data/sp_data/test_set_partition_0_{params.ORGANISM}.json',
            #              f'data/sp_data/test_set_partition_1_{params.ORGANISM}.json']
            train_paths = [f'data/sp_data/train_set_partition_0.json', f'data/sp_data/train_set_partition_1.json']
            val_paths = [f'data/sp_data/test_set_partition_0.json', f'data/sp_data/test_set_partition_1.json']
            self.train_set = SPDataset(json_paths=ut.abspaths(train_paths), data_type=self.data_type)
            self.val_set = SPDataset(json_paths=ut.abspaths(val_paths), data_type=self.data_type)
        elif stage == "test":
            # test_paths = [f'data_type/sp_data/train_set_partition_2_{params.ORGANISM}.json',
            #               f'data_type/sp_data/test_set_partition_2_{params.ORGANISM}.json']
            test_paths = [f'data/sp_data/train_set_partition_2.json', f'data/sp_data/test_set_partition_2.json']
            self.test_set = SPDataset(json_paths=ut.abspaths(test_paths), data_type=self.data_type)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    spdata_module = SPDataModule(data_type=params.DATA, batch_size=params.BATCH_SIZE, num_workers=params.NUM_WORKERS)
    spdata_module.prepare_data()
