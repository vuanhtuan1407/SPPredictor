from pathlib import Path
from typing import Optional, Literal

import lightning as L
# from lightning.data.datasets.iterable import DataLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

import params
from data.sp_dataset import SPDataset
from params import ROOT_DIR


class SPDataModule(L.LightningDataModule):
    def __init__(self, dataset_type: Literal["aa", "smiles"] = "smiles"):
        super().__init__()
        self.sp_dataset = None
        self.test_set = None
        self.val_set = None
        self.train_set = None
        self.dataset_type = dataset_type

    # def prepare_data(self) -> None:
    #     dut.extract_raw_dataset_by_partition(raw_path=str(Path(ROOT_DIR) / params.TRAIN_PATH))
    #     dut.extract_raw_dataset_by_partition(raw_path=str(Path(ROOT_DIR) / params.BENCHMARK_PATH), benchmark=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # print(f"\nSetting up: Using {self.dataset_type}\n")
            train_path = [str(Path(ROOT_DIR) / 'data/sp_data/train_set_partition_0.json'),
                          str(Path(ROOT_DIR) / 'data/sp_data/train_set_partition_1.json')]
            val_path = [str(Path(ROOT_DIR) / 'data/sp_data/test_set_partition_0.json'),
                        str(Path(ROOT_DIR) / 'data/sp_data/test_set_partition_1.json')]
            self.train_set = SPDataset(json_paths=train_path, data_type=self.dataset_type)
            self.val_set = SPDataset(json_paths=val_path, data_type=self.dataset_type)
        elif stage == "test":
            test_path = [str(Path(ROOT_DIR) / 'data/sp_data/train_set_partition_2.json'),
                         str(Path(ROOT_DIR) / 'data/sp_data/test_set_partition_2.json')]
            self.test_set = SPDataset(json_paths=test_path, data_type=self.dataset_type)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=1,
                          persistent_workers=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=params.BATCH_SIZE, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=params.BATCH_SIZE, shuffle=False)


if __name__ == "__main__":
    spdata_module = SPDataModule(dataset_type='smiles')
    spdata_module.prepare_data()
