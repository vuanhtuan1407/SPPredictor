from typing import Optional, Dict, Any

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import data.data_utils as dut
import params
import utils as ut
from data.data_utils import SPDataLoader
from data.sp_dataset import SPDataset


class SPDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_type: str,
            batch_size: int = 8,
            num_workers: int = 1,
            use_prepare_data: bool = False,
            use_split_dataset: bool = False
    ):
        super().__init__()
        self.current_training_epoch = 0
        self.save_hyperparameters()

        self.test_set = None
        self.val_set = None
        self.train_set = None

        self.data_type = data_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_prepare_data = use_prepare_data
        self.use_split_dataset = use_split_dataset
        self.persistent_workers = False
        if num_workers > 0:
            self.persistent_workers = True
        self.use_graph_collate_fn = False
        if self.data_type == "graph":
            self.use_graph_collate_fn = True

    def prepare_data(self) -> None:
        if self.use_prepare_data and self.data_type == 'graph':
            dut.extract_3d_dataset_by_partition()
        elif self.use_prepare_data and self.data_type != 'graph':
            dut.extract_raw_dataset_by_partition(raw_path=ut.abspath(params.TRAIN_PATH))
            dut.extract_raw_dataset_by_partition(raw_path=ut.abspath(params.BENCHMARK_PATH), benchmark=True)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {
            'current_training_epoch': self.trainer.current_epoch
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.current_training_epoch = state_dict['current_training_epoch']

    def setup(self, stage: Optional[str] = None) -> None:
        train_paths, val_paths, test_paths = [[], [], []]
        if stage == "fit" or stage is None:
            if self.data_type == 'graph' and not self.use_split_dataset:
                train_paths = [f'data/sp_data/train_set_graph_partition_0.json',
                               f'data/sp_data/train_set_graph_partition_1.json']
                val_paths = [f'data/sp_data/test_set_graph_partition_0.json',
                             f'data/sp_data/test_set_graph_partition_1.json']

            elif self.data_type == 'graph' and self.use_split_dataset:
                raise NotImplemented('Do not have split graph dataset!')

            elif self.data_type != "graph" and not self.use_split_dataset:
                train_paths = [f'data/sp_data/train_set_partition_0.json', f'data/sp_data/train_set_partition_1.json']
                val_paths = [f'data/sp_data/test_set_partition_0.json', f'data/sp_data/test_set_partition_1.json']

            elif self.data_type != 'graph' and self.use_split_dataset:
                train_paths = [f'data_type/sp_data/train_set_partition_0_{params.ON_ORGANISM}.json',
                               f'data_type/sp_data/train_set_partition_1_{params.ORGANISMS}.json']
                val_paths = [f'data_type/sp_data/test_set_partition_0_{params.ON_ORGANISM}.json',
                             f'data_type/sp_data/test_set_partition_1_{params.ORGANISMS}.json']

            self.train_set = SPDataset(json_paths=ut.abspaths(train_paths), data_type=self.data_type)
            self.val_set = SPDataset(json_paths=ut.abspaths(val_paths), data_type=self.data_type)
        elif stage == "test":
            if self.data_type == 'graph' and not self.use_split_dataset:
                test_paths = [f'data/sp_data/train_set_graph_partition_2.json',
                              f'data/sp_data/test_set_graph_partition_2.json']

            elif self.data_type == 'graph' and self.use_split_dataset:
                raise NotImplemented('Do not have split graph dataset!')

            elif self.data_type != "graph" and not self.use_split_dataset:
                test_paths = [f'data/sp_data/train_set_partition_2.json', f'data/sp_data/test_set_partition_2.json']

            elif self.data_type != 'graph' and self.use_split_dataset:
                test_paths = [f'data_type/sp_data/train_set_partition_2_{params.ON_ORGANISM}.json',
                              f'data_type/sp_data/test_set_partition_2_{params.ON_ORGANISM}.json']

            self.test_set = SPDataset(json_paths=ut.abspaths(test_paths), data_type=self.data_type)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return SPDataLoader(self.train_set, current_epoch=self.trainer.current_epoch, shuffle=True, use_sp_sampler=True,
                            use_workers_init_fn=False, use_graph_collate_fn=self.use_graph_collate_fn,
                            batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return SPDataLoader(self.val_set, current_epoch=self.trainer.current_epoch, shuffle=False, use_sp_sampler=False,
                            use_workers_init_fn=False, use_graph_collate_fn=self.use_graph_collate_fn,
                            batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return SPDataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, use_workers_init_fn=False,
                            use_sp_sampler=False, use_graph_collate_fn=self.use_graph_collate_fn)


if __name__ == "__main__":
    spdata_module = SPDataModule(data_type=params.DATA_TYPE, batch_size=params.BATCH_SIZE,
                                 num_workers=params.NUM_WORKERS)
    # spdata_module.prepare_data()
    spdata_module.setup()
    train_dataloader = spdata_module.train_dataloader()
    for batch in train_dataloader:
        print(batch)
