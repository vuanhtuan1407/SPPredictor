import json
import math
import random
import time
# from copy import deepcopy
from typing import List, Optional

import dgl
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Sampler, Dataset, RandomSampler, Subset
from tqdm import tqdm

import utils as ut

AA_DATA = pd.read_csv(ut.abspath('data/aa_data/smiles_string_aa.csv'))  # Load data_type into DataFrame
UNIPROT_PATH = ut.abspath('data/sp_data/uniprot_sprot.fasta')
SMILES_CORPUS_PATH = ut.abspath('data/sp_data/uniprot_smiles.txt')
SIGNALP6_PATH = ut.abspath('data/sp_data/train_set.fasta')
SIGNAL5_BENCHMARK_PATH = ut.abspath('data/sp_data/benchmark_set_sp5.fasta')
EUKARYAN_PATH = ut.abspath('data/sp_data/eukarya_dataset.fasta')
OTHERS_PATH = ut.abspath('data/sp_data/others_dataset.fasta')
PROT3D_PATH = ut.abspath('data/sp_data/train_set_graph.json')


# *** PUBLIC FUNCTION *** #

def get_index_of_short_abbreviation(short_abbreviation):
    index = AA_DATA[AA_DATA['short_abbreviations'] == short_abbreviation].index.tolist()
    if len(index) != 0:
        return index[0]
    else:
        return 22  # Unknown amino acid


def get_smiles_string_aa(short_abbreviation):
    seeder = time.time_ns()
    random.seed(seeder)
    aa = short_abbreviation
    if short_abbreviation == 'B':
        aa = random.choice(['N', 'D'])
    elif short_abbreviation == 'Z':
        aa = random.choice(['Q', 'E'])
    elif short_abbreviation == 'J':
        aa = random.choice(['I', 'L'])
    elif short_abbreviation == 'X':
        aa = AA_DATA['short_abbreviations'][random.randint(0, 19)]
    index = get_index_of_short_abbreviation(short_abbreviation=aa)
    return AA_DATA['smiles_convert'][index]


def get_smiles_of_prot(prot: str):
    """
    *** Do not insert space between amino acid ***
    :param prot: Amino acid-based protein
    """
    smiles_prot = ''
    for aa in prot:
        # Need remove element Oxygen at the end of sequence to make residue
        smiles_prot = smiles_prot[:-1] + get_smiles_string_aa(short_abbreviation=aa)
    return smiles_prot


def create_smiles_training_tokenizer():
    # Extract uniprot from aa sequence to smiles
    src_path = UNIPROT_PATH
    records = SeqIO.parse(src_path, 'fasta')
    smiles_prot = []
    for record in tqdm(records):
        aa_seq = str(record.seq)
        if 'U' not in aa_seq and 'O' not in aa_seq:  # ignore proteins having 2 special amino acids U and O
            smiles_prot.append(get_smiles_of_prot(aa_seq))

    with open(SMILES_CORPUS_PATH, 'w') as f:
        f.writelines('\n'.join(smiles_prot))


def extract_raw_dataset_by_partition(raw_path: str | None = None, benchmark: bool = False, include_smiles: bool = True,
                                     organism=None):
    """
    Extract raw dataset by partition.

    :param organism:
    :param include_smiles: Include smiles convert in the dataset
    :param benchmark: Define whether raw data_type will be used for benchmarking or not?
    :param raw_path: Path to raw data_type, defaults to `train_set.fasta`
    """
    raw_path = SIGNALP6_PATH if raw_path is None else ut.abspath(raw_path)
    partitioned_prot = {}

    records = SeqIO.parse(raw_path, 'fasta')
    for record in tqdm(records):
        prot_id, kingdom, label, partition = str(record.id).split('|')
        aa_seq = str(record.seq)[:len(record.seq) // 2]
        prot_info = {
            'prot_id': prot_id,
            'organism': kingdom,
            'label': label,
            "aa_seq": aa_seq
        }
        if include_smiles:
            prot_info['smiles'] = get_smiles_of_prot(aa_seq)

        if partition in partitioned_prot.keys():
            partitioned_prot[partition].append(prot_info)
        else:
            partitioned_prot[partition] = []

    if not benchmark:
        _create_smiles_training_json(partitioned_prot, split_rate=0.1, organism=organism)
    else:
        _create_smiles_benchmark_json(partitioned_prot, organism)


def extract_3d_dataset_by_partition(dataset_path: str | None = None, benchmark: bool = False, organism=None):
    dataset_path = PROT3D_PATH if dataset_path is None else ut.abspath(dataset_path)
    partitioned_prot = {}
    with open(dataset_path, 'r') as f:
        records = json.load(f)
    for record in tqdm(records):
        partition = record['partition']
        if partition in partitioned_prot.keys():
            partitioned_prot[partition].append(record)
        else:
            partitioned_prot[partition] = []

    if not benchmark:
        _create_graph_training_json(partitioned_prot, split_rate=0.1, organism=organism)
    else:
        _create_graph_benchmark_json(partitioned_prot, organism)


# *** PRIVATE FUNCTION *** #

def _create_smiles_training_json(partitioned_prot, split_rate: float = 0.1, organism: str | None = None):
    """
    Each partition is divided into train and test sets with split rates is 0.9 and 0.1 as default
    :param partitioned_prot:
    :param split_rate: Percentage of dataset used for testing
    """

    if organism is None:
        for partition, data in partitioned_prot.items():
            train_set, test_set = train_test_split(data, test_size=split_rate, shuffle=True)
            with open(ut.abspath(f"data/sp_data/train_set_partition_{partition}.json"), "w") as file:
                json.dump(train_set, fp=file, ensure_ascii=False)
            with open(ut.abspath(f"data/sp_data/test_set_partition_{partition}.json"), "w") as file:
                json.dump(test_set, fp=file, ensure_ascii=False)
    else:
        for partition, data in partitioned_prot.items():
            train_set, test_set = train_test_split(data, test_size=split_rate, shuffle=True)
            with open(ut.abspath(f"data/sp_data/train_set_partition_{partition}_{organism}.json"), "w") as file:
                json.dump(train_set, fp=file, ensure_ascii=False)
            with open(ut.abspath(f"data/sp_data/test_set_partition_{partition}_{organism}.json"), "w") as file:
                json.dump(test_set, fp=file, ensure_ascii=False)


def _create_smiles_benchmark_json(partitioned_prot, organism: str | None = None):
    if organism is None:
        for partition, data in partitioned_prot.items():
            with open(ut.abspath(f"data/sp_data/benchmark_partition_{partition}.json"), "w") as file:
                json.dump(data, fp=file, ensure_ascii=False)
    else:
        for partition, data in partitioned_prot.items():
            with open(ut.abspath(f"data/sp_data/benchmark_partition_{partition}_{organism}.json"), "w") as file:
                json.dump(data, fp=file, ensure_ascii=False)


def _create_graph_training_json(partitioned_prot, split_rate: float = 0.1, organism: str | None = None):
    """
    Each partition is divided into train and test sets with split rates is 0.9 and 0.1 as default
    :param partitioned_prot:
    :param split_rate: Percentage of dataset used for testing
    """

    if organism is None:
        for partition, data in partitioned_prot.items():
            train_set, test_set = train_test_split(data, test_size=split_rate, shuffle=True)
            with open(ut.abspath(f"data/sp_data/train_set_graph_partition_{partition}.json"), "w") as file:
                json.dump(train_set, fp=file, ensure_ascii=False)
            with open(ut.abspath(f"data/sp_data/test_set_graph_partition_{partition}.json"), "w") as file:
                json.dump(test_set, fp=file, ensure_ascii=False)
    else:
        for partition, data in partitioned_prot.items():
            train_set, test_set = train_test_split(data, test_size=split_rate, shuffle=True)
            with open(ut.abspath(f"data/sp_data/train_set_graph_partition_{partition}_{organism}.json"), "w") as f:
                json.dump(train_set, fp=f, ensure_ascii=False)
            with open(ut.abspath(f"data/sp_data/test_set_graph_partition_{partition}_{organism}.json"), "w") as f:
                json.dump(test_set, fp=f, ensure_ascii=False)


def _create_graph_benchmark_json(partitioned_prot, organism: str | None = None):
    if organism is None:
        for partition, data in partitioned_prot.items():
            with open(ut.abspath(f"data/sp_data/benchmark_graph_partition_{partition}.json"), "w") as file:
                json.dump(data, fp=file, ensure_ascii=False)
    else:
        for partition, data in partitioned_prot.items():
            with open(ut.abspath(f"data/sp_data/benchmark_graph_partition_{partition}_{organism}.json"), "w") as f:
                json.dump(data, fp=f, ensure_ascii=False)


def _split_dataset_by_organism(file=None):
    # TODO: split SignalP6.0 dataset file into 2 parts: EUKARYA and OTHERS
    if file is None:
        file = SIGNALP6_PATH
    eukarya = []
    others = []
    records = SeqIO.parse(file, format='fasta')
    for record in records:
        organism = record.id.split('|')[1]
        if organism == "EUKARYA":
            eukarya.append(record)
        else:
            others.append(record)
    SeqIO.write(eukarya, ut.abspath(f"data/sp_data/eukarya_dataset.fasta"), format='fasta')
    SeqIO.write(others, ut.abspath(f"data/sp_data/others_dataset.fasta"), format='fasta')


class SPDataLoader(DataLoader):
    def __init__(
            self,
            dataset,
            shuffle=False,
            use_workers_init_fn=False,
            use_sp_sampler=False,
            use_graph_collate_fn=False,
            current_epoch=0,
            batch_size=1,
            num_workers=0,
            pin_memory=False
    ):
        self.dataset = dataset
        self.current_epoch = current_epoch
        self.batch_size = batch_size
        persistent_workers = False
        if num_workers > 0:
            persistent_workers = True
        worker_init_fn = None
        if use_workers_init_fn:
            worker_init_fn = self.worker_init_fn
        collate_fn = None
        if use_graph_collate_fn:
            collate_fn = SPDataLoader.graph_collate_fn
        if shuffle and use_sp_sampler:
            # warnings.warn("Do not set `shuffle` while using `use_sp_sampler`. Automatically set `shuffle=True`.")
            sp_sampler = SPBatchRandomSampler(dataset, batch_size, current_epoch, shuffle=True)
            super().__init__(
                dataset=dataset,
                batch_sampler=sp_sampler,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                worker_init_fn=worker_init_fn,
                collate_fn=collate_fn,
                pin_memory=pin_memory
            )
        else:
            super().__init__(
                dataset=dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                worker_init_fn=worker_init_fn,
                collate_fn=collate_fn,
                pin_memory=pin_memory
            )

    def worker_init_fn(self, worker_id):
        seed = worker_id + self.current_epoch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def graph_collate_fn(batch):
        graphs, lbs, organisms = map(list, zip(*batch))
        g_feats = dgl.batch(graphs)
        lbs = torch.stack(lbs)
        organisms = torch.stack(organisms)
        return g_feats, lbs, organisms


class SPBatchRandomSampler(Sampler[List[int]]):
    def __init__(self, dataset: Dataset, batch_size: int, current_epoch: int, valid_indices=None, shuffle=False,
                 replacement: bool = False, num_samples: Optional[int] = None, generator=None, drop_last=False):
        super(Sampler, self).__init__()
        if num_samples is None:
            num_samples = len(dataset)
        self.num_samples = num_samples
        self.dataset = dataset  # dataset must implement __len__ method
        if valid_indices is None:
            valid_indices = range(len(dataset))
        self.valid_indices = valid_indices
        data_source = Subset(dataset, valid_indices)
        self.batch_size = batch_size
        self.drop_last = drop_last
        if shuffle and generator is None:
            torch.manual_seed(current_epoch)
            torch.cuda.manual_seed(current_epoch)
        self.standard_sampler = RandomSampler(data_source=data_source, replacement=replacement,
                                              num_samples=num_samples, generator=generator)

    def __iter__(self):
        batch = []
        for idx in self.standard_sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)


if __name__ == "__main__":
    # extract_raw_dataset_by_partition()
    # extract_raw_dataset_by_partition(raw_path=ut.abspath(params.BENCHMARK_PATH), benchmark=True)
    # _split_dataset_by_organism()
    # extract_raw_dataset_by_partition(raw_path=EUKARYAN_PATH, organism='eukarya')
    # extract_raw_dataset_by_partition(raw_path=OTHERS_PATH, organism='others')
    extract_3d_dataset_by_partition()
