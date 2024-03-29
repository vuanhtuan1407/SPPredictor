import json
import random
import time
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import params
import utils as ut

AA_DATA = pd.read_csv(ut.get_absolute_path('data/aa_data/smiles_string_aa.csv'))  # Load data into DataFrame
UNIPROT_PATH = 'data/sp_data/uniprot_sprot.fasta'
SMILES_CORPUS_PATH = 'data/sp_data/uniprot_smiles.txt'


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
        smiles_prot = smiles_prot + get_smiles_string_aa(short_abbreviation=aa)

    return smiles_prot


def create_smiles_training_tokenizer():
    # Extract uniprot from aa sequence to smiles
    src_path = ut.get_absolute_path(UNIPROT_PATH)
    records = SeqIO.parse(src_path, 'fasta')
    smiles_prot = []
    for record in tqdm(records, total=len(records)):
        aa_seq = str(record.seq)
        smiles_prot.append(get_smiles_of_prot(aa_seq))

    with open(ut.get_absolute_path(SMILES_CORPUS_PATH), 'w') as f:
        f.writelines(smiles_prot)


def extract_raw_dataset_by_partition(raw_path: str | None = None, benchmark: bool = False):
    """
    Extract raw dataset by partition.

    :param benchmark: Define whether raw data will be used for benchmarking or not?
    :param raw_path: Path to raw data, defaults to `train_set.fasta`
    """
    raw_path = './sp_data/train_set.fasta' if raw_path is None else raw_path
    partitioned_prot = {}

    records = SeqIO.parse(raw_path, 'fasta')
    for record in records:
        prot_id, kingdom, label, partition = str(record.id).split('|')
        aa_seq = str(record.seq)[:len(record.seq) // 2]
        prot_info = {
            'prot_id': prot_id,
            'kingdom': kingdom,
            'label': label,
            'smiles': get_smiles_of_prot(aa_seq)
        }
        if partition in partitioned_prot.keys():
            partitioned_prot[partition].append(prot_info)
        else:
            partitioned_prot[partition] = []

    if not benchmark:
        __create_smiles_training_json(partitioned_prot, split_rate=0.1)
    else:
        __create_smiles_benchmark_json(partitioned_prot)


# *** PRIVATE FUNCTION *** #

def __create_smiles_training_json(partitioned_prot, split_rate: float = 0.1):
    """
    Each partition is divided into train and test sets with split rates is 0.9 and 0.1 as default
    :param partitioned_prot:
    :param split_rate: Percentage of dataset used for testing
    """
    # partitioned_prot = self.extract_raw_dataset_by_partition(create_files=False)
    for partition, data in partitioned_prot.items():
        train_set, test_set = train_test_split(data, test_size=split_rate, shuffle=True)
        with open(f"./sp_data/train_set_partition_{partition}.json", "w") as file:
            json.dump(train_set, fp=file, ensure_ascii=False)
        with open(f"./sp_data/test_set_partition_{partition}.json", "w") as file:
            json.dump(test_set, fp=file, ensure_ascii=False)


def __create_smiles_benchmark_json(partitioned_prot):
    for partition, data in partitioned_prot.items():
        with open(f"./sp_data/benchmark_partition_{partition}.json", "w") as file:
            json.dump(data, fp=file, ensure_ascii=False)


if __name__ == "__main__":
    extract_raw_dataset_by_partition()
    extract_raw_dataset_by_partition(raw_path=str(Path(params.ROOT_DIR) / params.BENCHMARK_PATH), benchmark=True)
