import json
import random
import time

import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import utils as ut

AA_DATA = pd.read_csv(ut.abspath('data/aa_data/smiles_string_aa.csv'))  # Load data_type into DataFrame
UNIPROT_PATH = ut.abspath('data/sp_data/uniprot_sprot.fasta')
SMILES_CORPUS_PATH = ut.abspath('data/sp_data/uniprot_smiles.txt')
SIGNALP6_PATH = ut.abspath('data/sp_data/train_set.fasta')
EUKARYAN_PATH = ut.abspath('data/sp_data/eukarya_dataset.fasta')
OTHERS_PATH = ut.abspath('data/sp_data/others_dataset.fasta')


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
    raw_path = SIGNALP6_PATH if raw_path is None else raw_path
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
            with open(ut.abspath(f"data_type/sp_data/train_set_partition_{partition}.json"), "w") as file:
                json.dump(train_set, fp=file, ensure_ascii=False)
            with open(ut.abspath(f"data_type/sp_data/test_set_partition_{partition}.json"), "w") as file:
                json.dump(test_set, fp=file, ensure_ascii=False)
    else:
        for partition, data in partitioned_prot.items():
            train_set, test_set = train_test_split(data, test_size=split_rate, shuffle=True)
            with open(ut.abspath(f"data_type/sp_data/train_set_partition_{partition}_{organism}.json"), "w") as file:
                json.dump(train_set, fp=file, ensure_ascii=False)
            with open(ut.abspath(f"data_type/sp_data/test_set_partition_{partition}_{organism}.json"), "w") as file:
                json.dump(test_set, fp=file, ensure_ascii=False)


def _create_smiles_benchmark_json(partitioned_prot, organism: str | None = None):
    if organism is None:
        for partition, data in partitioned_prot.items():
            with open(ut.abspath(f"data_type/sp_data/benchmark_partition_{partition}.json"), "w") as file:
                json.dump(data, fp=file, ensure_ascii=False)
    else:
        for partition, data in partitioned_prot.items():
            with open(ut.abspath(f"data_type/sp_data/benchmark_partition_{partition}_{organism}.json"), "w") as file:
                json.dump(data, fp=file, ensure_ascii=False)


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
    SeqIO.write(eukarya, ut.abspath(f"data_type/sp_data/eukarya_dataset.fasta"), format='fasta')
    SeqIO.write(others, ut.abspath(f"data_type/sp_data/others_dataset.fasta"), format='fasta')


if __name__ == "__main__":
    # extract_raw_dataset_by_partition()
    # extract_raw_dataset_by_partition(raw_path=ut.abspath(params.BENCHMARK_PATH), benchmark=True)
    # _split_dataset_by_organism()
    extract_raw_dataset_by_partition(raw_path=EUKARYAN_PATH, organism='eukarya')
    extract_raw_dataset_by_partition(raw_path=OTHERS_PATH, organism='others')
