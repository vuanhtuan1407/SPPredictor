import json
import random
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import params


class DataUtils:
    def __init__(self):
        self.df = pd.read_csv('./aa_data/smiles_string_aa.csv')

    def get_index_of_short_abbreviation(self, short_abbreviation):
        index = self.df[self.df['short_abbreviations'] == short_abbreviation].index.tolist()
        if len(index) != 0:
            return index[0]
        else:
            return 22  # Unknown amino acid

    def get_smiles_string_aa(self, short_abbreviation):
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
            aa = self.df['short_abbreviations'][random.randint(0, 19)]
        index = self.get_index_of_short_abbreviation(short_abbreviation=aa)
        return self.df['smiles_convert'][index]

    def get_smiles_str_of_prot(self, prot: str):
        """
        *** Insert space between amino acid ***
        :param prot: Amino acid-based protein
        """
        smiles_prot = ''
        for aa in prot:
            smiles_prot = smiles_prot + self.get_smiles_string_aa(short_abbreviation=aa) + ' '

        return smiles_prot[:len(smiles_prot) - 1]

    def extract_raw_dataset_by_partition(self, raw_path: str | None = None,
                                         benchmark: bool = False):
        """
        Extract raw dataset by partition.

        :param benchmark: Define whether raw data will be used for benchmarking or not?
        :param raw_path: Path to raw data, defaults to `train_set.fasta`
        """
        raw_path = './sp_data/train_set.fasta' if raw_path is None else raw_path
        partitioned_prot = {}

        with open(raw_path) as fasta:
            data = fasta.readlines()
            for i in tqdm(range(0, len(data), 3), desc='Processing data'):
                annotation = data[i][1:len(data[i]) - 1].split("|")  # Ignore character `\n` at the end of line
                # Some documents use the word "life groups" instead of "kingdom"
                prot_id, kingdom, label, partition = annotation
                aa_seq = data[i + 1][:len(data[i + 1]) - 1]
                smiles = self.get_smiles_str_of_prot(prot=aa_seq)
                prot = {
                    "id": prot_id,
                    "kingdom": kingdom,
                    "label": label,
                    "aa_seq": aa_seq,
                    "smiles": smiles
                }
                if partition in partitioned_prot.keys():
                    partitioned_prot[partition].append(prot)
                else:
                    partitioned_prot[partition] = []

        if not benchmark:
            self.__create_smiles_training_json(partitioned_prot, split_rate=0.1)
        else:
            self.__create_smiles_benchmark_json(partitioned_prot)

    @staticmethod
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

    @staticmethod
    def __create_smiles_benchmark_json(partitioned_prot):
        for partition, data in partitioned_prot.items():
            with open(f"./sp_data/benchmark_partition_{partition}.json", "w") as file:
                json.dump(data, fp=file, ensure_ascii=False)

    # @staticmethod
    # def split_train_val_test(dataset: Dataset, train_size: float = 0.8, test_size: float = 0.2):
    #     if train_size + test_size > 1:
    #         raise Exception()
    #     elif train_size + test_size < 1:
    #         val_size = 1 - train_size - test_size
    #         train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size],
    #                                                     generator=Generator().manual_seed(42))
    #         # train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    #         return train_set, val_set, test_set
    #     else:
    #         train_set, test_set = random_split(dataset, [train_size, test_size],
    #                                            generator=Generator().manual_seed(42))
    #         # train_set, test_set = random_split(dataset, [train_size, test_size])
    #
    #         return train_set, test_set


if __name__ == "__main__":
    data_utils = DataUtils()
    data_utils.extract_raw_dataset_by_partition(raw_path=str(Path(params.ROOT_DIR) / params.TRAIN_PATH))
    data_utils.extract_raw_dataset_by_partition(raw_path=str(Path(params.ROOT_DIR) / params.BENCHMARK_PATH),
                                                benchmark=True)
