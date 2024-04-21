from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

import params


class SPDataset(Dataset):
    def __init__(self, json_paths: Optional[list[str]], data_type: str):
        if json_paths is None or isinstance(json_paths, str):
            raise ValueError('provide path to dataset in list of str')
        df = pd.concat(pd.read_json(path) for path in json_paths)
        self.smiles = df['smiles'].tolist()
        self.aa_seq = df['aa_seq'].tolist()
        self.labels = df['label'].tolist()
        self.organisms = df['organism'].tolist()
        self.data_type = data_type

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        organism = self.organisms[index]
        seq = self.aa_seq[index] if self.data_type == 'aa' else self.smiles[index]
        label = torch.zeros(len(params.SP_LABELS), dtype=torch.int64)
        label[params.SP_LABELS[self.labels[index]]] = 1
        return seq, label, organism
