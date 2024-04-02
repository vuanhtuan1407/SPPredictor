import json
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast, BertTokenizer

import params
import utils as ut


class SPDataset(Dataset):
    def __init__(self, json_paths: list[str], dtype: Literal['aa', 'smiles'] = 'smiles'):
        df = pd.concat(pd.read_json(path) for path in json_paths)
        self.smiles = df['smiles'].tolist()
        self.aa_seq = df['aa_seq'].tolist()
        self.labels = df['label'].tolist()
        self.kingdoms = df['kingdom'].tolist()
        self.dtype = dtype
        self._config = self.__load_data_config(dtype=dtype)
        tokenizer_path = ut.abspath(self._config['tokenizer_path'])
        tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer = tokenizer
        if params.MODEL == 'bert_pretrained':
            self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        kingdom = self.kingdoms[index]
        seq = self.smiles[index] if self.dtype == 'smiles' else self.aa_seq[index]
        label = torch.zeros(len(params.SP_LABELS), dtype=torch.int64)
        label[params.SP_LABELS[self.labels[index]]] = 1
        input_ids = self.tokenizer.encode(
            seq,
            max_length=self._config['max_len'],
            padding="max_length",
            truncation=True
        )
        return torch.tensor(input_ids, dtype=torch.int64), label, kingdom

    @staticmethod
    def __load_data_config(dtype: str):
        with open(ut.abspath(f'configs/data_config/{dtype}_config.json')) as f:
            config = json.load(f)
            return config
