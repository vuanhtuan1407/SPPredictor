import json
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast, BertTokenizer

import params


class SPDataset(Dataset):
    def __init__(self, json_paths: list[str], data_type: Literal['aa', 'smiles'] = 'smiles'):
        df = pd.concat(pd.read_json(path) for path in json_paths)
        self.smiles = df['smiles'].tolist()
        self.aa_seq = df['aa_seq'].tolist()
        self.labels = df['label'].tolist()
        self.kingdoms = df['kingdom'].tolist()
        self.data_type = data_type
        self.config = self.__load_data_config(data_type=data_type)
        tokenizer_path = str(Path(params.ROOT_DIR) / self.config['tokenizer_path'])
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
        seq = self.smiles[index] if self.data_type == 'smiles' else self.aa_seq[index]
        label = torch.zeros(len(params.SP_LABELS), dtype=torch.int64)
        label[params.SP_LABELS[self.labels[index]]] = 1
        input_ids = self.tokenizer.encode(seq, max_length=self.config['max_len'], padding="max_length", truncation=True)
        return torch.tensor(input_ids, dtype=torch.int64), label, kingdom

    @staticmethod
    def __load_data_config(data_type: str):
        with open(str(Path(params.ROOT_DIR) / f'configs/data_config/{data_type}_config.json')) as f:
            config = json.load(f)
            return config
