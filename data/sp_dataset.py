from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

import params


class SPDataset(Dataset):
    def __init__(self, json_paths: Optional[list[str]], dtype: str):
        if json_paths is None or isinstance(json_paths, str):
            raise ValueError('provide path to dataset in list of str')
        df = pd.concat(pd.read_json(path) for path in json_paths)
        self.smiles = df['smiles'].tolist()
        self.aa_seq = df['aa_seq'].tolist()
        self.labels = df['label'].tolist()
        self.kingdoms = df['kingdom'].tolist()
        self.dtype = dtype
        # tokenizer_path = ut.abspath(self._config['tokenizer_path'])
        # tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path)
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.tokenizer = tokenizer
        # if params.MODEL == 'bert_pretrained':
        #     self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")

    # @property
    # def _config(self):
    #     with open(ut.abspath(f'configs/data_config/{self.dtype}_config.json')) as f:
    #         config = json.load(f)
    #         return config

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        kingdom = self.kingdoms[index]
        seq = self.aa_seq[index] if self.dtype == 'aa' else self.smiles[index]
        label = torch.zeros(len(params.SP_LABELS), dtype=torch.int64)
        label[params.SP_LABELS[self.labels[index]]] = 1
        return seq, label, kingdom

    # def __getitem__(self, index):
    #     kingdom = self.kingdoms[index]
    #     seq = self.smiles[index] if self.dtype == 'smiles' else self.aa_seq[index]
    #     label = torch.zeros(len(params.SP_LABELS), dtype=torch.int64)
    #     label[params.SP_LABELS[self.labels[index]]] = 1
    #     input_ids = self.tokenizer.encode(
    #         seq,
    #         max_length=self._config['max_len'],
    #         padding="max_length",
    #         truncation=True
    #     )
    #     return torch.tensor(input_ids, dtype=torch.int64), label, kingdom
