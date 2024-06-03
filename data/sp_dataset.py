import json
from typing import Optional

import dgl
import pandas as pd
import torch
from torch.utils.data import Dataset

import params


class SPDataset(Dataset):
    def __init__(self, json_paths: Optional[list[str]], data_type: str):
        self.data_type = data_type
        if json_paths is None or isinstance(json_paths, str):
            raise ValueError('provide path to dataset in list of str')
        df = pd.DataFrame(self._read_jsons(json_paths))
        self.length = len(df)
        self.labels = df['label'].tolist()
        self.organisms = df['kingdom'].tolist()
        if data_type == 'graph':
            self.from_list = df['from_list'].tolist()
            self.to_list = df['to_list'].tolist()
            self.adj_matrix = df['adj_matrix'].tolist()
        else:
            self.smiles = df['smiles'].tolist()
            self.aa_seq = df['aa_seq'].tolist()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        organism = torch.tensor(params.ORGANISMS[self.organisms[index]])
        label = torch.zeros(len(params.SP_LABELS), dtype=torch.int64)
        label[params.SP_LABELS[self.labels[index]]] = 1
        if self.data_type == 'graph':
            graph = dgl.graph((self.from_list[index], self.to_list[index]))
            graph = dgl.add_self_loop(graph)
            graph.ndata['n_feat'] = torch.tensor(self.adj_matrix[index], dtype=torch.float)
            return graph, label.clone().detach(), organism.clone().detach()
        else:
            seq = self.aa_seq[index] if self.data_type == 'aa' else self.smiles[index]
            return seq, label.clone().detach(), organism.clone().detach()  # return (list[int], list[int], int)

    @staticmethod
    def _read_jsons(json_paths: list[str]):
        data = []
        for path in json_paths:
            with open(path, 'r') as f:
                data.extend(json.load(f))
        return data
