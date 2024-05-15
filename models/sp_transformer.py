# import pandas as pd
import json

import torch
from torch import nn

import params
from models.nn_layers import InputEmbedding, PositionalEncoding, TransformerEncoder, Classifier, OrganismEmbedding


class TransformerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_embedding = InputEmbedding(
            vocab_size=config['vocab_size'],
            d_model=config['d_model']
        )
        self.positional_encoding = PositionalEncoding(
            d_model=config['d_model'],
            dropout=config['dropout'],
            max_len=config['max_len']
        )
        self.encoder = TransformerEncoder(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers']
        )
        self.classifier = Classifier(
            d_model=config['d_model'],
            num_class=len(params.SP_LABELS)
        )

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class TransformerOrganismClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_embedding = InputEmbedding(
            vocab_size=config['vocab_size'],
            d_model=config['d_model']
        )
        self.positional_encoding = PositionalEncoding(
            d_model=config['d_model'],
            dropout=config['dropout'],
            max_len=config['max_len']
        )
        self.encoder = TransformerEncoder(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers']
        )
        self.organism_embedding = OrganismEmbedding(
            num_orgs=len(params.ORGANISMS),
            e_dim=config['d_model']
        )
        self.classifier = Classifier(
            d_model=config['d_model'] * 2,
            num_class=len(params.SP_LABELS)
        )

    def forward(self, x, org):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        org = self.organism_embedding(org)
        inp = torch.cat((x, org), dim=1)  # concat along sequence dim
        out = self.classifier(inp)
        return out


if __name__ == '__main__':
    config = json.load(open('../configs/smiles_configs/transformer_config_default.json'))
    model = TransformerClassifier(config)
    print(model)
    print(model.state_dict().keys())
    print(sum(p.numel() for p in model.parameters()))
