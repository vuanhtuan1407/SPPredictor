import torch
from torch import nn

import params
from models.nn_layers import InputEmbedding, LSTMEncoder, Classifier, OrganismEmbedding


class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_embedding = InputEmbedding(
            vocab_size=config['vocab_size'],
            d_model=config['d_model']
        )
        self.encoder = LSTMEncoder(
            embedding_dim=config['d_model'],
            hidden_size=config['hidden_size'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],

        )
        self.classifier = Classifier(num_class=len(params.SP_LABELS), d_model=config['hidden_size'])

    def forward(self, x):
        x = self.input_embedding(x)
        x, h_n, c_n = self.encoder(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


class LSTMOrganismClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_embedding = InputEmbedding(
            vocab_size=config['vocab_size'],
            d_model=config['d_model']
        )
        self.encoder = LSTMEncoder(
            embedding_dim=config['d_model'],
            hidden_size=config['hidden_size'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],

        )
        self.organism_embedding = OrganismEmbedding(num_orgs=len(params.ORGANISMS), e_dim=config['hidden_size'])
        self.classifier = Classifier(num_class=len(params.SP_LABELS), d_model=config['hidden_size'] * 2)

    def forward(self, x, org):
        x = self.input_embedding(x)
        x, h_n, c_n = self.encoder(x)
        x = x[:, -1, :]
        org = self.organism_embedding(org)
        inp = torch.cat((x, org), dim=1)
        out = self.classifier(inp)
        return out
