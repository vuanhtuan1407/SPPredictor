import torch
from torch import nn

import params
from models.nn_layers import InputEmbedding, StackedBiLSTMEncoder, Classifier, OrganismEmbedding


class StackedBiLSTMClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_embedding = InputEmbedding(
            vocab_size=config['vocab_size'],
            d_model=config['d_model']
        )
        self.stacked_encoder = StackedBiLSTMEncoder(
            embedding_dim=config['d_model'],
            hidden_size=config['hidden_size'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],

        )
        self.classifier = Classifier(num_class=len(params.SP_LABELS), d_model=config['hidden_size'] * 2)

    def forward(self, x):
        x = self.input_embedding(x)
        x, h_n, c_n = self.stacked_encoder(x)
        x = x[:, -1, :]  # get the last hidden embedding
        x = self.classifier(x)
        return x


class StackedBiLSTMOrganismClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_embedding = InputEmbedding(
            vocab_size=config['vocab_size'],
            d_model=config['d_model']
        )
        self.stacked_encoder = StackedBiLSTMEncoder(
            embedding_dim=config['d_model'],
            hidden_size=config['hidden_size'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],

        )
        self.organism_embedding = OrganismEmbedding(num_orgs=len(params.ORGANISMS), e_dim=config['hidden_size'] * 2)
        self.classifier = Classifier(num_class=len(params.SP_LABELS), d_model=config['hidden_size'] * 4)

    def forward(self, x, org):
        x = self.input_embedding(x)
        x, h_n, c_n = self.stacked_encoder(x)
        x = x[:, -1, :]
        org = self.organism_embedding(org)
        inp = torch.cat((x, org), dim=1)
        out = self.classifier(inp)
        return out
