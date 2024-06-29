import torch
from torch import nn

import params
from models.nn_layers import GraphConvEncoder, TransformerEncoder, Classifier, OrganismEmbedding, PositionalEncoding


class GraphConvTransformerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graphconv_encoder = GraphConvEncoder(
            d_model=config['d_model'],
            dropout=config['dropout'],
            use_relu_act=config['use_relu_act'],
            d_hidden=config['d_hidden'],
            use_special_tokens=True
        )
        self.positional_encoding = PositionalEncoding(
            d_model=config['d_model'],
            max_len=config['max_len'],
        )
        self.transformer_encoder = TransformerEncoder(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers']
        )
        self.classifier = Classifier(
            d_model=config['d_model'],
            num_class=len(params.SP_LABELS)
        )

    def forward(self, x):
        x, mask = self.graphconv_encoder(x, x.ndata['n_feat'])
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=mask)
        x = self.classifier(x)
        return x


class GraphConvTransformerOrganismClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graphconv_encoder = GraphConvEncoder(
            d_model=config['d_model'],
            dropout=config['dropout'],
            use_relu_act=config['use_relu_act'],
            d_hidden=config['d_hidden'],
            use_special_tokens=True
        )
        self.positional_encoding = PositionalEncoding(
            d_model=config['d_model'],
            max_len=config['max_len'],
        )
        self.transformer_encoder = TransformerEncoder(
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
        x, mask = self.graphconv_encoder(x, x.ndata['n_feat'])
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=mask)
        org = self.organism_embedding(org)
        inp = torch.cat((x, org), dim=1)
        out = self.classifier(inp)
        return out
