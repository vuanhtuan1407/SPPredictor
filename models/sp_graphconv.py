import torch
from torch import nn

import params
from models.nn_layers import GraphConvEncoder, OrganismEmbedding, Classifier


class GraphConvClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graphconv_encoder = GraphConvEncoder(
            d_model=config['d_model'],
            dropout=config['dropout'],
            use_relu_act=config['use_relu_act'],
            d_hidden=config['d_hidden'],
            use_special_tokens=False
        )
        self.classifier = Classifier(
            d_model=config['d_model'],
            num_class=len(params.SP_LABELS)
        )

    def forward(self, x):
        x = self.graphconv_encoder(x, x.ndata['n_feat'])
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x


class GraphConvOrganismClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graphconv_encoder = GraphConvEncoder(
            d_model=config['d_model'],
            dropout=config['dropout'],
            use_relu_act=config['use_relu_act'],
            d_hidden=config['d_hidden']
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
        x = self.graphconv_encoder(x, x.ndata['n_feat'])
        x = torch.mean(x, dim=1)
        org = self.organism_embedding(org)
        inp = torch.cat((x, org), dim=1)
        out = self.classifier(inp)
        return out
