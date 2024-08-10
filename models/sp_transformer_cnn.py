import torch
from torch import nn

import params
from models.nn_layers import InputEmbedding, ConvolutionalEncoder, PositionalEncoding, TransformerEncoder, Classifier


class TransformerCNNClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_embedding = InputEmbedding(
            vocab_size=config['vocab_size'],
            d_model=config['d_model']
        )
        self.conv_encoder = ConvolutionalEncoder(
            embedding_dim=config['d_model'],
            kernel_size=config['kernel_size']
        )
        self.positional_encoding = PositionalEncoding(
            d_model=config['d_model'],
            dropout=config['dropout'],
            max_len=config['max_len']
        )
        self.trans_encoder = TransformerEncoder(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers']
        )
        self.flatten = nn.Flatten()
        self.classifier = Classifier(
            d_model=config['d_model'],
            num_class=len(params.SP_LABELS)
        )

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.trans_encoder(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv_encoder(x)
        x = torch.transpose(x, 1, 2)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
