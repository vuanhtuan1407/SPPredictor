# import pandas as pd
import json

from torch import nn

import params
from model.nn_layers import InputEmbedding, PositionalEncoding, TransformerEncoder, Classifier


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
        # print("Is Nan Input: ", x.isnan().any())
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        # print("Is Nan Embedding: ", x.isnan().any(), x[:2,:10])
        # print(x[0, :)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    config = json.load(open('../configs/transformer_config_default.json'))
    model = TransformerClassifier(config)
    print(model)
    print(model.state_dict().keys())
    print(sum(p.numel() for p in model.parameters()))
