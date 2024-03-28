import json

from torch import nn

import params
from model.nn_layers import InputEmbedding, LSTMEncoder, Classifier


class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        self.classifier = Classifier(num_class=len(params.SP_LABELS), d_model=config['hidden_size'] * 2)

    def forward(self, x):
        x = self.input_embedding(x)
        x, h_n, c_n = self.encoder(x)
        x = self.classifier(x[:, -1, :])
        return x


if __name__ == '__main__':
    configs = json.load(open('../configs/model_configs/lstm_config_default.json'))
    model = LSTMClassifier(configs)
    print(model)
