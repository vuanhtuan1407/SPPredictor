import torch
from torch import nn

import params
from models.nn_layers import InputEmbedding, ConvolutionalEncoder, Classifier


class ConvolutionalClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_embedding = InputEmbedding(
            vocab_size=config['vocab_size'],
            d_model=config['d_model']
        )
        self.conv_encoder = ConvolutionalEncoder(
            embedding_dim=config['d_model'],
            kernel_size=config['kernel_size'],
            n_base=config['n_base']
        )
        self.flatten = nn.Flatten()
        self.classifier = Classifier(num_class=len(params.SP_LABELS), d_model=1024)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.input_embedding(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv_encoder(x)
        x = self.flatten(x)
        #         print(x.shape)
        x = self.classifier(x)
        return x
