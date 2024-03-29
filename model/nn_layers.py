import math

import torch
from torch import nn

import params


class OrganismEmbedding(nn.Module):
    """
    Parameters `num_orgs` and `e_dim` must be the same
    """

    def __init__(self, num_orgs: int = 4, e_dim: int = 4):
        super().__init__()
        self.num_orgs = num_orgs
        self.embedding_dim = e_dim
        oe = torch.zeros(num_orgs, e_dim)
        for i in range(num_orgs):
            oe[i, i] = 1
        # oe = oe.unsqueeze(0)
        # self.register_buffer('oe', oe)
        self.oe = nn.Embedding.from_pretrained(torch.FloatTensor(oe), freeze=False)

    def forward(self, x):
        return self.oe(x)


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 100, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.input_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

    def forward(self, x):
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_x = self.pe[:, :x.size(1), :]
        x = x + pe_x
        return self.dropout(x)


class LinearPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.pe(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.encoder(x)
        # use average [CLS] token with all other word tokens
        # x = torch.mean(x, dim=1)

        # use only [CLS] token
        x = x[:, 0, :]
        return x


class ConvolutionalEncoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 512,
            dropout: float = 0.1,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            n_base: int = 1024,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_base, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=n_base, out_channels=n_base * 4, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=n_base * 4, out_channels=n_base, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=n_base, out_channels=embedding_dim, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        return x


class LSTMEncoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 512,
            hidden_size: int = 1024,
            n_layers: int = 4,
            dropout: float = 0.1,
            random_init: bool = False
    ):
        super().__init__()
        self.random_init = random_init
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return out, h_n, c_n


class StackedBiLSTMEncoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 512,
            hidden_size: int = 1024,
            n_layers: int = 4,
            dropout: float = 0.1,
            random_init: bool = False
    ):
        super().__init__()
        self.random_init = random_init
        # Init state
        if random_init:
            (h_0, c_0) = self.__init_state(n_layers=n_layers, hidden_size=hidden_size)
            self.init_state = (h_0, c_0)

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x):
        if self.random_init:
            out, (h_n, c_n) = self.bilstm(x, (self.init_state[0].detach(), self.init_state[1].detach()))
            return out, h_n, c_n
        else:
            out, (h_n, c_n) = self.bilstm(x)
            return out, h_n, c_n

    @staticmethod
    def __init_state(n_layers: int = 4, hidden_size: int = 1024):
        h_0 = torch.zeros(n_layers * 2, params.BATCH_SIZE, hidden_size).requires_grad_(True)
        c_0 = torch.zeros(n_layers * 2, params.BATCH_SIZE, hidden_size).requires_grad_(True)
        return h_0, c_0


class ParallelBiLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Classifier(nn.Module):
    def __init__(self, num_class: int, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.ff1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.ff2 = nn.Linear(in_features=d_ff, out_features=num_class)
        self.act1 = nn.ReLU()
        # self.act2 = nn.Softmax()

    def forward(self, x):
        x = self.act1(self.ff1(x))
        # x = self.act2(self.ff2(x))
        x = self.ff2(x)
        return x


if __name__ == "__main__":
    x = torch.randn(params.BATCH_SIZE, 30, 512)
    print(x.shape)
    model = StackedBiLSTMEncoder()
    out, h_n, c_n = model(x)
    print(out.shape, h_n.shape, c_n.shape)
