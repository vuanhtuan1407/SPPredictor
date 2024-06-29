import math
from itertools import islice

import torch
from dgl.nn.pytorch import GraphConv
from torch import nn

import params


class OrganismEmbedding(nn.Module):
    def __init__(self, num_orgs: int = 4, e_dim: int = 512):
        super().__init__()
        self.num_orgs = num_orgs
        self.embedding_dim = e_dim
        torch.random.manual_seed(0)
        oe = torch.randn(num_orgs, e_dim)
        self.organism_embedding = nn.Embedding.from_pretrained(oe, freeze=False)

    def forward(self, x):
        return self.organism_embedding(x)  # upsize from (batch, model) to (batch, 1, model)


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
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 2048):
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
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, mask=None):
        x = self.encoder(x, src_key_padding_mask=mask)
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
        self.conv1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_base, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=n_base, out_channels=n_base * 4, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=n_base * 4, out_channels=n_base, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=n_base, out_channels=embedding_dim, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
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


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""

    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data)
        h = self.activation(h)
        return {'h': h}


class GraphConvEncoder(nn.Module):
    def __init__(self, d_model: int = 512, n_layers=2, dropout: float = 0.1, use_relu_act: bool = True,
                 d_hidden: int = 1024, use_special_tokens: bool = False):
        super().__init__()
        self.d_model = d_model
        self.use_special_tokens = use_special_tokens
        self.act = None
        if use_relu_act:
            self.act = nn.ReLU()
        self.convs = nn.ModuleList()
        convFirst = GraphConv(20, d_hidden, norm='both', bias=True, activation=self.act, allow_zero_in_degree=True)
        self.convs.append(convFirst)

        for i in range(1, n_layers - 1):
            convIn = GraphConv(d_hidden, d_hidden, norm='both', bias=True, activation=self.act,
                               allow_zero_in_degree=True)
            self.convs.append(convIn)

        convLast = GraphConv(d_hidden, d_model, norm='both', bias=True, activation=self.act,
                             allow_zero_in_degree=True)
        self.convs.append(convLast)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h):
        for (i, conv) in enumerate(self.convs):
            h = conv(x, h)
        # h = self.return_batch(h, x.batch_num_nodes(), max_len='longest')
        if self.use_special_tokens:
            h, mask = self.return_batch_plus(h, x.batch_num_nodes(), max_len='longest')
            # h = torch.reshape(h, (-1, 20, self.d_model))
            h = self.dropout(h)
            return h, mask
        else:
            h = self.return_batch(h, x.batch_num_nodes(), max_len='longest')
            h = self.dropout(h)
            return h

    def return_batch(self, h, batch_num_nodes, max_len: str | int = 70):
        device = h.get_device()
        tmp = [list(islice(iter(h), 0, num_nodes)) for num_nodes in batch_num_nodes]
        ret = []
        if max_len == "longest":
            max_len = torch.max(batch_num_nodes).item()
        if not isinstance(max_len, int):
            raise ValueError('Use `int` or "longest"')

        for i, sample in enumerate(tmp):
            if len(sample) > max_len:
                sample = sample[:max_len]
            else:
                while len(sample) < max_len:
                    pad = torch.zeros(self.d_model, device=device)
                    sample.append(pad)
            ret.append(torch.stack(sample))

        return torch.stack(ret)

    def return_batch_plus(self, h, batch_num_nodes, max_len: str | int = 70):
        device = h.get_device()
        tmp = [list(islice(iter(h), 0, num_nodes)) for num_nodes in batch_num_nodes]
        ret = []
        if max_len == "longest":
            max_len = torch.max(batch_num_nodes).item()
        if not isinstance(max_len, int):
            raise ValueError('Use `int` or "longest"')
        mask = torch.zeros((len(batch_num_nodes), max_len + 2), dtype=torch.float, device=device)
        mask[:, -1] = float('-inf')
        for i, sample in enumerate(tmp):
            if len(sample) > max_len:
                sample = sample[:max_len]
            else:
                while len(sample) < max_len:
                    pad = torch.zeros(self.d_model, device=device)
                    sample.append(pad)
                    mask[i, len(sample)] = float('-inf')
            sample = self.add_special_tokens(sample, device)
            ret.append(torch.stack(sample))

        return torch.stack(ret), mask

    def add_special_tokens(self, sample, device):
        cls = torch.zeros(self.d_model, device=device)  # begin of sentence [CLS]
        eos = torch.zeros(self.d_model, device=device)  # end of sentence [EOS]
        return [cls, *sample, eos]


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
    x = torch.tensor(2)
    oe = OrganismEmbedding(num_orgs=4, e_dim=5)
    print(oe(x))
