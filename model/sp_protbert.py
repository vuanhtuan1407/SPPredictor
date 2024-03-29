from torch import nn
from transformers import BertModel

import params
from model.nn_layers import Classifier


class ProtBertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config=config)
        # self.bert_encoder = get_peft_model(encoder, peft_config)
        self.classifier = Classifier(num_class=len(params.SP_LABELS), d_model=config.hidden_size)

    def forward(self, x):
        x = self.bert(x)
        x = x.last_hidden_state[:, 0, :]
        x = self.classifier(x)
        return x
