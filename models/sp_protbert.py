import torch
from torch import nn
from transformers import BertModel

import params
from models.nn_layers import Classifier, OrganismEmbedding


class ProtBertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config=config)
        # self.bert_encoder = get_peft_model(encoder, peft_config)
        if params.FREEZE_PRETRAINED and params.MODEL_TYPE == 'bert_pretrained':
            self.freeze_pretrained_layer()
        self.classifier = Classifier(num_class=len(params.SP_LABELS), d_model=config.hidden_size)

    def forward(self, x):
        x = self.bert(x)
        x = x.last_hidden_state[:, 0, :]
        x = self.classifier(x)
        return x

    def freeze_pretrained_layer(self):
        for param in self.bert.parameters():
            param.requires_grad = False


class ProtBertOrganismClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config=config)
        if params.FREEZE_PRETRAINED and params.MODEL_TYPE == "bert_pretrained":
            self.freeze_pretrained_layer()
        self.classifier = Classifier(num_class=len(params.SP_LABELS), d_model=config.hidden_size * 2)
        self.organism_embedding = OrganismEmbedding(num_orgs=len(params.ORGANISMS), e_dim=config.hidden_size)

    def forward(self, x, org):
        x = self.bert(x)
        x = x.last_hidden_state[:, 0, :]
        org = self.organism_embedding(org)
        inp = torch.cat((x, org), dim=1)
        out = self.classifier(inp)
        return out

    def freeze_pretrained_layer(self):
        for param in self.bert.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    import configs.config_utils as cut
    config = cut.load_config('bert', 'aa', 'default')
    model = ProtBertOrganismClassifier(config)
    print(model)
