from transformers import BertModel

import configs.config_utils as cut
import params
from models.sp_bilstm import StackedBiLSTMClassifier
from models.sp_cnn import ConvolutionalClassifier
from models.sp_cnn_transformer import CNNTransformerClassifier
from models.sp_lstm import LSTMClassifier
from models.sp_transformer import TransformerClassifier

# Define types of model
TRANSFORMER = 'transformer'
CNN = 'cnn'
LSTM = 'lstm'
STACKED_BILSTM = 'st_bilstm'
BERT = 'bert'
BERT_PRETRAINED = 'bert_pretrained'
CNN_TRANSFORMER = 'cnn_trans'


def load_model(model, data, conf_type):
    config = cut.load_config(model, data, conf_type)
    if model == TRANSFORMER:
        return TransformerClassifier(config)
    elif model == CNN:
        return ConvolutionalClassifier(config)
    elif model == STACKED_BILSTM:
        return StackedBiLSTMClassifier(config)
    elif model == BERT or model == BERT_PRETRAINED:
        return BertModel(config)
    elif model == LSTM:
        return LSTMClassifier(config)
    elif model == CNN_TRANSFORMER:
        return CNNTransformerClassifier(config)
    else:
        return ValueError("Unknown model type")
