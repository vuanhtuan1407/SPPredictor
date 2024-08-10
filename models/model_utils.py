import configs.config_utils as cut
from models.sp_bilstm import StackedBiLSTMClassifier, StackedBiLSTMOrganismClassifier
from models.sp_cnn import ConvolutionalClassifier, ConvolutionalOrganismClassifier
from models.sp_cnn_transformer import CNNTransformerClassifier, CNNTransformerOrganismClassifier
from models.sp_graphconv import GraphConvClassifier, GraphConvOrganismClassifier
from models.sp_graphconv_transformer import GraphConvTransformerOrganismClassifier, GraphConvTransformerClassifier
from models.sp_lstm import LSTMClassifier, LSTMOrganismClassifier
from models.sp_protbert import ProtBertClassifier, ProtBertOrganismClassifier
from models.sp_transformer import TransformerClassifier, TransformerOrganismClassifier

# Define types of model
TRANSFORMER = 'transformer'
CNN = 'cnn'
LSTM = 'lstm'
STACKED_BILSTM = 'st_bilstm'
BERT = 'bert'
BERT_PRETRAINED = 'bert_pretrained'
CNN_TRANSFORMER = 'cnn_trans'
GCONV = 'gconv'
GCONV_TRANSFORMER = 'gconv_trans'


def load_model(model_type, data_type, conf_type, use_organism=False):
    config = cut.load_config(model_type, data_type, conf_type)
    if use_organism:
        if model_type == TRANSFORMER:
            return TransformerOrganismClassifier(config)
        elif model_type == CNN:
            return ConvolutionalOrganismClassifier(config)
        elif model_type == STACKED_BILSTM:
            return StackedBiLSTMOrganismClassifier(config)
        elif model_type == BERT or model_type == BERT_PRETRAINED:
            return ProtBertOrganismClassifier(config)
        elif model_type == LSTM:
            return LSTMOrganismClassifier(config)
        elif model_type == CNN_TRANSFORMER:
            return CNNTransformerOrganismClassifier(config)
        elif model_type == GCONV and data_type == 'graph':
            return GraphConvOrganismClassifier(config)
        elif model_type == GCONV_TRANSFORMER and data_type == 'graph':
            return GraphConvTransformerOrganismClassifier(config)
        else:
            return ValueError("Unknown model_type type")
    else:
        if model_type == TRANSFORMER:
            return TransformerClassifier(config)
        elif model_type == CNN:
            return ConvolutionalClassifier(config)
        elif model_type == STACKED_BILSTM:
            return StackedBiLSTMClassifier(config)
        elif model_type == BERT or model_type == BERT_PRETRAINED:
            return ProtBertClassifier(config)
        elif model_type == LSTM:
            return LSTMClassifier(config)
        elif model_type == CNN_TRANSFORMER:
            return CNNTransformerClassifier(config)
        elif model_type == GCONV and data_type == 'graph':
            return GraphConvClassifier(config)
        elif model_type == GCONV_TRANSFORMER and data_type == 'graph':
            return GraphConvTransformerClassifier(config)
        else:
            return ValueError("Unknown model_type type")
