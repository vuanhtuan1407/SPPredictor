# import json
#
# from model.sp_bilstm import StackedBiLSTMClassifier
# from model.sp_cnn import ConvolutionalClassifier
# from model.sp_transformer import TransformerClassifier
import os.path
from pathlib import Path

import params


# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_absolute_path(path):
    apath = str(Path(params.ROOT_DIR) / path)
    if not os.path.exists(apath):
        raise FileNotFoundError("Path does not exist")
    else:
        return apath
