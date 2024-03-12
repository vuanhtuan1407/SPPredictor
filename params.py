import os

import torch

TRAIN_PATH = 'data/sp_data/train_set.fasta'
BENCHMARK_PATH = 'data/sp_data/benchmark_set_sp5.fasta'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# TRANSFORMER_CONFIG_DEFAULT = 'configs/transformer_config_default.json'

EPOCHS = 3

"""
MODEL CONFIGURATION
"""
SP_LABELS = dict(NO_SP=0, SP=1, LIPO=2, TAT=3, PILIN=4, TATLIPO=5)
KINGDOM = dict(EUKARYA=0, POSITIVE=1, NEGATIVE=2, ARCHAEA=1)
BATCH_SIZE = 8
MODEL = 'transformer'
DATA = 'aa'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
LOGGER CONFIGURATION
"""
KAGGLE_DIR = '/kaggle/working'
DEFAULT_LOG_DIR = 'logs'
