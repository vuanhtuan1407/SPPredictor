# import os
import platform
from typing import Union

OS_PLATFORM = platform.system()

TRAIN_PATH = 'data/sp_data/train_set.fasta'
BENCHMARK_PATH = 'data/sp_data/benchmark_set_sp5.fasta'

SP_LABELS = dict(NO_SP=0, SP=1, LIPO=2, TAT=3, PILIN=4, TATLIPO=5)
# KINGDOM = dict(EUKARYA=0, POSITIVE=1, NEGATIVE=2, ARCHAEA=1)
KINGDOM = dict(POSITIVE=1, NEGATIVE=2, ARCHAEA=3)
"""
MODEL AND TRAINING CONFIGURATION
"""
ENV = 'local'
EPOCHS = 3 if OS_PLATFORM == 'Windows' else 10
BATCH_SIZE = 8
MODEL = 'cnn'
DATA = 'aa'
CONF_TYPE = 'default'
DEVICES: Union[list[int], str, int] = 'auto'
ACCELERATOR = 'auto'
NUM_WORKERS = 1 if OS_PLATFORM == 'Windows' else 2
ORGANISM = 'others'

DEVICE = 'cpu'  # use for apply old training process

"""
LOGGER CONFIGURATION
"""
KAGGLE_DIR = '/kaggle/working'
LOG_DIR = 'logs'
