# import os
import platform

OS_PLATFORM = platform.system()

TRAIN_PATH = 'data/sp_data/train_set.fasta'
BENCHMARK_PATH = 'data/sp_data/benchmark_set_sp5.fasta'

SP_LABELS = dict(NO_SP=0, SP=1, LIPO=2, TAT=3, PILIN=4, TATLIPO=5)
ORGANISMS = dict(EUKARYA=0, POSITIVE=1, NEGATIVE=2, ARCHAEA=3)
# ORGANISMS = dict(POSITIVE=1, NEGATIVE=2, ARCHAEA=3)

"""
MODEL AND TRAINER CONFIGURATION
"""
# Adjust the 5 following params to choose the checkpoint for testing
MODEL_TYPE = 'cnn'
DATA_TYPE = 'aa'
CONF_TYPE = 'default'
EPOCHS = 1
ENV = 'local'
CHECKPOINT_VER: int | None = None

BATCH_SIZE = 8
LEARNING_RATE = 1e-7
NUM_WORKERS = 1 if OS_PLATFORM == 'Windows' else 2
ORGANISM = 'others'
FREEZE_PRETRAINED = False

DEVICES: list[int] | str | int = 'auto'
ACCELERATOR = 'auto'

# CHECKPOINT: str | None = None

DEVICE = 'cpu'  # use for apply old training process

"""
LOGGER CONFIGURATION
"""
KAGGLE_DIR = '/kaggle/working'
LOG_DIR = 'logs'  # relative path
