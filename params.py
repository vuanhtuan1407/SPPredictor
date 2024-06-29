# import os
import platform
from typing import Literal

OS_PLATFORM = platform.system()

SP_LABELS = dict(NO_SP=0, SP=1, LIPO=2, TAT=3, PILIN=4, TATLIPO=5)
ORGANISMS = dict(EUKARYA=0, POSITIVE=1, NEGATIVE=2, ARCHAEA=3)

"""
DATA PREPARATION
"""
USE_PREPARED_DATA = False
TRAIN_PATH = 'data/sp_data/train_set.fasta'
BENCHMARK_PATH = 'data/sp_data/benchmark_set_sp5.fasta'

USE_SPLIT_DATASET = False
ON_ORGANISM: Literal['eukarya', 'others'] = 'others'  # use when you set `USE_SPLIT_DATASET=True`

"""
MODEL AND TRAINER CONFIGURATION
"""
# Training
MODEL_TYPE = "gconv_trans"
DATA_TYPE: Literal['aa', 'smiles', 'graph'] = 'graph'
CONF_TYPE = 'default'
EPOCHS = 1
# ENV = 'kaggle'
USE_ORGANISM = True

# MODEL_TYPE = "transformer"
# DATA_TYPE: Literal['aa', 'smiles', 'graph'] = 'aa'
# CONF_TYPE = 'default'
# EPOCHS = 1
# # ENV = 'kaggle'
# USE_ORGANISM = True

# Testing
CHECKPOINT: str = "bert-aa-default-0_epochs=100.ckpt"

BATCH_SIZE = 8
LEARNING_RATE = 1e-7
NUM_WORKERS = 0  # set to 0 because of some random_seeding reason
FREEZE_PRETRAINED = False

DEVICES: list[int] | str | int = 'auto'
ACCELERATOR = 'auto'

ENABLE_CHECKPOINTING = False

# DEVICE = 'cpu'  # use when applying old training process

"""
LOGGER CONFIGURATION
"""
USE_LOGGER = False
LOG_DIR = 'logs'  # relative path
