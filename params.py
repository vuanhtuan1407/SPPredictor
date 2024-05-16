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
# Training
MODEL_TYPE = "cnn"
DATA_TYPE = 'smiles'
CONF_TYPE = 'default'
EPOCHS = 100
# ENV = 'kaggle'
USE_ORGANISM = False

# Testing
CHECKPOINT: str = "cnn-aa-lite-0_epochs=1.ckpt"

BATCH_SIZE = 8
LEARNING_RATE = 1e-7
NUM_WORKERS = 0  # set to 0 because of some random_seeding reason
# ORGANISM = 'others'  # currently do not need to use this param
FREEZE_PRETRAINED = False

DEVICES: list[int] | str | int = 'auto'
ACCELERATOR = 'cpu'

ENABLE_CHECKPOINTING = True

# DEVICE = 'cpu'  # use when applying old training process

"""
LOGGER CONFIGURATION
"""
USE_LOGGER = True
LOG_DIR = 'logs'  # relative path
