import os

from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

import utils as ut
from data import data_utils as dut

SAVE_PATH = './tokenizer_smiles.json'


def train_bpe_tokenizer():
    # Build tokenizer model
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))

    # Normalization
    tokenizer.normalizer = Sequence([NFD(), StripAccents()])

    # Pre-tokenization
    tokenizer.pre_tokenizer = Whitespace()

    # Trainer
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = BpeTrainer(special_tokens=special_tokens)

    # Post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", special_tokens.index("[CLS]")),
            ("[SEP]", special_tokens.index("[SEP]")),
        ],
    )

    # Decoder
    tokenizer.decoder = BPEDecoder()

    # Training
    corpus_path = ut.get_absolute_path(dut.SMILES_CORPUS_PATH)
    if not os.path.exists(corpus_path):
        print("File not found. Creating new corpus file...")
        dut.create_smiles_training_tokenizer()
        print('Creating finished')
    else:
        print('File already existed')

    print("Training tokenizer")
    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(SAVE_PATH)
    print('Training finished')


if __name__ == "__main__":
    train_bpe_tokenizer()
