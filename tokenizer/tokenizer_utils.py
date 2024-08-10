import os

from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import GPT2TokenizerFast, BertTokenizer

import data.data_utils as dut
import utils as ut

SAVE_PATH = './tokenizer_smiles.json'


def train_bpe_tokenizer():
    # Build corpus file
    corpus_path = dut.SMILES_CORPUS_PATH
    if not os.path.exists(corpus_path):
        print("File not found. Creating new corpus file...")
        dut.create_smiles_training_tokenizer()
        print('Creating finished')
    else:
        print('Corpus already existed')

    # Training
    if not os.path.exists(SAVE_PATH):
        print("Build tokenizer")
        # Build tokenizer models
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

        print("Training tokenizer")
        tokenizer.train(files=[corpus_path], trainer=trainer)
        tokenizer.save(SAVE_PATH)
        print('Training finished')
    else:
        print('Tokenizer already existed')


def load_tokenizer(model_type, data_type):
    if data_type in ['aa', 'smiles']:
        tokenizer_path = ut.abspath(f'tokenizer/tokenizer_{data_type}.json')
        tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if model_type == 'bert_pretrained':
            tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
        return tokenizer
    else:
        return None


if __name__ == '__main__':
    train_bpe_tokenizer()
