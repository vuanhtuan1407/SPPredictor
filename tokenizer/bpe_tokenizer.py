from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


class BpeTokenizer:
    def __init__(self, vocab_size, model=BPE(unk_token='[UNK]')):
        tokenizer = Tokenizer(model=BPE(unk_token='[UNK]'))
        self.tokenizer = tokenizer
        self.tokenizer.vocab_size = vocab_size
        self.tokenizer.normalizer = Sequence([NFD(), StripAccents()])
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.tokenizer.special_tokens.index("[CLS]")),
                ("[SEP]", self.tokenizer.special_tokens.index("[SEP]")),
            ],
        )
        self.trainer = BpeTrainer(vocab_size=self.tokenizer.vocab_size, special_tokens=self.tokenizer.special_tokens)
        self.tokenizer.decoder = BPEDecoder()
        # self.save_path = 'tokenizer.json'
        self.save_path = 'tokenizer_aa.json'

    def train_tokenizer(self):
        # src_path = '../data/sp_data/data_full_for_training_tokenizer.txt'
        vocab_path = 'vocab.txt'
        self.tokenizer.train(files=[vocab_path], trainer=self.trainer)
        self.tokenizer.save(path=self.save_path)  # Auto save

    # def build_tokenizer(self):
    #     # Build tokenizer model
    #     tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    #
    #     # Normalization
    #     tokenizer.normalizer = Sequence([NFD(), StripAccents()])
    #
    #     # Pre-tokenization
    #     tokenizer.pre_tokenizer = Whitespace()
    #
    #     # Trainer
    #     special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    #     trainer = BpeTrainer(vocal_size=self.vocab_size, special_tokens=special_tokens)
    #
    #     # Post-processing
    #     tokenizer.post_processor = TemplateProcessing(
    #         single="[CLS] $A [SEP]",
    #         pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    #         special_tokens=[
    #             ("[CLS]", special_tokens.index("[CLS]")),
    #             ("[SEP]", special_tokens.index("[SEP]")),
    #         ],
    #     )
    #
    #     # Decoder
    #     tokenizer.decoder = BPEDecoder()
    #
    #     return tokenizer


if __name__ == "__main__":
    bpe_tokenizer = BpeTokenizer(vocab_size=100)
    bpe_tokenizer.train_tokenizer()
