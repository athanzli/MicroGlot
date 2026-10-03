"""MicroGlot's byte-pair-encoding (BPE) tokenizer: built here, trained by train_bpe_tokenizer.py."""
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

PAD, BOS, EOS = "[PAD]", "[BOS]", "[EOS]"


class GenomicBPETokenizer(PreTrainedTokenizerFast):
    """BPE over DNA. Lowercase is read as uppercase; every N is a token of its own (an ambiguous base); characters
    outside the vocabulary become N; sequences are wrapped as [BOS] ... [EOS]."""

    def __init__(self, vocab_size=8192, max_token_length=None, tokenizer_object=None, **kwargs):
        self.target_vocab_size, self.max_token_length = vocab_size, max_token_length
        if tokenizer_object is None and "tokenizer_file" not in kwargs:  # a new tokenizer, to be trained
            tokenizer_object = Tokenizer(models.BPE(unk_token="N"))
            tokenizer_object.normalizer = normalizers.Sequence(
                [normalizers.NFKC()] + [normalizers.Replace(c, c.upper()) for c in "acgtn"])
            tokenizer_object.pre_tokenizer = pre_tokenizers.Split(pattern="N", behavior="isolated")
            tokenizer_object.decoder = decoders.Fuse()
        super().__init__(tokenizer_object=tokenizer_object, **kwargs)

    def train_from_sequences(self, sequences, min_frequency=0, show_progress=True):
        """Learn merges from the N-free segments of `sequences` (N itself is a single-character token)."""
        trainer = trainers.BpeTrainer(vocab_size=self.target_vocab_size, min_frequency=min_frequency,
                                      special_tokens=[PAD, BOS, EOS], initial_alphabet=["A", "T", "C", "G", "N"],
                                      show_progress=show_progress, max_token_length=self.max_token_length)
        segments = (part for seq in sequences for part in seq.upper().split("N") if part)
        self.backend_tokenizer.train_from_iterator(segments, trainer=trainer)
        self.add_special_tokens({"pad_token": PAD, "bos_token": BOS, "eos_token": EOS})
        vocab = self.backend_tokenizer.get_vocab()
        self.backend_tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS}:0 $A:0 {EOS}:0", pair=f"{BOS}:0 $A:0 {EOS}:0 {BOS}:1 $B:1 {EOS}:1",
            special_tokens=[(BOS, vocab[BOS]), (EOS, vocab[EOS])])
