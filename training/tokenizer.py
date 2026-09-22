"""Byte-pair-encoding tokenizer for genomic sequences (trainer + loader)."""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

class GenomicBPETokenizer(PreTrainedTokenizerFast):
    """Fast BPE Tokenizer for genomic sequences."""

    vocab_files_names = {
        "tokenizer_file": "tokenizer.json",
        "vocab_file": "vocab.json",
    }

    model_input_names = ["input_ids", "attention_mask"]

    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"

    DNA_NUCLEOTIDES = ["A", "T", "C", "G", "N"]

    def __init__(
        self,
        tokenizer_object: Optional[Tokenizer] = None,
        vocab_size: int = 1024,
        max_token_length: Optional[int] = 16,
        **kwargs,
    ):
        """Initialize the genomic BPE tokenizer for causal language modeling."""
        self.target_vocab_size = vocab_size
        self.max_token_length = max_token_length

        self.training_stats = kwargs.pop("training_stats", None)

        self._is_new_tokenizer = tokenizer_object is None and "tokenizer_file" not in kwargs

        if self._is_new_tokenizer:
            tokenizer_object = self._create_base_tokenizer()
        else:
            kwargs.setdefault("pad_token", self.PAD_TOKEN)
            kwargs.setdefault("eos_token", self.EOS_TOKEN)
            kwargs.setdefault("bos_token", self.BOS_TOKEN)
            kwargs.setdefault("unk_token", "N")

        super().__init__(
            tokenizer_object=tokenizer_object,
            **kwargs,
        )

    def _create_base_tokenizer(self) -> Tokenizer:
        """Create the base tokenizer with BPE model."""
        tokenizer = Tokenizer(models.BPE(unk_token="N"))

        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Replace("a", "A"),
            normalizers.Replace("c", "C"),
            normalizers.Replace("g", "G"),
            normalizers.Replace("t", "T"),
            normalizers.Replace("n", "N"),
        ])

        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern="N",
            behavior="isolated",
        )

        tokenizer.decoder = decoders.Fuse()

        return tokenizer

    def _setup_post_processor(self):
        """Set up the post-processor to add BOS and EOS tokens."""
        vocab = self.backend_tokenizer.get_vocab()

        if self.BOS_TOKEN not in vocab or self.EOS_TOKEN not in vocab:
            print(f"Warning: BOS ({self.BOS_TOKEN}) or EOS ({self.EOS_TOKEN}) not in vocab, skipping post-processor setup")
            return

        bos_id = vocab[self.BOS_TOKEN]
        eos_id = vocab[self.EOS_TOKEN]

        self.backend_tokenizer.post_processor = TemplateProcessing(
            single=f"{self.BOS_TOKEN}:0 $A:0 {self.EOS_TOKEN}:0",
            pair=f"{self.BOS_TOKEN}:0 $A:0 {self.EOS_TOKEN}:0 {self.BOS_TOKEN}:1 $B:1 {self.EOS_TOKEN}:1",
            special_tokens=[
                (self.BOS_TOKEN, bos_id),
                (self.EOS_TOKEN, eos_id),
            ],
        )

    def train_from_sequences(
        self,
        sequences: List[str],
        min_frequency: int = 100,
        show_progress: bool = True,
        special_tokens: Optional[List[str]] = None,
    ):
        """Train the BPE tokenizer on genomic sequences."""
        if special_tokens is None:
            special_tokens = [
                self.PAD_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
            ]

        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=self.DNA_NUCLEOTIDES,
            show_progress=show_progress,
            max_token_length=self.max_token_length,
        )

        def split_sequences_by_n(seqs):
            for seq in seqs:
                seq_upper = seq.upper()
                for segment in seq_upper.split("N"):
                    if segment:
                        yield segment

        segments = split_sequences_by_n(sequences)
        self.backend_tokenizer.train_from_iterator(segments, trainer=trainer)

        self._update_special_token_ids()

        self._setup_post_processor()

        print(f"Training complete. Vocabulary size: {self.backend_tokenizer.get_vocab_size()}")

    def _update_special_token_ids(self):
        """Update special token attributes after training."""
        vocab = self.backend_tokenizer.get_vocab()

        special_tokens_dict = {}
        if self.PAD_TOKEN in vocab:
            special_tokens_dict["pad_token"] = self.PAD_TOKEN
        if self.BOS_TOKEN in vocab:
            special_tokens_dict["bos_token"] = self.BOS_TOKEN
        if self.EOS_TOKEN in vocab:
            special_tokens_dict["eos_token"] = self.EOS_TOKEN

        if special_tokens_dict:
            self.add_special_tokens(special_tokens_dict)

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str, ...]:
        """Save the tokenizer to a directory (HuggingFace standard method)."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        saved_files = super().save_pretrained(
            save_directory,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub,
            **kwargs,
        )

        vocab = self.backend_tokenizer.get_vocab()

        vocab_file = save_directory / "vocab.json"
        with open(vocab_file, "w") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)

        try:
            tokenizer_json_path = save_directory / "tokenizer.json"
            if tokenizer_json_path.exists():
                with open(tokenizer_json_path, "r") as f:
                    tokenizer_data = json.load(f)
                if "model" in tokenizer_data and "merges" in tokenizer_data["model"]:
                    merges = tokenizer_data["model"]["merges"]
                    merges_file = save_directory / "merges.txt"
                    with open(merges_file, "w") as f:
                        f.write(f"# BPE merges ({len(merges)} total)\n")
                        f.write(f"# Format: token1 token2 (merged to form new token)\n")
                        for merge in merges:
                            f.write(f"{merge}\n")
        except Exception:
            pass

        config_file = save_directory / "tokenizer_config.json"
        config = {}
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)

        config.update({
            "tokenizer_class": "GenomicBPETokenizer",
            "vocab_size": self.target_vocab_size,
            "max_token_length": self.max_token_length,
            "actual_vocab_size": len(vocab),
            "training_stats": self.training_stats,
            "auto_map": {
                "AutoTokenizer": [None, "tokenizer.GenomicBPETokenizer"],
            },
        })

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Tokenizer saved to {save_directory}")
        print(f"  - tokenizer.json: Full tokenizer (for loading)")
        print(f"  - vocab.json: Vocabulary backup ({len(vocab)} tokens)")
        print(f"  - tokenizer_config.json: Config + training stats")
        if self.training_stats:
            print(f"  - Training stats: {self.training_stats['total_bp']:,} bp from {self.training_stats['total_segments']:,} segments")

        return saved_files

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *init_inputs,
        **kwargs,
    ) -> "GenomicBPETokenizer":
        """Load a pre-trained genomic tokenizer (HuggingFace standard method)."""
        path = Path(pretrained_model_name_or_path)
        training_stats = None
        target_vocab_size = kwargs.pop("vocab_size", 1024)
        max_token_length = kwargs.pop("max_token_length", 16)

        if path.exists():
            config_path = path / "tokenizer_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                training_stats = config.get("training_stats", None)
                target_vocab_size = config.get("vocab_size", target_vocab_size)
                max_token_length = config.get("max_token_length", max_token_length)

        instance = super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            **kwargs,
        )

        instance.target_vocab_size = target_vocab_size
        instance.max_token_length = max_token_length
        instance.training_stats = training_stats

        if instance.backend_tokenizer.post_processor is None:
            instance._setup_post_processor()

        vocab_size = instance.backend_tokenizer.get_vocab_size()
        print(f"Loaded GenomicBPETokenizer from {pretrained_model_name_or_path}")
        print(f"  Vocabulary size: {vocab_size}")
        if training_stats:
            print(f"  Trained on: {training_stats.get('total_bp', 'unknown'):,} bp")

        return instance

    def __repr__(self) -> str:
        return f"GenomicBPETokenizer(vocab_size={len(self)})"


def create_genomic_tokenizer(
    sequences: Optional[List[str]] = None,
    vocab_size: int = 1024,
    min_frequency: int = 100,
    max_token_length: int = 16,
) -> GenomicBPETokenizer:
    """Convenience function to create and optionally train a genomic tokenizer."""
    tokenizer = GenomicBPETokenizer(
        vocab_size=vocab_size,
        max_token_length=max_token_length,
    )

    if sequences is not None:
        tokenizer.train_from_sequences(
            sequences,
            min_frequency=min_frequency,
        )

    return tokenizer


class GenomicCharTokenizer(PreTrainedTokenizerFast):
    """Character-level tokenizer for genomic sequences."""

    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"

    DNA_NUCLEOTIDES = ["A", "C", "G", "T", "N"]

    def __init__(
        self,
        tokenizer_object: Optional[Tokenizer] = None,
        **kwargs,
    ):
        """Initialize the character-level genomic tokenizer."""
        if tokenizer_object is None and "tokenizer_file" not in kwargs:
            tokenizer_object = self._create_char_tokenizer()

        kwargs.setdefault("pad_token", self.PAD_TOKEN)
        kwargs.setdefault("eos_token", self.EOS_TOKEN)
        kwargs.setdefault("bos_token", self.BOS_TOKEN)

        super().__init__(
            tokenizer_object=tokenizer_object,
            **kwargs,
        )

    def _create_char_tokenizer(self) -> Tokenizer:
        """Create a character-level tokenizer with fixed vocabulary."""
        vocab = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
        }
        for i, nuc in enumerate(self.DNA_NUCLEOTIDES):
            vocab[nuc] = i + 3

        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="N"))

        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Replace("a", "A"),
            normalizers.Replace("c", "C"),
            normalizers.Replace("g", "G"),
            normalizers.Replace("t", "T"),
            normalizers.Replace("n", "N"),
        ])

        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern="",
            behavior="isolated",
        )

        tokenizer.decoder = decoders.Fuse()

        tokenizer.post_processor = TemplateProcessing(
            single=f"{self.BOS_TOKEN}:0 $A:0 {self.EOS_TOKEN}:0",
            pair=f"{self.BOS_TOKEN}:0 $A:0 {self.EOS_TOKEN}:0 {self.BOS_TOKEN}:1 $B:1 {self.EOS_TOKEN}:1",
            special_tokens=[
                (self.BOS_TOKEN, vocab[self.BOS_TOKEN]),
                (self.EOS_TOKEN, vocab[self.EOS_TOKEN]),
            ],
        )

        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size (8 = 3 special + 5 nucleotides)."""
        return 8

    def __repr__(self) -> str:
        return f"GenomicCharTokenizer(vocab_size={self.vocab_size})"
