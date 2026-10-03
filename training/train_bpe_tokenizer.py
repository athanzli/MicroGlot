"""Train MicroGlot's BPE tokenizer.

    python train_bpe_tokenizer.py --sequences seq_metadata_subsampled_bpe.csv --parquet-root parquet --out tokenizer

--sequences is the output of subsample_bpe_sequences.py (columns seq_id, source); --parquet-root holds one folder of
parquet shards (columns seq_id, sequence) per source. MicroGlot's tokenizer was trained with the defaults: vocabulary
8,192, every sequence longer than 100 kb cut to one random 100-kb window (seed 42), minimum merge frequency 0, no
maximum token length.
"""
import argparse
import os
import random
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast

from tokenizer import GenomicBPETokenizer


def load_sequences(sequences_csv, parquet_root):
    """The listed sequences, read source by source and shard by shard."""
    ids = pd.read_csv(sequences_csv, keep_default_na=False, usecols=["seq_id", "source"])
    seqs = []
    for source, wanted in ids.groupby("source")["seq_id"].apply(set).items():
        remaining = set(wanted)
        for shard in sorted(Path(parquet_root, source).glob("*.parquet")):
            table = pq.read_table(shard, columns=["seq_id", "sequence"])
            hit = table.filter(pc.is_in(table["seq_id"], value_set=pa.array(list(remaining))))
            seqs += hit.column("sequence").to_pylist()
            remaining -= set(hit.column("seq_id").to_pylist())
            if not remaining:
                break
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--parquet-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-subseq-len", type=int, default=100_000)
    ap.add_argument("--min-freq", type=int, default=0)
    ap.add_argument("--max-token-length", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seqs = load_sequences(args.sequences, args.parquet_root)
    rng = random.Random(args.seed)
    for i, s in enumerate(seqs):
        if len(s) > args.max_subseq_len:
            start = rng.randint(0, len(s) - args.max_subseq_len)
            seqs[i] = s[start:start + args.max_subseq_len]
    print(f"{len(seqs):,} sequences, {sum(map(len, seqs)):,} bp", flush=True)

    tok = GenomicBPETokenizer(vocab_size=args.vocab_size, max_token_length=args.max_token_length)
    tok.train_from_sequences(seqs, min_frequency=args.min_freq)
    os.makedirs(args.out, exist_ok=True)
    PreTrainedTokenizerFast(tokenizer_object=tok.backend_tokenizer, pad_token="[PAD]", bos_token="[BOS]",
                            eos_token="[EOS]").save_pretrained(args.out)  # loads with AutoTokenizer
    print(f"vocabulary {len(tok.get_vocab()):,} -> {args.out}")


if __name__ == "__main__":
    main()
