"""Pick one sequence per species for training the tokenizer.

    python subsample_bpe_sequences.py --metadata seq_metadata_subsampled.csv --out seq_metadata_subsampled_bpe.csv

--metadata lists the pretraining sequences (columns seq_id, source, Species, ...).
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--metadata", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

df = pd.read_csv(args.metadata, keep_default_na=False)
one = df.groupby("Species", sort=False).sample(n=1, random_state=args.seed)
one.sample(frac=1, random_state=args.seed).reset_index(drop=True).to_csv(args.out, index=False)
print(f"{len(one):,} sequences, one per species, -> {args.out}")
