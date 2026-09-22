#!/usr/bin/env python3
"""Build the species-name to Poincare-embedding lookup table from a fitted checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

SPECIES_PREFIX = "Species:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the species-name -> Poincare embedding lookup table")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Fitted Poincare checkpoint, e.g. species_poincare_32.pth.best, "
             "as written by compute_poincare_embeddings.py")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destination .pt file for the {names, name_to_idx, embeddings} "
             "lookup table (parent directories are created)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"No Poincare checkpoint at {args.checkpoint}. Fit one first with "
            f"compute_poincare_embeddings.py --dim 32.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading Poincare checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    objects = state["objects"]
    emb = state["embeddings"]
    print(f"  {len(objects):,} nodes; embeddings {tuple(emb.shape)} {emb.dtype}")

    species_idx = [i for i, o in enumerate(objects) if o.startswith(SPECIES_PREFIX)]
    names = [objects[i][len(SPECIES_PREFIX):] for i in species_idx]
    species_emb = F.normalize(emb[species_idx].float(), dim=-1)
    print(f"  Species nodes: {len(names):,}; normalised emb {tuple(species_emb.shape)}")

    name_to_idx: dict[str, int] = {}
    for i, n in enumerate(names):
        name_to_idx.setdefault(n, i)

    payload = {"names": names, "name_to_idx": name_to_idx, "embeddings": species_emb}
    torch.save(payload, args.output)
    n = species_emb.norm(dim=-1)
    print(f"\nWrote {args.output}")
    print(f"  {len(name_to_idx):,} unique species; "
          f"norm[min/med/max]={n.min():.4f}/{n.median():.4f}/{n.max():.4f}")
    print(f"  sample: {names[:3]}")


if __name__ == "__main__":
    main()
