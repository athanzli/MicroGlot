#!/usr/bin/env python3
"""Fit a Poincare (hyperbolic) embedding of the microbial taxonomy tree."""

import argparse
import glob as glob_mod
import os
import subprocess
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
import torch as th

TAX_COLS = [
    "Species", "Genus", "Family", "Order", "Class",
    "Phylum", "Kingdom", "Domain_realm", "Domain_or_Viruses", "Life",
]


def build_relations_csv(taxonomy_csv, csv_out):
    """Build transitive closure CSV (id1, id2, weight) for embed.py."""
    print(f"Loading taxonomy tree from {taxonomy_csv} ...")
    df = pd.read_csv(taxonomy_csv, keep_default_na=False)
    print(f"  Species: {len(df):,}")

    print("Generating transitive closure relations ...")
    relations = set()
    for row in df[TAX_COLS].itertuples(index=False):
        path = []
        for col_name, val in zip(TAX_COLS, row):
            if val != "":
                path.append(f"{col_name}:{val.strip()}")
        for child, parent in combinations(path, 2):
            relations.add((child, parent))

    rel_df = pd.DataFrame(list(relations), columns=["id1", "id2"])
    rel_df["weight"] = 1
    rel_df.to_csv(csv_out, index=False)
    print(f"  Generated {len(rel_df):,} unique relations")
    print(f"  Saved to {csv_out}")


def extract_embeddings(checkpoint, dim, result_dir):
    """Load the best checkpoint and save embeddings as .npz files."""
    best_path = checkpoint + ".best"
    ckpt_path = best_path if os.path.exists(best_path) else checkpoint

    if not os.path.exists(ckpt_path):
        epoch_files = sorted(
            glob_mod.glob(f"{checkpoint}.*"),
            key=lambda p: int(p.rsplit(".", 1)[-1])
            if p.rsplit(".", 1)[-1].isdigit() else -1,
        )
        epoch_files = [f for f in epoch_files if f.rsplit(".", 1)[-1].isdigit()]
        if epoch_files:
            ckpt_path = epoch_files[-1]
        else:
            print("ERROR: No checkpoint found after training.")
            sys.exit(1)

    print(f"\nLoading checkpoint from {ckpt_path} ...")
    chkpnt = th.load(ckpt_path, map_location="cpu", weights_only=False)
    embeddings = chkpnt["embeddings"]
    objects = chkpnt["objects"]

    if hasattr(objects, "tolist"):
        objects = objects.tolist()

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Total nodes: {len(objects)}")

    all_npz = f"{result_dir}/species_poincare_{dim}_embeddings.npz"
    np.savez(
        all_npz,
        embeddings=embeddings.numpy(),
        objects=np.array(objects),
    )
    print(f"  All embeddings saved to {all_npz}")

    species_mask = np.array([o.startswith("Species:") for o in objects])
    species_idx = np.where(species_mask)[0]
    species_embeddings = embeddings[species_idx].numpy()
    species_names = np.array([objects[i].replace("Species:", "") for i in species_idx])

    species_npz = f"{result_dir}/species_poincare_{dim}_species_only.npz"
    np.savez(
        species_npz,
        embeddings=species_embeddings,
        species=species_names,
    )
    print(f"  Species-only embeddings saved to {species_npz}")
    print(f"    Shape: {species_embeddings.shape}")
    print(f"    Species count: {len(species_names)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Poincare embedding of species taxonomy tree")
    parser.add_argument("--taxonomy-csv", required=True,
                        help="Input taxonomy table (CSV, one row per species, "
                             "columns: " + ", ".join(TAX_COLS) + ")")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for the relations CSV, checkpoints and "
                             ".npz embedding files (created if missing)")
    parser.add_argument("--poincare-repo", required=True,
                        help="Path to a clone of facebookresearch/poincare-embeddings; "
                             "its embed.py performs the optimisation")
    parser.add_argument("--relations-csv", default=None,
                        help="Where to write the (id1,id2,weight) edge list "
                             "(default: <output-dir>/species_taxonomy_relations.csv)")
    parser.add_argument("--dim", type=int, required=True,
                        help="Embedding dimension (32 for the released MicroGlot "
                             "species embeddings)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id (-1 for CPU, default: 0)")
    parser.add_argument("--epochs", type=int, default=1500,
                        help="Training epochs (default: 1500, the setting used "
                             "for the released embeddings)")
    parser.add_argument("--skip-relations", action="store_true",
                        help="Skip building relations CSV (reuse existing)")
    args = parser.parse_args()

    dim = args.dim
    result_dir = os.path.abspath(args.output_dir)
    poincare_repo = os.path.abspath(args.poincare_repo)
    csv_out = (os.path.abspath(args.relations_csv) if args.relations_csv
               else os.path.join(result_dir, "species_taxonomy_relations.csv"))
    embed_py = os.path.join(poincare_repo, "embed.py")
    checkpoint = f"{result_dir}/species_poincare_{dim}.pth"

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    if not os.path.exists(embed_py):
        print(f"ERROR: embed.py not found at {embed_py}. Pass the root of a "
              f"poincare-embeddings clone with --poincare-repo.")
        sys.exit(1)

    if args.skip_relations and os.path.exists(csv_out):
        print(f"Reusing existing relations CSV: {csv_out}")
    else:
        build_relations_csv(args.taxonomy_csv, csv_out)

    cmd = [
        sys.executable, embed_py,
        "-checkpoint", checkpoint,
        "-dset", csv_out,
        "-dim", str(dim),
        "-manifold", "poincare",
        "-model", "distance",
        "-lr", "0.02",
        "-lr_type", "scale",
        "-epochs", str(args.epochs),
        "-negs", "50",
        "-burnin", "20",
        "-dampening", "1.0",
        "-ndproc", "4",
        "-eval_each", "100",
        "-fresh",
        "-sparse",
        "-burnin_multiplier", "0.01",
        "-neg_multiplier", "0.1",
        "-batchsize", "1024",
        "-gpu", str(args.gpu),
        "-train_threads", "1",
    ]

    print(f"\nStarting Poincare training (dim={dim}, gpu={args.gpu}, "
          f"epochs={args.epochs}) ...")
    print(f"  Command: {' '.join(cmd)}")
    t_start = time.time()

    result = subprocess.run(cmd, cwd=poincare_repo)

    elapsed = time.time() - t_start
    print(f"\nTraining finished in {elapsed/3600:.2f} hr ({elapsed/60:.1f} min)")

    if result.returncode != 0:
        print(f"ERROR: embed.py exited with code {result.returncode}")
        sys.exit(1)

    extract_embeddings(checkpoint, dim, result_dir)

    print(f"\nTotal wall time: {elapsed/3600:.2f} hr")
    print("Done.")


if __name__ == "__main__":
    main()
