# MicroGlot

A taxonomy-informed sparse DNA foundation model for microbial genomics.

MicroGlot is a 23-layer decoder-only mixture-of-experts transformer pretrained on **378.3 billion
nucleotides** from **3.70 million sequences** across **99,700 microbial species** — bacteria, archaea,
fungi, protists, viruses and plasmids. It encodes the taxonomic hierarchy as hyperbolic (Poincaré)
embeddings learned independently of the language-modelling objective, and uses them both as an input
token and to steer expert routing.

For details, see [our manuscript: A Taxonomy-Informed Sparse DNA Foundation Model for Microbial Genomics](https://www.biorxiv.org/content/10.64898/2026.09.22.753215v1).

**Model weights, the tokenizer and the species assets live on Hugging Face:**
[huggingface.co/athanzli/MicroGlot](https://huggingface.co/athanzli/MicroGlot)

This repository holds the source code. It deliberately contains no large binaries.

## Install

```bash
git clone https://github.com/athanzli/MicroGlot.git
cd MicroGlot
pip install -r requirements.txt
```

`flash-attn` is optional. Install it for the fastest rotary kernel; without it the model falls back to
an equivalent pure-PyTorch implementation.

## Quickstart

Weights are fetched from Hugging Face on first use.

```python
from microglot import MicroGlot

model = MicroGlot.from_pretrained("athanzli/MicroGlot")

dna = "ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGT"

# species known -> use its precomputed taxonomy embedding
emb = model.embed(dna, species="Escherichia coli")      # [1, 1024]

# species names are resolved loosely; the following are equivalent
emb = model.embed(dna, species="escherichia_coli")
emb = model.embed(dna, species="ESCHERICHIA-COLI")

# species unknown -> the built-in encoder infers one from the sequence
emb = model.embed(dna)

# no species information at all
plain = MicroGlot.from_pretrained("athanzli/MicroGlot", variant="plain")
emb = plain.embed(dna)
```

Per-layer representations, for probing:

```python
states, mask = model.hidden_states(dna, species="Escherichia coli")
len(states)          # 23, one per decoder layer: states[0] is layer 1, states[-1] layer 23

emb = model.embed(dna, species="Escherichia coli", layer=11)   # decoder layer 11
```

`python example.py` runs a short end-to-end demonstration.

Full usage documentation, including the species assets and the standard `transformers` interface, is
in the [model card](https://huggingface.co/athanzli/MicroGlot).

## Repository contents

| Path | Description |
|---|---|
| `microglot.py` | User-facing helper: loading, species resolution, embeddings, per-layer states |
| `modeling_microglot.py` | Reference implementation of the architecture (MoE, FiLM-modulated routing, species conditioning) |
| `example.py` | Minimal end-to-end example |
| `taxonomy/compute_poincare_embeddings.py` | Fits the hyperbolic taxonomy embeddings on the taxonomy tree |
| `taxonomy/build_species_lookup.py` | Builds the released species lookup table from fitted embeddings |
| `training/tokenizer.py` | Byte-pair-encoding tokenizer used for pretraining |
| `benchmarks/baselines.py` | Feature extractors for the baseline models evaluated in the paper |

`microglot.py`, `modeling_microglot.py` and `example.py` are identical to the copies distributed with
the model on Hugging Face.

### Reproducing the taxonomy embeddings

`taxonomy/compute_poincare_embeddings.py` builds the parent–child relation set from a taxonomy table
and fits Poincaré embeddings with
[facebookresearch/poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings),
which must be cloned separately. `taxonomy/build_species_lookup.py` then converts a fitted checkpoint
into the unit-norm lookup table the model consumes. Both scripts take all inputs and outputs as
command-line arguments; run either with `--help`.

### Baselines

`benchmarks/baselines.py` provides the frozen-embedding extractors for the baseline models compared
against MicroGlot: ProkBERT-mini-long, three Nucleotide Transformer multispecies checkpoints,
DNABERT-2, DNABERT-S and Evo2. All checkpoints are downloaded from their public sources; none is
bundled here. Evo2 additionally requires the `evo2` and `vortex` packages, and a CUDA GPU is required
throughout, since the encoders run under CUDA autocast.

## Licence

Source code in this repository is released under the [MIT License](LICENSE).
The model weights and species assets on Hugging Face are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use MicroGlot, please cite [our manuscript: A Taxonomy-Informed Sparse DNA Foundation Model for Microbial Genomics](https://www.biorxiv.org/content/10.64898/2026.09.22.753215v1):

> Li, A. Z., Wang, S., Cheng, S., Du, Y. & Liu, R. A Taxonomy-Informed Sparse DNA Foundation Model for Microbial Genomics. *bioRxiv* (2026). https://doi.org/10.64898/2026.09.22.753215

```bibtex
@article{li2026microglot,
  author  = {Li, Athan Z. and Wang, Shiyuan and Cheng, Shupeng and Du, Yuxuan and Liu, Ruishan},
  title   = {A Taxonomy-Informed Sparse {DNA} Foundation Model for Microbial Genomics},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.22.753215},
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.22.753215v1}
}
```
