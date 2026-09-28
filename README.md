# MicroGlot

A taxonomy-informed sparse DNA foundation model for microbial genomics.

MicroGlot is a 23-layer decoder-only mixture-of-experts transformer pretrained on **378.3 billion
nucleotides** from **3.70 million sequences** across **99,700 microbial species** — bacteria, archaea,
fungi, protists, viruses and plasmids. It encodes the taxonomic hierarchy as hyperbolic (Poincaré)
embeddings and uses them both as an input token and to steer expert routing.

For details, see our manuscript, [A Taxonomy-Informed Sparse DNA Foundation Model for Microbial Genomics](https://www.biorxiv.org/content/10.64898/2026.09.22.753215v1).

This repository holds the source code. The model weights, tokenizer and species assets are on
Hugging Face: [huggingface.co/athanzli/MicroGlot](https://huggingface.co/athanzli/MicroGlot).

## Install

```bash
git clone https://github.com/athanzli/MicroGlot.git
cd MicroGlot
pip install -r requirements.txt
python example.py   # smoke test; the first run downloads the models (~18 GB)
```

Run your code from this folder, or copy `microglot.py` next to your script or notebook.
A GPU with 16 GB of memory runs either model; without a GPU, MicroGlot runs on the CPU, more slowly.

## Quickstart

```python
from microglot import MicroGlot

model = MicroGlot.from_pretrained("athanzli/MicroGlot")
dna = "ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGT"

# 1. species known: use its taxonomy embedding
emb = model.embed(dna, species="Escherichia coli")   # [1, 1024]

# 2. species unknown: the built-in encoder infers it from the sequence
emb = model.embed(dna)

# 3. no species information: use MicroGlot-plain (on a 16 GB GPU, run `del model` first)
plain = MicroGlot.from_pretrained("athanzli/MicroGlot", variant="plain")
emb = plain.embed(dna)

# a specific decoder layer (1 to 23; intermediate layers often work better than the last one)
emb = model.embed(dna, species="Escherichia coli", layer=11)
```

For embedding your own FASTA files, long sequences, the species assets and the standard
`transformers` interface, see the [model card](https://huggingface.co/athanzli/MicroGlot).

## Repository contents

| Path | Description |
|---|---|
| `microglot.py` | Helper for loading the model and computing embeddings |
| `modeling_microglot.py` | Model architecture |
| `example.py` | Minimal end-to-end example |
| `taxonomy/` | Scripts that fit the Poincaré taxonomy embeddings and build the species lookup table (run with `--help`) |
| `training/tokenizer.py` | Byte-pair-encoding tokenizer used for pretraining |
| `benchmarks/baselines.py` | Embedding extractors for the baseline models evaluated in the paper |

`microglot.py`, `modeling_microglot.py` and `example.py` are identical to the copies on Hugging Face.

## Licence

Source code in this repository is released under the [MIT License](LICENSE).
The model weights and species assets on Hugging Face are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use MicroGlot, please cite our manuscript, [A Taxonomy-Informed Sparse DNA Foundation Model for Microbial Genomics](https://www.biorxiv.org/content/10.64898/2026.09.22.753215v1):

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
