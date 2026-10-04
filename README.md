# MicroGlot

A taxonomy-informed sparse DNA foundation model for microbial genomics.

MicroGlot is a 23-layer decoder-only mixture-of-experts transformer pretrained on **378.3 billion
nucleotides** from **3.70 million sequences** across **99,700 microbial species**, spanning bacteria, archaea,
fungi, protists, viruses and plasmids. It encodes the taxonomic hierarchy as hyperbolic (Poincaré)
embeddings and uses them both as an input token and to steer expert routing.

**Paper:** [A Taxonomy-Informed Sparse DNA Foundation Model for Microbial Genomics](https://www.biorxiv.org/content/10.64898/2026.09.22.753215v2)\
**Models:** [huggingface.co/athanzli/MicroGlot](https://huggingface.co/athanzli/MicroGlot)

## Models

| Model | Input | Use it when | Load with |
|---|---|---|---|
| **MicroGlot** | DNA and its species | most of your sequences have a known species | `from_pretrained("athanzli/MicroGlot", ...)` |
| **MicroGlot-plain** | DNA | most of your sequences have no known species | `from_pretrained("athanzli/MicroGlot", subfolder="plain", ...)` |

A species is known if it is one of the 99,700 pretraining species (check with `tokenizer.has_species(name)`).
For tasks that predict taxonomy, use MicroGlot-plain, because giving the model the species would reveal the answer.

## Model details

| | |
|---|---|
| Architecture | decoder-only transformer, next-token prediction |
| Layers / hidden size | 23 / 1024 |
| Parameters | 2.98 B, of which 479 M are active per token |
| Mixture of experts | 312 routed experts across layers (U-shaped), top-1 routing plus a shared expert |
| Species conditioning | MicroGlot uses a 32-d Poincaré embedding as an input token and to modulate expert routing, and MicroGlot-plain uses none |
| Context | 8,192 tokens (about 43 kb) |
| Tokenizer | byte-pair encoding, vocabulary 8,192 |

## Installation

MicroGlot is tested on Linux with an NVIDIA GPU and requires [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)
(`flash-attn`), whose rotary position-embedding kernel it was trained with.

```bash
conda create -n microglot python=3.12 -y
conda activate microglot
pip install torch==2.8.0 "transformers>=4.51.3,<5.19"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Tested with Python 3.10 to 3.13, PyTorch 2.7 to 2.13, transformers 4.51.3 to 5.18 and flash-attn 2.7.4 to 2.8.3.post1.
For another Python or PyTorch version, install the matching flash-attn wheel from the
[flash-attn releases](https://github.com/Dao-AILab/flash-attention/releases).

## Usage

### MicroGlot

```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("athanzli/MicroGlot", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "athanzli/MicroGlot", trust_remote_code=True, dtype=torch.bfloat16
).to("cuda")

sequences = ["ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCC", "TTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGC"]
species = ["Escherichia coli", "Bacillus subtilis"]

inputs = tokenizer(sequences, species=species, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

last_layer = outputs.last_hidden_state   # [2, length, 1024]
```

- `species=` takes one name per sequence, or one name for all of them. Names are NCBI Taxonomy scientific
  names (September 2025), e.g. *Clostridioides difficile*. Case and extra spaces do not matter, and `_` or
  `-` count as spaces. An unknown name raises a `KeyError` listing similarly spelled names; check that a
  suggestion is the same organism.
- `outputs.hidden_states[k]` is the output of decoder layer k (1 to 22); `[0]` holds the token
  embeddings, and `[23]` is `last_hidden_state`, the output of layer 23 after the final normalization.
  All outputs line up with `input_ids`; padded positions have `attention_mask` 0.

### Sequences without a known species

If a small portion of your sequences have no known species, consider discarding them, so that every
remaining sequence is given its exact taxonomy embedding. To keep them instead, use the Species-encoder
to infer their species embeddings from the DNA and fill these gaps. Continuing the MicroGlot example,
pass `None` as their species.

```python
model.load_species_encoder()   # downloads the Species-encoder (6 GB) and attaches it to MicroGlot

inputs = tokenizer(sequences, species=["Escherichia coli", None], padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)   # the second sequence's species is inferred
```

If most of your sequences have no known species, using the Species-encoder to infer their species is not
recommended due to performance considerations. We recommend using MicroGlot-plain instead.

### Long sequences

The context is 8,192 tokens (about 43 kb). To encode a longer sequence, one viable way is "chunk and
encode", by cutting the sequence into windows that fit the context and encoding each window. Continuing
the MicroGlot example, the tokenizer does the chunking.

```python
genome = "ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCC" * 3000   # stand-in for a 114 kb sequence
windows = tokenizer(
    genome,
    species="Escherichia coli",        # one species, repeated for every window
    truncation=True, max_length=8192,  # at most 8,192 tokens per window, [BOS] and [EOS] included
    return_overflowing_tokens=True,    # return every window (consecutive, non-overlapping), not only the first
    padding=True,                      # pad the last, shorter window with [PAD]
    return_tensors="pt",               # PyTorch tensors, one row per window
)
windows.pop("overflow_to_sample_mapping")   # which input each window came from; not a model input

with torch.no_grad():
    for i in range(0, len(windows["input_ids"]), 4):        # 4 windows at a time
        batch = {k: v[i:i + 4].to("cuda") for k, v in windows.items()}
        window_states = model(**batch).last_hidden_state     # [windows, 8192, 1024]; use them before the next batch
```

### MicroGlot-plain

```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("athanzli/MicroGlot", subfolder="plain", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "athanzli/MicroGlot", subfolder="plain", trust_remote_code=True, dtype=torch.bfloat16
).to("cuda")

sequences = ["ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCC", "TTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGC"]

inputs = tokenizer(sequences, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

last_layer = outputs.last_hidden_state   # [2, length, 1024]
```

MicroGlot-plain takes no species; everything else works as for MicroGlot.

## Citation

> Li, A. Z., Wang, S., Cheng, S., Du, Y. & Liu, R. A Taxonomy-Informed Sparse DNA Foundation Model for Microbial Genomics. *bioRxiv* (2026). https://doi.org/10.64898/2026.09.22.753215

```bibtex
@article{li2026microglot,
  author  = {Li, Athan Z. and Wang, Shiyuan and Cheng, Shupeng and Du, Yuxuan and Liu, Ruishan},
  title   = {A Taxonomy-Informed Sparse {DNA} Foundation Model for Microbial Genomics},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.22.753215},
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.22.753215v2}
}
```
