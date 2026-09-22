"""Minimal MicroGlot example."""

import torch

from microglot import MicroGlot

DNA = (
    "ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATGTT"
    "AATGGGCACAAATTTTCTGTCAGTGGAGAGGGTGAAGGTGATGCAACATACGGAAAACTTACCCTT"
)

model = MicroGlot.from_pretrained("athanzli/MicroGlot")

known = model.embed(DNA, species="Escherichia coli")

inferred = model.embed(DNA)

cos = torch.nn.functional.cosine_similarity(known, inferred).item()
print(f"embedding shape            : {tuple(known.shape)}")
print(f"cos(known prior, inferred) : {cos:.4f}")

other = model.embed(DNA, species="Saccharomyces cerevisiae")
print(f"cos(E. coli, S. cerevisiae): "
      f"{torch.nn.functional.cosine_similarity(known, other).item():.4f}")

states, _ = model.hidden_states(DNA, species="Escherichia coli")
print(f"hidden states              : {len(states)} x {tuple(states[-1].shape)}")

plain = MicroGlot.from_pretrained("athanzli/MicroGlot", variant="plain")
print(f"plain embedding shape      : {tuple(plain.embed(DNA).shape)}")
