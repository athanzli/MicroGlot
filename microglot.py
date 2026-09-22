"""Minimal helper for using MicroGlot."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["MicroGlot"]

REPO = "athanzli/MicroGlot"
CONTEXT = 8192


def _fetch(repo_or_path: str, filename: str) -> Path:
    """Return a local path for `filename`, downloading from the Hub if needed."""
    local = Path(repo_or_path) / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_or_path, filename=filename))


class MicroGlot:
    """MicroGlot with optional species conditioning."""

    def __init__(self, model, tokenizer, species_table=None):
        self.model = model
        self.tokenizer = tokenizer
        self._table = species_table
        self._loose = None

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str = REPO,
        variant: str = "species",
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MicroGlot":
        """Load MicroGlot."""
        if variant not in ("species", "plain"):
            raise ValueError(f'variant must be "species" or "plain", got {variant!r}')
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        subfolder = "" if variant == "species" else "plain"

        kwargs = {"trust_remote_code": True}
        if subfolder:
            kwargs["subfolder"] = subfolder
        model = AutoModelForCausalLM.from_pretrained(repo_or_path, **kwargs)
        model.to(device=device, dtype=dtype).eval()
        tokenizer = AutoTokenizer.from_pretrained(repo_or_path, trust_remote_code=True)

        table = None
        if variant == "species":
            blob = torch.load(
                _fetch(repo_or_path, "species/species_embeddings.pt"),
                map_location="cpu",
                weights_only=False,
            )
            table = blob
        return cls(model, tokenizer, table)

    @property
    def species_names(self) -> list[str]:
        """The 99,700 species with a precomputed taxonomy embedding."""
        return list(self._table["names"]) if self._table else []

    @staticmethod
    def _key(name: str) -> str:
        """Loose matching key: case, underscores, hyphens and spacing are ignored."""
        return " ".join(str(name).replace("_", " ").replace("-", " ").lower().split())

    def _loose_index(self) -> dict:
        if self._loose is None:
            self._loose = {}
            for i, n in enumerate(self._table["names"]):
                self._loose.setdefault(self._key(n), i)
        return self._loose

    def resolve_species(self, name: str) -> str:
        """Canonical species name for `name`, matched loosely."""
        if not self._table:
            raise RuntimeError("the plain model has no species table")
        if name in self._table["name_to_idx"]:
            return name
        idx = self._loose_index().get(self._key(name))
        if idx is not None:
            return self._table["names"][idx]
        genus = self._key(name).split(" ")[0]
        near = [n for n in self._table["names"] if self._key(n).startswith(genus)][:5]
        raise KeyError(
            f"{name!r} is not among the 99,700 species with a precomputed embedding."
            + (f" Did you mean one of: {near}?" if near else "")
            + " Otherwise omit `species=` and the built-in species encoder will infer"
            " the embedding from the sequence."
        )

    def has_species(self, name: str) -> bool:
        """True if `name` resolves to a species in the table (matched loosely)."""
        if not self._table:
            return False
        return (name in self._table["name_to_idx"]
                or self._key(name) in self._loose_index())

    def species_embedding(self, name: str) -> torch.Tensor:
        """The unit-norm 32-d Poincare embedding for `name`, matched loosely."""
        canonical = self.resolve_species(name)
        return self._table["embeddings"][self._table["name_to_idx"][canonical]]

    def taxonomy(self, repo_or_path: str = REPO):
        """The lineage table for the 99,700 species."""
        path = _fetch(repo_or_path, "species/species_taxonomy.tsv.gz")
        try:
            import pandas as pd
        except ImportError:
            import csv

            with gzip.open(path, "rt") as fh:
                return list(csv.DictReader(fh, delimiter="\t"))
        with gzip.open(path, "rt") as fh:
            return pd.read_csv(fh, sep="\t", dtype=str, keep_default_na=False)

    def _encode(self, sequences: Sequence[str]):
        batch = self.tokenizer(
            [s.strip().upper() for s in sequences],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONTEXT,
        )
        dev = next(self.model.parameters()).device
        keep = ("input_ids", "attention_mask")
        return {k: v.to(dev) for k, v in batch.items() if k in keep}

    def _species_tensor(self, species, n: int, dev, dtype=None):
        if species is None:
            return None
        names = [species] * n if isinstance(species, str) else list(species)
        if len(names) != n:
            raise ValueError(f"got {len(names)} species for {n} sequences")
        vecs = torch.stack([self.species_embedding(s) for s in names])
        return vecs.to(device=dev, dtype=torch.float32)

    @torch.no_grad()
    def hidden_states(
        self,
        sequences: str | Iterable[str],
        species: str | Sequence[str] | None = None,
        species_emb: torch.Tensor | None = None,
    ):
        """The 23 decoder layer outputs, one tensor per layer."""
        if isinstance(sequences, str):
            sequences = [sequences]
        sequences = list(sequences)
        batch = self._encode(sequences)
        dtype = next(self.model.parameters()).dtype
        if species_emb is None:
            species_emb = self._species_tensor(
                species, len(sequences), batch["input_ids"].device
            )
        elif species is not None:
            raise ValueError("pass either `species` or `species_emb`, not both")
        else:
            species_emb = torch.as_tensor(species_emb).to(
                device=batch["input_ids"].device, dtype=torch.float32
            )
        backbone = getattr(self.model, "model", self.model)
        out = backbone(
            **batch,
            species_emb=species_emb,
            output_hidden_states=True,
            return_dict=True,
        )
        return tuple(out.hidden_states)[1:], batch["attention_mask"]

    @torch.no_grad()
    def embed(
        self,
        sequences: str | Iterable[str],
        species: str | Sequence[str] | None = None,
        species_emb: torch.Tensor | None = None,
        layer: int = -1,
    ) -> torch.Tensor:
        """A [batch, 1024] mean-pooled embedding, masked to real tokens."""
        states, mask = self.hidden_states(sequences, species, species_emb)
        n = len(states)
        if layer == 0 or layer > n or layer < -n:
            raise ValueError(
                f"layer must be a decoder layer number in 1..{n} "
                f"(or -1..-{n} counting from the last), got {layer}"
            )
        h = states[layer - 1 if layer > 0 else layer].float()
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)
