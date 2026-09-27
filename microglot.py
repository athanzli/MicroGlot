"""Minimal helper for using MicroGlot."""

from __future__ import annotations

import difflib
import gzip
import unicodedata
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["MicroGlot"]

REPO = "athanzli/MicroGlot"
CONTEXT = 8192
_DNA = frozenset("ACGTNRYSWKMBDHV")  # ACGTN and IUPAC codes (the tokenizer reads the codes as N)


def _fetch(repo_or_path: str, filename: str) -> Path:
    """Return a local path for `filename`, downloading from the Hub if needed."""
    local = Path(repo_or_path) / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_or_path, filename=filename))


class MicroGlot:
    """MicroGlot with optional species conditioning."""

    def __init__(self, model, tokenizer, species_table=None, source: str = REPO):
        self.model = model
        self.tokenizer = tokenizer
        self._table = species_table
        self._loose = None
        self._source = source

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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                warnings.warn(
                    f"No usable CUDA GPU (torch {torch.__version__}, CUDA build {torch.version.cuda}); "
                    "running MicroGlot on CPU, which is slow. With an NVIDIA GPU, install a CUDA build "
                    "of torch: https://pytorch.org/get-started/locally/",
                    stacklevel=2,
                )
        subfolder = "" if variant == "species" else "plain"

        # torch_dtype: load straight into `dtype` instead of building an fp32 copy in RAM first
        kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        if subfolder:
            kwargs["subfolder"] = subfolder
        model = AutoModelForCausalLM.from_pretrained(repo_or_path, **kwargs)
        model.to(device=device, dtype=dtype).eval()  # also casts the RoPE buffers, as before
        tokenizer = AutoTokenizer.from_pretrained(repo_or_path, trust_remote_code=True)

        table = None
        if variant == "species":
            blob = torch.load(
                _fetch(repo_or_path, "species/species_embeddings.pt"),
                map_location="cpu",
                weights_only=False,
            )
            table = blob
        return cls(model, tokenizer, table, repo_or_path)

    @property
    def species_names(self) -> list[str]:
        """The 99,700 species with a precomputed taxonomy embedding."""
        return list(self._table["names"]) if self._table else []

    @staticmethod
    def _key(name: str) -> str:
        """Loose matching key: case, underscores, hyphens and spacing are ignored."""
        name = unicodedata.normalize("NFC", str(name))
        return " ".join(name.replace("_", " ").replace("-", " ").lower().split())

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
        near = self._suggest(name)
        raise KeyError(
            f"{name!r} is not among the 99,700 species with a precomputed embedding."
            + (f" Did you mean one of: {near}?" if near else "")
            + " Otherwise omit `species=` and the built-in species encoder will infer"
            " the embedding from the sequence."
        )

    def _suggest(self, name: str, n: int = 5) -> list[str]:
        """Closest table names: the species of a strain/subsp./serovar name, 'E. coli' style, then fuzzy."""
        lk, words = self._loose_index(), self._key(name).split(" ")
        out = [" ".join(words[:k]) for k in range(len(words) - 1, 1, -1) if " ".join(words[:k]) in lk]
        if len(words) > 1 and len(words[0].rstrip(".")) == 1:  # abbreviated genus
            out += [k for k in lk if k[0] == words[0][0] and k.split(" ")[1:2] == words[1:2]]
        pool = [k for k in lk if k.split(" ", 1)[0] == words[0]] or list(lk)
        out += difflib.get_close_matches(" ".join(words), pool, n=n, cutoff=0.6)
        return [self._table["names"][lk[k]] for k in dict.fromkeys(out)][:n]

    def has_species(self, name: str) -> bool:
        """True if `name` resolves to a species in the table (matched loosely).

        Always False for MicroGlot-plain, which has no species table."""
        if not self._table:
            return False
        return (name in self._table["name_to_idx"]
                or self._key(name) in self._loose_index())

    def species_embedding(self, name: str) -> torch.Tensor:
        """The unit-norm 32-d Poincare embedding for `name`, matched loosely."""
        canonical = self.resolve_species(name)
        return self._table["embeddings"][self._table["name_to_idx"][canonical]]

    def taxonomy(self, repo_or_path: str | None = None):
        """The lineage table for the 99,700 species (by default from where the model was loaded)."""
        path = _fetch(repo_or_path or self._source, "species/species_taxonomy.tsv.gz")
        try:
            import pandas as pd
        except ImportError:
            import csv

            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return list(csv.DictReader(fh, delimiter="\t"))
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return pd.read_csv(fh, sep="\t", dtype=str, keep_default_na=False)

    @staticmethod
    def _prepare(sequences) -> list[str]:
        """Bare upper-case sequences: FASTA line breaks, carriage returns and spaces are removed."""
        if isinstance(sequences, str):
            sequences = [sequences]
        seqs = ["".join(str(s).split()).upper() for s in sequences]
        bad = [i for i, s in enumerate(seqs) if not _DNA.issuperset(s)]
        if bad:
            s = seqs[bad[0]]
            j = next(j for j, c in enumerate(s) if c not in _DNA)
            warnings.warn(
                f"{len(bad)} of {len(seqs)} sequence(s) contain non-DNA characters (sequence {bad[0]}: "
                f"{s[j]!r} at position {j}); each is read as an N token. Pass bare sequences without "
                "the FASTA '>' header line; for RNA, replace U with T.",
                stacklevel=4,
            )
        return seqs

    def _encode(self, sequences: Sequence[str]):
        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONTEXT,
        )
        # (length, bp kept) of each sequence cut at CONTEXT tokens
        cut = [(len(s), max(end for _, end in e.offsets))
               for s, e in zip(sequences, getattr(batch, "encodings", None) or []) if e.overflowing]
        dev = next(self.model.parameters()).device
        keep = ("input_ids", "attention_mask")
        return {k: v.to(dev) for k, v in batch.items() if k in keep}, cut

    @staticmethod
    def _warn_truncated(cut):
        if cut:
            total, kept = cut[0]
            warnings.warn(
                f"{len(cut)} sequence(s) exceed the {CONTEXT:,}-token context (about 43 kb) and were "
                f"truncated (e.g. only the first {kept:,} of {total:,} bp were used). Split longer "
                "sequences, e.g. into 40 kb pieces, and pool the pieces yourself.",
                stacklevel=4,
            )

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
        """The 23 decoder layer outputs, one tensor per layer (all sequences in one padded batch)."""
        states, mask, cut = self._hidden_states(self._prepare(sequences), species, species_emb)
        self._warn_truncated(cut)
        return states, mask

    def _hidden_states(self, sequences: list[str], species, species_emb):
        if not self._table and (species is not None or species_emb is not None):
            raise ValueError(
                "MicroGlot-plain takes no species information: drop `species=`/`species_emb=`, "
                'or load MicroGlot.from_pretrained(..., variant="species")'
            )
        batch, cut = self._encode(sequences)
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
        return tuple(out.hidden_states)[1:], batch["attention_mask"], cut

    @torch.no_grad()
    def embed(
        self,
        sequences: str | Iterable[str],
        species: str | Sequence[str] | None = None,
        species_emb: torch.Tensor | None = None,
        layer: int = -1,
        batch_size: int | None = 16,
    ) -> torch.Tensor:
        """A [n, 1024] mean-pooled embedding, masked to real tokens.

        Sequences run `batch_size` at a time (None: all in one padded batch), so GPU memory stays
        bounded; results agree across batch sizes up to bfloat16 padding noise.
        """
        sequences = self._prepare(sequences)
        n_seq = len(sequences)
        step = batch_size or max(n_seq, 1)
        if species is not None and not isinstance(species, str):
            species = list(species)
            if len(species) != n_seq:
                raise ValueError(f"got {len(species)} species for {n_seq} sequences")
        if species_emb is not None:
            species_emb = torch.as_tensor(species_emb)
        per_seq = species_emb is not None and species_emb.dim() == 2 and species_emb.shape[0] == n_seq
        parts, cut = [], []
        for i in range(0, n_seq, step):
            sp = species[i:i + step] if isinstance(species, list) else species
            se = species_emb[i:i + step] if per_seq else species_emb
            states, mask, c = self._hidden_states(sequences[i:i + step], sp, se)
            parts.append(self._pool(states, mask, layer))
            cut += c
        self._warn_truncated(cut)
        return torch.cat(parts)

    @staticmethod
    def _pool(states, mask, layer: int) -> torch.Tensor:
        n = len(states)
        if layer == 0 or layer > n or layer < -n:
            raise ValueError(
                f"layer must be a decoder layer number in 1..{n} "
                f"(or -1..-{n} counting from the last), got {layer}"
            )
        h = states[layer - 1 if layer > 0 else layer].float()
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)
