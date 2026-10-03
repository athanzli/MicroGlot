"""MicroGlot tokenizer: the pretraining byte-pair-encoding tokenizer (tokenizer.json) plus a species vocabulary.

`tokenizer(dna, species=...)` adds `species_ids` (one per sequence) next to `input_ids` and `attention_mask`,
in the way NLLB's tokenizer handles `src_lang` and XLM's handles `lang2id`: names are resolved here, and the
model looks the 32-d vectors up in its own table. `None` (or an unknown name with `unknown_species="infer"`)
gives -1, which the model fills in with an attached species encoder.
"""

import difflib
import hashlib
import os
import re
import unicodedata
from typing import List, Optional, Sequence, Union

from transformers import PreTrainedTokenizerFast
from transformers.utils import logging

logger = logging.get_logger(__name__)

UNKNOWN_SPECIES_ID = -1


class MicroGlotTokenizerFast(PreTrainedTokenizerFast):
    """MicroGlot's DNA tokenizer (byte-pair encoding, [BOS] ... [EOS], right padding) with species names.

    Token ids are those of tokenizer.json. `species_vocab.txt` lists the 99,700 pretraining species;
    line i is `species_ids` i and row i of the model's species table. Names match exactly first, then loosely
    (case and extra spaces ignored, `_` and `-` read as spaces, NCBI's genus brackets optional); a MetaPhlAn or
    GTDB name is then matched by its `s__` field ("k__Bacteria|...|s__Escherichia_coli", "s__Escherichia coli").
    """

    vocab_files_names = {"tokenizer_file": "tokenizer.json", "species_vocab_file": "species_vocab.txt"}
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"  # the model reads [BOS] at position 0 and pools the last real token

    # No __init__ override: transformers 5 rebuilds tokenizer subclasses that define __init__ from the vocab alone
    # (no merges, no normalizer) -> character-level ids. The species vocabulary is read lazily instead.
    @property
    def species_vocab_file(self):
        """Path of the species list this tokenizer was loaded with (None if it has none)."""
        return self.init_kwargs.get("species_vocab_file")

    @property
    def id_to_species(self) -> List[str]:
        names = self.__dict__.get("_id_to_species")
        if names is None:
            path, names = self.species_vocab_file, []
            if path is not None and os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    names = [line.rstrip("\n") for line in fh]
            self.__dict__["_id_to_species"] = names
            self.__dict__["_species_to_id"] = {n: i for i, n in enumerate(names)}
        return names

    @property
    def species_to_id(self) -> dict:
        self.id_to_species
        return self.__dict__["_species_to_id"]

    @property
    def species_vocab_sha256(self) -> Optional[str]:
        """sha256 of the species names joined by newlines; equals the model's `config.species_vocab_sha256`."""
        if "_species_sha256" not in self.__dict__:
            names = self.id_to_species
            self.__dict__["_species_sha256"] = (
                hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest() if names else None
            )
        return self.__dict__["_species_sha256"]

    _loose = None

    # -- species vocabulary ---------------------------------------------------------------------
    @property
    def species_names(self) -> List[str]:
        """The species names in table order (a copy)."""
        return list(self.id_to_species)

    @staticmethod
    def _key(name: str) -> str:
        name = unicodedata.normalize("NFC", str(name))
        name = name.replace("[", "").replace("]", "")  # NCBI's provisional genus: "[Clostridium] scindens"
        return " ".join(name.replace("_", " ").replace("-", " ").lower().split())

    @staticmethod
    def _species_field(name: str) -> Optional[str]:
        """The `s__` field of a MetaPhlAn/GTDB-style name, without the prefix ("k__Bacteria|...|s__Escherichia_coli",
        "d__Bacteria;...;s__Escherichia coli", "s__Escherichia coli"); None unless the name has exactly one."""
        species = [f.strip()[3:] for f in re.split(r"[|;]", str(name)) if f.strip().startswith("s__")]
        return species[0] if len(species) == 1 else None

    def _lookup(self, name) -> Optional[int]:
        """Row of `name`: an exact match, else a loose match (`_key`), else the same for its `s__` field
        (`_species_field`); None if none matches."""
        for n in (name, self._species_field(name)):
            if n is None:
                continue
            idx = self.species_to_id.get(n)
            if idx is None:
                idx = self._loose_index().get(self._key(n))
            if idx is not None:
                return idx
        return None

    def _loose_index(self):
        if self._loose is None:
            self._loose = {}
            for i, n in enumerate(self.id_to_species):
                self._loose.setdefault(self._key(n), i)
        return self._loose

    def has_species(self, name: str) -> bool:
        """True if `name` matches a species in the table (see the class docstring)."""
        return self._lookup(name) is not None

    def _suggest(self, name: str, n: int = 5) -> List[str]:
        lk = self._loose_index()
        key = re.sub(r"^([a-z])\.(?=[a-z])", r"\1. ", self._key(self._species_field(name) or name))  # "E.coli" -> "e. coli"
        words = key.split(" ")
        # Candidatus taxa: also try the name with the prefix added or removed
        alt = key[len("candidatus "):] if key.startswith("candidatus ") else "candidatus " + key
        out = [alt] if alt in lk else []
        for w in (words, alt.split(" ")):
            out += [" ".join(w[:k]) for k in range(len(w) - 1, 1, -1) if " ".join(w[:k]) in lk]
        if len(words) > 1 and len(words[0].rstrip(".")) == 1:  # "E. coli"
            out += [k for k in lk if k[0] == words[0][0] and k.split(" ")[1:2] == words[1:2]]
        pool = [k for k in lk if k.split(" ", 1)[0] == words[0]] or list(lk)
        out += difflib.get_close_matches(" ".join(words), pool, n=n, cutoff=0.6)
        return [self.id_to_species[lk[k]] for k in dict.fromkeys(out)][:n]

    def convert_species_to_ids(
        self, species: Union[str, None, Sequence[Optional[str]]], unknown_species: str = "raise"
    ) -> Union[int, List[int]]:
        """Species name(s) -> row(s) of the model's species table: an int for one name (or None), a list for a
        list. `None` or NaN (an empty pandas cell) -> -1 (infer). An unknown name raises `KeyError` with
        suggestions, or with `unknown_species="infer"` becomes -1 (with one warning)."""
        if unknown_species not in ("raise", "infer"):
            raise ValueError(f'unknown_species must be "raise" or "infer", got {unknown_species!r}.')
        if species is None or isinstance(species, (str, float)):  # one name, None, or NaN (an empty pandas cell)
            return self._one_species_id(species, unknown_species)
        return [self._one_species_id(s, unknown_species) for s in species]

    def _one_species_id(self, name: Optional[str], unknown_species: str) -> int:
        if name is None or (isinstance(name, float) and name != name):  # None or NaN (missing in pandas)
            return UNKNOWN_SPECIES_ID
        if not self.id_to_species:
            raise ValueError("This tokenizer has no species vocabulary (MicroGlot-plain takes no species).")
        idx = self._lookup(name)
        if idx is not None:
            return idx
        if unknown_species == "infer":
            logger.warning_once(f"{name!r} is not in the species table; its species will be inferred.")
            return UNKNOWN_SPECIES_ID
        near = self._suggest(name)
        raise KeyError(
            f"{name!r} is not among the {len(self.id_to_species):,} species with a precomputed embedding."
            + (f" Did you mean one of: {near}?" if near else "")
            + " Pass None (or unknown_species='infer') to let the species encoder infer it."
        )

    def convert_ids_to_species(self, ids):
        """Row(s) of the species table -> name(s); -1 -> None. Takes an int, a list or a tensor."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            return None if ids < 0 else self.id_to_species[ids]
        return [self.convert_ids_to_species(i) for i in ids]

    # -- encoding ---------------------------------------------------------------------------------
    def __call__(self, text=None, *args, species=None, unknown_species: str = "raise", **kwargs):
        """Tokenize DNA (all standard tokenizer arguments apply) and, with `species`, add `species_ids`.

        species: one name for all sequences, or one name (or None) per sequence; None leaves out
            `species_ids` (the model then infers every sequence). For a single sequence with
            `return_tensors`, `species_ids` has shape [1]. With `return_overflowing_tokens=True`, each window
            gets its sequence's species (following `overflow_to_sample_mapping`).
        unknown_species: "raise" (default, `KeyError` with suggestions) or "infer" (-1).
        """
        encoding = super().__call__(text, *args, **kwargs)
        if species is None:
            return encoding
        opts = {**kwargs, **(kwargs.get("tokenizer_kwargs") or {})}  # as transformers 5 merges tokenizer_kwargs
        # one sequence or a batch, decided as transformers does: with is_split_into_words, a list of strings is
        # ONE pre-split sequence and a batch is a list of such lists
        if opts.get("is_split_into_words"):
            batched = isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], (list, tuple))
        else:
            batched = isinstance(text, (list, tuple))
        n = len(encoding["input_ids"]) if batched else None
        if n is None:  # a single sequence
            if not isinstance(species, str):
                raise ValueError("Pass one species name for one sequence.")
            ids = self.convert_species_to_ids(species, unknown_species)
            if opts.get("return_overflowing_tokens"):
                ids = [ids] * len(encoding["input_ids"])
            elif opts.get("return_tensors") is not None:
                ids = [ids]  # input_ids is [1, length]: keep species_ids [1]
        else:
            names = [species] * n if isinstance(species, str) else list(species)
            if opts.get("return_overflowing_tokens"):  # windows of long sequences
                mapping = encoding["overflow_to_sample_mapping"]
                names = [names[int(i)] for i in mapping] if len(names) != len(mapping) else names
            if len(names) != len(encoding["input_ids"]):
                raise ValueError(f"Got {len(names)} species for {len(encoding['input_ids'])} sequences.")
            ids = self.convert_species_to_ids(names, unknown_species)
        encoding["species_ids"] = ids
        return_tensors = opts.get("return_tensors")
        if return_tensors is not None:
            encoding.convert_to_tensors(tensor_type=return_tensors)
        return encoding

    # -- saving -----------------------------------------------------------------------------------
    def _save_pretrained(self, save_directory, file_names, legacy_format=None, filename_prefix=None):
        file_names = super()._save_pretrained(save_directory, file_names, legacy_format, filename_prefix)
        if self.id_to_species:
            path = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "species_vocab.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.id_to_species) + "\n")
            file_names = file_names + (path,)
        return file_names


# Let save_pretrained() copy this file and write auto_map, also when the tokenizer was loaded from the Hub.
MicroGlotTokenizerFast.register_for_auto_class("AutoTokenizer")
