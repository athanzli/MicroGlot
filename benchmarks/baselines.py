"""Frozen sequence-embedding extractors for baseline genomic language models."""

import sys
from typing import Any, Dict

import torch
import torch.nn as nn

try:
    torch.backends.cuda.preferred_blas_library("cublaslt")
except Exception:
    pass


MODEL_CONFIGS = {
    "DNABERT-S": {
        "model_name": "zhihan1996/DNABERT-S",
        "max_length": 512,
        "pooling": "mean",
    },
    "DNABERT-2": {
        "model_name": "zhihan1996/DNABERT-2-117M",
        "max_length": 512,
        "pooling": "mean",
    },
    "NT-v2-250M": {
        "model_name": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        "max_length": 2048,
        "pooling": "mean",
    },
    "NT-v2-500M": {
        "model_name": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "max_length": 2048,
        "pooling": "mean",
    },
    "NT-2.5B-multi-species": {
        "model_name": "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        "max_length": 1000,
        "pooling": "mean",
    },
    "ProkBERT-mini-long": {
        "model_name": "neuralbioinfo/prokbert-mini-long",
        "max_length": 2048,
        "pooling": "mean",
    },
    "Evo2-7B-1M-ml8192": {
        "model_name": "evo2_7b",
        "max_length": 8192,
        "pooling": "mean",
    },
}


def _validate_model_config(model_name: str, config, expected_configs: Dict[str, Dict],
                           config_key: str = "max_position_embeddings") -> None:
    """Warn if a downloaded checkpoint no longer matches the expected config."""
    import warnings

    if model_name not in expected_configs:
        warnings.warn(f"Unknown model: {model_name}. Cannot validate config.")
        return

    expected = expected_configs[model_name]
    mismatches = []

    for key, expected_val in expected.items():
        actual_val = getattr(config, key, None)
        if actual_val != expected_val:
            mismatches.append(f"{key}: expected {expected_val}, got {actual_val}")

    if mismatches:
        warnings.warn(
            f"Model config mismatch for {model_name}! "
            f"The model may have been updated. Mismatches: {'; '.join(mismatches)}. "
            f"Please update MODEL_CONFIGS in baselines.py."
        )
    else:
        primary_val = getattr(config, config_key, "N/A")
        print(f"    Config validated: {config_key}={primary_val}")


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.float().cpu()


def _needs_bf16_autocast(device: str) -> bool:
    """Return True if bfloat16 autocast is needed on this device."""
    if device == "cpu" or not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability()
    return cc[0] >= 12


def tokenize_full_sequence(sequence: str, tokenizer) -> list:
    """Tokenize an arbitrarily long sequence into token IDs without truncation."""
    if hasattr(tokenizer, 'encode') and callable(tokenizer.encode):
        return tokenizer.encode(sequence, add_special_tokens=False)
    result = tokenizer(sequence, truncation=False)
    ids = result["input_ids"]
    if isinstance(ids, torch.Tensor):
        return ids[0].tolist()
    return list(ids[0])


import re as _re

def _sanitize_dna(sequence: str) -> str:
    """Replace non-ACGT characters with random valid nucleotides."""
    if not _re.search(r'[^ACGT]', sequence):
        return sequence
    import random
    return ''.join(
        c if c in 'ACGT' else random.choice('ACGT')
        for c in sequence
    )


def chunk_and_encode(
    sequence: str,
    extractor,
    tokenizer,
    max_length: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode a sequence, chunking it if it is longer than max_length."""
    sequence = _sanitize_dna(sequence)

    all_ids = tokenize_full_sequence(sequence, tokenizer)

    if len(all_ids) <= max_length:
        encoded = tokenizer(sequence, truncation=True, max_length=max_length,
                            return_tensors="pt")
        encoded = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in encoded.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb = extractor.encode(encoded["input_ids"], encoded["attention_mask"])
        result = emb[0].cpu()
        if torch.isnan(result).any():
            raise FloatingPointError(
                f"NaN in single-pass embedding (input length {len(all_ids)} tokens, "
                f"extractor {type(extractor).__name__}). Refusing to silently cache NaN."
            )
        return result

    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    wrap_with_special = bos_id is not None and eos_id is not None
    if wrap_with_special:
        content_per_chunk = max_length - 2
    else:
        content_per_chunk = max_length

    chunk_list = []
    for i in range(0, len(all_ids), content_per_chunk):
        content_ids = all_ids[i:i + content_per_chunk]
        chunk_list.append(([bos_id] + list(content_ids) + [eos_id]) if wrap_with_special
                          else list(content_ids))

    import os as _os_cb
    chunk_bs = max(1, int(_os_cb.environ.get("CHUNK_BATCH", "24"))) \
        if getattr(extractor, "supports_chunk_batching", False) else 1
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0

    embeddings = []
    for b in range(0, len(chunk_list), chunk_bs):
        batch = chunk_list[b:b + chunk_bs]
        Lmax = max(len(c) for c in batch)
        input_ids = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), Lmax), dtype=torch.long)
        for r, c in enumerate(batch):
            input_ids[r, :len(c)] = torch.tensor(c, dtype=torch.long)
            attention_mask[r, :len(c)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb = extractor.encode(input_ids, attention_mask)
        emb_cpu = emb.cpu()
        if torch.isnan(emb_cpu).any():
            raise FloatingPointError(
                f"NaN in batched chunk embedding (batch start {b}, "
                f"extractor {type(extractor).__name__}). Refusing to silently cache NaN."
            )
        embeddings.append(emb_cpu)

    return torch.cat(embeddings, dim=0).mean(dim=0)


class BaseExtractor(nn.Module):
    """Base class for all DNA model feature extractors."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.hidden_size = None
        self.tokenizer = None
        self.model = None
        self._use_bf16_autocast = _needs_bf16_autocast(device)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract mean-pooled sequence embeddings from the frozen model."""
        raise NotImplementedError("Subclasses must implement encode()")

    def get_tokenizer(self):
        """Return the tokenizer for this model."""
        return self.tokenizer


def _disable_dnabert_triton_flash():
    """Select the MosaicBERT PyTorch attention path for DNABERT-2 / DNABERT-S."""
    for nm, md in list(sys.modules.items()):
        if nm.endswith("bert_layers") and "DNABERT" in nm:
            setattr(md, "flash_attn_qkvpacked_func", None)


@torch.no_grad()
def _mosaic_encode_multilayer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                              layers) -> torch.Tensor:
    """Multi-layer capture for MosaicBERT models."""
    input_ids = input_ids.to(self.device)
    attention_mask = attention_mask.to(self.device)
    want = {int(l) for l in layers}
    pooled = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._use_bf16_autocast):
        emb = self.model.embeddings(input_ids, torch.zeros_like(input_ids), None)
        if 0 in want:
            pooled[0] = _mean_pool(emb, attention_mask)
        outs = self.model.encoder(emb, attention_mask,
                                  output_all_encoded_layers=True, subset_mask=None)
    lens = [int(v) for v in attention_mask.sum(1).tolist()]
    offs = [0]
    for L in lens:
        offs.append(offs[-1] + L)
    for i, h in enumerate(outs, start=1):
        if i in want:
            pooled[i] = torch.stack([h[offs[b]:offs[b + 1]].float().mean(0)
                                     for b in range(len(lens))]).cpu()
    return torch.stack([pooled[int(l)] for l in layers], dim=1)


class DNABERTSExtractor(BaseExtractor):
    """DNABERT-S feature extractor."""

    EXPECTED_CONFIGS = {
        "zhihan1996/DNABERT-S": {"hidden_size": 768, "vocab_size": 4096},
    }

    def __init__(self, model_name: str = None, device: str = "cuda"):
        super().__init__(device)
        self.supports_chunk_batching = True
        from transformers import AutoModel, AutoConfig, AutoTokenizer
        import warnings

        if device == "cpu":
            raise RuntimeError("DNABERT-S requires CUDA for Flash Attention. Please use GPU.")

        model_name = model_name or MODEL_CONFIGS["DNABERT-S"]["model_name"]

        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            _validate_model_config(model_name, config, self.EXPECTED_CONFIGS, "hidden_size")
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pooler.*")
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _disable_dnabert_triton_flash()
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"    DNABERT-S loaded with {num_params / 1e6:.1f}M parameters (frozen)")

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._use_bf16_autocast):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        return _mean_pool(hidden_states, attention_mask)

    encode_multilayer = _mosaic_encode_multilayer


class DNABERT2Extractor(BaseExtractor):
    """DNABERT-2 feature extractor."""

    EXPECTED_CONFIGS = {
        "zhihan1996/DNABERT-2-117M": {"hidden_size": 768, "vocab_size": 4096},
    }

    def __init__(self, model_name: str = None, device: str = "cuda"):
        super().__init__(device)
        self.supports_chunk_batching = True
        from transformers import AutoModel, AutoConfig, AutoTokenizer
        import warnings

        model_name = model_name or MODEL_CONFIGS["DNABERT-2"]["model_name"]

        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            _validate_model_config(model_name, config, self.EXPECTED_CONFIGS, "hidden_size")
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _disable_dnabert_triton_flash()
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"    DNABERT-2 loaded with {num_params / 1e6:.1f}M parameters (frozen)")

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._use_bf16_autocast):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        return _mean_pool(hidden_states, attention_mask)

    encode_multilayer = _mosaic_encode_multilayer


class NTExtractor(BaseExtractor):
    """Nucleotide Transformer feature extractor."""

    EXPECTED_CONFIGS = {
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species": {
            "max_position_embeddings": 2050, "hidden_size": 768, "vocab_size": 4107
        },
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species": {
            "max_position_embeddings": 2050, "hidden_size": 1024, "vocab_size": 4107
        },
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species": {
            "max_position_embeddings": 1002, "hidden_size": 2560, "vocab_size": 4105
        },
    }

    def __init__(self, model_name: str = None, device: str = "cuda"):
        super().__init__(device)
        self.supports_chunk_batching = True
        from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig

        model_name = model_name or MODEL_CONFIGS["NT-v2-250M"]["model_name"]

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        _validate_model_config(model_name, config, self.EXPECTED_CONFIGS, "max_position_embeddings")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"    NucleotideTransformer loaded with {num_params / 1e6:.1f}M parameters (frozen)")

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._use_bf16_autocast):
            outputs = self.model.esm(input_ids, attention_mask=attention_mask)
        return _mean_pool(outputs.last_hidden_state, attention_mask)

    @torch.no_grad()
    def encode_multilayer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          layers) -> torch.Tensor:
        """Capture multiple hidden-state layers in one forward pass."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._use_bf16_autocast):
            outputs = self.model.esm(input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True)
        hs = outputs.hidden_states
        pooled = [_mean_pool(hs[li], attention_mask) for li in layers]
        return torch.stack(pooled, dim=1)


class ProkBERTExtractor(BaseExtractor):
    """ProkBERT feature extractor."""

    EXPECTED_CONFIGS = {
        "neuralbioinfo/prokbert-mini-long": {
            "max_position_embeddings": 2048, "hidden_size": 384, "vocab_size": 4200
        },
    }

    def __init__(self, model_name: str = None, device: str = "cuda"):
        super().__init__(device)
        self.supports_chunk_batching = True
        from transformers import AutoModel, AutoTokenizer, AutoConfig

        model_name = model_name or MODEL_CONFIGS["ProkBERT-mini-long"]["model_name"]

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        _validate_model_config(model_name, config, self.EXPECTED_CONFIGS, "max_position_embeddings")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"    ProkBERT loaded with {num_params / 1e6:.1f}M parameters (frozen)")

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._use_bf16_autocast):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return _mean_pool(outputs.last_hidden_state, attention_mask)

    @torch.no_grad()
    def encode_multilayer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          layers) -> torch.Tensor:
        """Capture multiple hidden-state layers in one forward pass."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._use_bf16_autocast):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
        hs = outputs.hidden_states
        pooled = [_mean_pool(hs[li], attention_mask) for li in layers]
        return torch.stack(pooled, dim=1)


def _evo2_disable_fp8_input_projections(models=("evo2_20b", "evo2_40b")):
    """Disable FP8 input projections in Evo2 configs that require Transformer Engine."""
    import os

    import yaml

    import evo2.utils as _u

    pkg = os.path.dirname(os.path.abspath(_u.__file__))
    changed = []
    for m in models:
        rel = _u.CONFIG_MAP.get(m)
        if rel is None:
            continue
        src = os.path.join(pkg, rel)
        dst = src.replace(".yml", "-nofp8.yml")
        cfg = yaml.safe_load(open(src))
        if not cfg.get("use_fp8_input_projections", False):
            continue
        cfg["use_fp8_input_projections"] = False
        yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
        _u.CONFIG_MAP[m] = os.path.relpath(dst, pkg)
        changed.append(f"{m}: {os.path.basename(dst)}")
    return "; ".join(changed) or "nothing to patch"


def _patch_evo2_flash_attn():
    """Monkey-patch vortex flash attention to use PyTorch SDPA."""
    import torch.nn.functional as F
    import vortex.ops.attn_interface as attn_mod

    class _SDPAFlashAttn:
        @staticmethod
        def fwd(q, k, v, out, alibi_slopes, dropout_p, softmax_scale, causal,
                window_size_left, window_size_right, softcap, return_softmax, gen_):
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            result = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                dropout_p=dropout_p if dropout_p > 0 else 0.0,
                is_causal=causal,
                scale=softmax_scale,
            )
            result = result.transpose(1, 2).contiguous()
            batch_size, num_heads, seqlen_q = q.shape[0], q.shape[2], q.shape[1]
            softmax_lse = torch.empty(batch_size, num_heads, seqlen_q,
                                      dtype=torch.float32, device=q.device)
            return result, softmax_lse, torch.empty(0), torch.empty(0)

    attn_mod.flash_attn_gpu = _SDPAFlashAttn()

    import vortex.model.attention as _vattn

    def _sdpa_qkvpacked(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                        window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                        deterministic=False, return_attn_probs=False):
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                           is_causal=bool(causal), scale=softmax_scale)
        return o.transpose(1, 2).contiguous()
    _vattn.local_flash_attn_qkvpacked_func = _sdpa_qkvpacked
    print("    Patched vortex flash_attn + local_flash_attn_qkvpacked_func -> PyTorch SDPA (Blackwell)")


EVO2_FP8_CONFIG_MODELS = ("evo2_20b", "evo2_40b", "evo2_7b_base")


class Evo2Extractor(BaseExtractor):
    """Evo2 (StripedHyena2) feature extractor."""

    HIDDEN_SIZES = {
        "evo2_1b_base": 1920,
        "evo2_7b_base": 4096,
        "evo2_7b": 4096,
    }

    def __init__(self, model_name: str = None, device: str = "cuda"):
        super().__init__(device)

        model_name = model_name or MODEL_CONFIGS["Evo2-7B-1M-ml8192"]["model_name"]

        if self._use_bf16_autocast:
            _patch_evo2_flash_attn()

        print(f"    Loading Evo2 model ({model_name})...")
        if model_name in EVO2_FP8_CONFIG_MODELS:
            print(f"    [evo2] fp8 override -> "
                  f"{_evo2_disable_fp8_input_projections(models=(model_name,))}", flush=True)
        from evo2 import Evo2
        self._evo2 = Evo2(model_name)
        self.model = self._evo2.model
        self.model.eval()

        import types as _types
        nb = sum(1 for _n, _ in self.model.named_modules()
                 if _n.startswith("blocks.") and _n[len("blocks."):].isdigit())
        self._n_blocks = nb
        if not hasattr(self.model, "config"):
            self.model.config = _types.SimpleNamespace()
        if nb:
            self.model.config.num_hidden_layers = nb

        self.hidden_size = self.HIDDEN_SIZES.get(model_name, 4096)

        self._char_tokenizer = self._evo2.tokenizer
        self.tokenizer = self._make_hf_compatible_tokenizer()

        num_params = sum(p.numel() for p in self.model.parameters())
        size_str = f"{num_params / 1e9:.1f}B" if num_params >= 1e9 else f"{num_params / 1e6:.0f}M"
        print(f"    Evo2 loaded with {size_str} parameters (frozen)")

    def _make_hf_compatible_tokenizer(self):
        """Wrap Evo2's CharLevelTokenizer to match the HuggingFace tokenizer API."""
        char_tok = self._char_tokenizer

        class Evo2TokenizerWrapper:
            def __init__(self, inner):
                self._inner = inner
                self.pad_token_id = 0

            def __call__(self, sequence, truncation=True, max_length=None,
                         return_tensors=None, **kwargs):
                token_ids = self._inner.tokenize(sequence)
                if truncation and max_length and len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                input_ids = torch.tensor([token_ids], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

        return Evo2TokenizerWrapper(char_tok)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device).int()
        attention_mask = attention_mask.to(self.device)

        ln = getattr(self, "layer_name", None) or "norm"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, embeddings = self._evo2(input_ids, return_embeddings=True, layer_names=[ln])

        hidden_states = embeddings[ln]
        return _mean_pool(hidden_states, attention_mask)

    @torch.no_grad()
    def encode_multilayer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          layers) -> torch.Tensor:
        """Capture multiple Evo2 layers in one forward pass."""
        input_ids = input_ids.to(self.device).int()
        attention_mask = attention_mask.to(self.device)
        names = [x if isinstance(x, str)
                 else ("norm" if int(x) >= self._n_blocks else f"blocks.{int(x)}")
                 for x in layers]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, embeddings = self._evo2(input_ids, return_embeddings=True, layer_names=names)
        pooled = [_mean_pool(embeddings[n], attention_mask) for n in names]
        return torch.stack(pooled, dim=1)


EXTRACTOR_CLASSES = {
    "DNABERT-S": DNABERTSExtractor,
    "DNABERT-2": DNABERT2Extractor,
    "NT-v2-250M": NTExtractor,
    "NT-v2-500M": NTExtractor,
    "NT-2.5B-multi-species": NTExtractor,
    "ProkBERT-mini-long": ProkBERTExtractor,
    "Evo2-7B-1M-ml8192": Evo2Extractor,
}


def get_extractor(model_name: str, device: str = "cuda", **kwargs) -> BaseExtractor:
    """Create a frozen feature extractor by model name."""
    if model_name not in EXTRACTOR_CLASSES:
        available = list(EXTRACTOR_CLASSES.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")

    extractor_class = EXTRACTOR_CLASSES[model_name]
    config = MODEL_CONFIGS[model_name]

    extractor = extractor_class(model_name=config["model_name"], device=device, **kwargs)
    if config.get("layer_name") is not None:
        extractor.layer_name = config["layer_name"]
    return extractor


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Return a copy of the configuration for a model."""
    if model_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")

    return MODEL_CONFIGS[model_name].copy()


def list_available_models() -> list:
    """Return the list of available baseline model names."""
    return list(MODEL_CONFIGS.keys())
