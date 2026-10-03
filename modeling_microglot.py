"""MicroGlot: a taxonomy-informed sparse (mixture-of-experts) decoder-only DNA language model.

One code file serves the three models of the repository ``athanzli/MicroGlot``:

* the repository root  - MicroGlot, the species-conditioned language model (``MicroGlotForCausalLM``)
* ``plain/``            - MicroGlot-plain, the same architecture without species conditioning
* ``species_encoder/``  - the Species-encoder, which predicts a species embedding from DNA (``MicroGlotSpeciesEncoder``)

Species inputs follow the ``input_ids`` / ``inputs_embeds`` convention:

* ``species_ids``    (LongTensor ``[batch]``): rows of the model's table of the 99,700 pretraining species
                     (produced by the tokenizer's ``species=`` argument); ``-1`` marks a sequence whose
                     species is unknown and must be inferred by an attached species encoder.
* ``species_embeds`` (FloatTensor ``[batch, 32]`` or ``[32]``): your own 32-d species vectors.

MicroGlot runs on NVIDIA GPUs with flash-attn installed, as it was trained and benchmarked. The numerical
code (attention, mixture of experts, decoder layers) is the code MicroGlot was trained with, unchanged.
"""

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from packaging import version
from torch.utils.checkpoint import checkpoint

# Supported versions (see the model card): PyTorch >= 2.7, transformers >= 4.51.3, flash-attn 2.7.4-2.8.3 tested.
# PyTorch 2.5 and 2.6 compute nn.RMSNorm(eps=None) of bf16 inputs with the bf16 epsilon, which changes the outputs.
_INSTALL = "https://huggingface.co/athanzli/MicroGlot#installation"
if version.parse(torch.__version__).release < (2, 7):
    raise ImportError(f"MicroGlot requires PyTorch 2.7 or later (found {torch.__version__}). See {_INSTALL}.")
if version.parse(transformers.__version__).release < (4, 51, 3):
    raise ImportError(f"MicroGlot requires transformers 4.51.3 or later (found {transformers.__version__}). "
                      f"See {_INSTALL}.")
# MicroGlot was trained and benchmarked with flash-attn's rotary position-embedding kernel, so it
# is required: every installation then computes the same embeddings.
try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError as _flash_attn_error:
    raise ImportError(
        "MicroGlot requires FlashAttention-2 (the flash-attn package, whose rotary position-embedding kernel was "
        f"used to train and benchmark the model) and an NVIDIA GPU. Install it as described at {_INSTALL}. "
        f"(Importing flash_attn failed with: {_flash_attn_error})"
    ) from _flash_attn_error

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)

# from_pretrained(torch_dtype=...) is called dtype= from transformers 4.56 on
_DTYPE_KWARG = "dtype" if version.parse(transformers.__version__).release >= (4, 56) else "torch_dtype"


@dataclass
class MicroGlotModelOutput(ModelOutput):
    """Backbone output. The species position is removed, so every tensor lines up with ``input_ids``.

    last_hidden_state: ``[batch, length, hidden]``, the final-norm output.
    hidden_states: with ``output_hidden_states=True``, ``num_hidden_layers + 1`` tensors (24): ``[0]`` the
        token embeddings, ``[k]`` the output of decoder layer k (1..22), ``[-1]`` the final-norm output
        (equal to ``last_hidden_state``).
    moe_loss: the router load-balancing loss summed over layers (already scaled by
        ``router_aux_loss_coef``), or None with ``output_router_logits=False``.
    species_embeds: ``[batch, 32]`` unit-norm species vectors actually used, in the model dtype
        (None for MicroGlot-plain).
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    moe_loss: Optional[torch.FloatTensor] = None
    species_embeds: Optional[torch.FloatTensor] = None


@dataclass
class MicroGlotCausalLMOutput(CausalLMOutputWithPast):
    """``CausalLMOutputWithPast`` plus the species vectors used (``species_embeds``, appended after the
    standard fields). ``past_key_values`` and ``attentions`` are always None (MicroGlot has no KV cache)."""

    species_embeds: Optional[torch.FloatTensor] = None


@dataclass
class MicroGlotSpeciesEncoderOutput(ModelOutput):
    """Species-encoder output, in the style of ``CLIPTextModelOutput``.

    species_embeds: ``[batch, 32]`` unit-norm predicted species vectors, normalised in the model dtype: the
        vectors MicroGlot is conditioned on (listed first, so pipelines return it).
    embeddings: ``[batch, 32]`` the projection output before normalisation.
    pooler_output: ``[batch, hidden]`` the pooled state (last real token).
    last_hidden_state, hidden_states, moe_loss: as in ``MicroGlotModelOutput``.
    """

    species_embeds: torch.FloatTensor = None
    embeddings: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    moe_loss: Optional[torch.FloatTensor] = None


@dataclass
class MoEParameterCounts:
    """Parameter counts for MoE models."""
    total_params: int
    activated_params: int
    non_embedding_total: int
    non_embedding_activated: int
    embedding_params: int
    moe_total_params: int
    moe_activated_params: int
    dense_params: int
    num_moe_layers: int
    num_dense_layers: int
    experts_per_layer: List[int]
    top_k: int
    
    def __str__(self) -> str:
        """Human-readable summary string."""
        return (
            f"MoE Parameter Counts:\n"
            f"  Total Parameters: {self.total_params:,} ({self.total_params / 1e9:.2f}B)\n"
            f"  Activated Parameters: {self.activated_params:,} ({self.activated_params / 1e6:.1f}M)\n"
            f"  Non-Embedding Total: {self.non_embedding_total:,} ({self.non_embedding_total / 1e9:.2f}B)\n"
            f"  Non-Embedding Activated: {self.non_embedding_activated:,} ({self.non_embedding_activated / 1e6:.1f}M)\n"
            f"  Embedding Parameters: {self.embedding_params:,} ({self.embedding_params / 1e6:.1f}M)\n"
            f"  MoE Layers: {self.num_moe_layers} (total: {self.moe_total_params:,}, activated: {self.moe_activated_params:,})\n"
            f"  Dense Layers: {self.num_dense_layers} ({self.dense_params:,})\n"
            f"  Top-K Routing: {self.top_k}\n"
            f"  Experts per Layer: {self.experts_per_layer}"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_params": self.total_params,
            "activated_params": self.activated_params,
            "non_embedding_total": self.non_embedding_total,
            "non_embedding_activated": self.non_embedding_activated,
            "embedding_params": self.embedding_params,
            "moe_total_params": self.moe_total_params,
            "moe_activated_params": self.moe_activated_params,
            "dense_params": self.dense_params,
            "num_moe_layers": self.num_moe_layers,
            "num_dense_layers": self.num_dense_layers,
            "experts_per_layer": self.experts_per_layer,
            "top_k": self.top_k,
        }


def compute_moe_parameter_counts(config: "GenomicModelConfig") -> MoEParameterCounts:
    """Compute total and activated parameter counts for MoE models from config."""
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    vocab_size = config.vocab_size
    num_layers = config.num_hidden_layers
    num_query_heads = config.num_query_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    top_k = config.num_experts_per_tok if config.use_moe else 0
    use_residual = config.moe_use_residual if config.use_moe else False
    
    if config.use_moe and config.moe_layer_experts is not None:
        experts_per_layer = config.moe_layer_experts
    else:
        experts_per_layer = [0] * num_layers
    
    embedding_params = vocab_size * hidden_size
    
    attn_params = (
        hidden_size * (num_query_heads * head_dim) +
        hidden_size * (num_kv_heads * head_dim) +
        hidden_size * (num_kv_heads * head_dim) +
        (num_query_heads * head_dim) * hidden_size
    )
    
    layernorm_params = 2 * hidden_size
    
    ffn_params = (
        hidden_size * (2 * intermediate_size) +
        intermediate_size * hidden_size
    )
    
    moe_total_params = 0
    moe_activated_params = 0
    dense_params = 0
    num_moe_layers = 0
    num_dense_layers = 0
    
    for layer_idx, num_experts in enumerate(experts_per_layer):
        layer_base_params = attn_params + layernorm_params
        
        if num_experts > 0:
            num_moe_layers += 1
            
            expert_total = num_experts * ffn_params
            
            gate_params = hidden_size * num_experts
            
            expert_activated = (top_k / num_experts) * expert_total + gate_params
            
            species_film_params = 0
            if config.species_emb_dim is not None:
                species_film_params = 2 * (config.species_emb_dim * num_experts + num_experts)
            
            residual_params = (ffn_params + hidden_size * 2) if use_residual else 0
            
            layer_moe_total = expert_total + gate_params + species_film_params + residual_params
            layer_moe_activated = expert_activated + species_film_params + residual_params
            
            moe_total_params += layer_base_params + layer_moe_total
            moe_activated_params += layer_base_params + layer_moe_activated
        else:
            num_dense_layers += 1
            layer_total = layer_base_params + ffn_params
            dense_params += layer_total
    
    final_norm_params = hidden_size
    
    species_encoder_params = 0
    if config.use_species_encoder and config.species_emb_dim is not None:
        species_encoder_params = config.species_emb_dim
    
    non_embedding_total = moe_total_params + dense_params + final_norm_params + species_encoder_params
    non_embedding_activated = moe_activated_params + dense_params + final_norm_params + species_encoder_params
    
    total_params = embedding_params + non_embedding_total
    activated_params = embedding_params + non_embedding_activated
    
    return MoEParameterCounts(
        total_params=total_params,
        activated_params=activated_params,
        non_embedding_total=non_embedding_total,
        non_embedding_activated=non_embedding_activated,
        embedding_params=embedding_params,
        moe_total_params=moe_total_params,
        moe_activated_params=moe_activated_params,
        dense_params=dense_params,
        num_moe_layers=num_moe_layers,
        num_dense_layers=num_dense_layers,
        experts_per_layer=experts_per_layer,
        top_k=top_k,
    )


def count_parameters(model: nn.Module, requires_grad_only: bool = False) -> int:
    """Count the number of parameters in a model."""
    if requires_grad_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_parameter_counts(model: "GenomicModel") -> MoEParameterCounts:
    """Get parameter counts from an instantiated model."""
    if hasattr(model, 'model'):
        base_model = model.model
        config = model.config
    else:
        base_model = model
        config = model.config
    
    counts = compute_moe_parameter_counts(config)
    
    actual_total = count_parameters(model)
    
    expected_diff = 0
    if hasattr(model, 'lm_head'):
        expected_diff = config.vocab_size * config.hidden_size
    
    if abs(actual_total - counts.total_params - expected_diff) > 1000:
        import warnings
        warnings.warn(
            f"Parameter count mismatch: analytical={counts.total_params:,}, "
            f"actual={actual_total:,}, expected_diff={expected_diff:,}. "
            f"Difference: {actual_total - counts.total_params - expected_diff:,}"
        )
    
    return counts


def format_parameter_count(params: int) -> str:
    """Format parameter count in human-readable form."""
    if params >= 1e9:
        return f"{params / 1e9:.2f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.1f}M"
    elif params >= 1e3:
        return f"{params / 1e3:.1f}K"
    else:
        return str(params)


def print_model_parameter_summary(config: "GenomicModelConfig") -> None:
    """Print a detailed parameter summary for a model configuration."""
    counts = compute_moe_parameter_counts(config)
    
    print("=" * 60)
    print("Model Parameter Summary")
    print("=" * 60)
    print(f"Total Parameters:      {format_parameter_count(counts.total_params):>10} ({counts.total_params:,})")
    print(f"Activated Parameters:  {format_parameter_count(counts.activated_params):>10} ({counts.activated_params:,})")
    print("-" * 60)
    print(f"Embedding Parameters:  {format_parameter_count(counts.embedding_params):>10}")
    print(f"Non-Embedding Total:   {format_parameter_count(counts.non_embedding_total):>10}")
    print(f"Non-Embedding Active:  {format_parameter_count(counts.non_embedding_activated):>10}")
    print("-" * 60)
    print(f"Dense Layers:          {counts.num_dense_layers:>10} ({format_parameter_count(counts.dense_params)})")
    print(f"MoE Layers:            {counts.num_moe_layers:>10} (total: {format_parameter_count(counts.moe_total_params)}, "
          f"active: {format_parameter_count(counts.moe_activated_params)})")
    print(f"Top-K Routing:         {counts.top_k:>10}")
    print(f"Experts per Layer:     {counts.experts_per_layer}")
    print("=" * 60)



class MicroGlotConfig(PretrainedConfig):
    """Configuration shared by MicroGlot, MicroGlot-plain and the Species-encoder.

    The architecture fields (vocab_size, hidden_size, num_hidden_layers, num_query_heads, num_kv_heads,
    intermediate_size, rope_theta, moe_layer_experts, num_experts_per_tok, moe_use_residual,
    gate_softmax_over_all_experts, router_aux_loss_coef, moe_dispatch, ...) are those of the pretrained models.
    `moe_dispatch`: "sorted" (default), "loop" or "grouped".

    Species-related fields:
        species_emb_dim (`int`, *optional*): size of the species vector (32). `None` -> no species input
            (MicroGlot-plain, and the backbone of the species encoder).
        prepend_species_token (`bool`): insert the projected species vector as a token after `[BOS]`.
        num_species (`int`): rows in the built-in species table addressed by `species_ids` (99,700; 0 = none).
        species_vocab_sha256 (`str`, *optional*): sha256 of the species names in table order, joined by
            newlines; the tokenizer's `species_vocab_sha256` must match it.
        species_encoder_repo / species_encoder_revision / species_encoder_subfolder (`str`, *optional*): where
            `load_species_encoder()` and `from_pretrained(..., species_encoder=True)` find the Species-encoder.
            `species_encoder_repo=None`: the repository or folder this model was loaded from, at the same
            revision; `species_encoder_subfolder` is the encoder's folder there (`"species_encoder"`).
        species_encoder_embedding_dim (`int`): output size of `MicroGlotSpeciesEncoder` (32).
        species_encoder_pooling_strategy (`str`): `"eos"` (last real token) or `"mean"`.
    """

    model_type = "microglot"
    keys_to_ignore_at_inference = ["moe_loss", "species_embeds"]

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 1024,
        num_hidden_layers: int = 23,
        num_query_heads: int = 16,
        num_kv_heads: int = 8,
        intermediate_size: Optional[int] = None,
        attention_dropout_prob: float = 0.0,
        classifier_dropout_prob: float = 0.0,
        attn_output_dropout_prob: float = 0.0,
        moe_output_dropout_prob: float = 0.0,
        ffn_dropout_prob: float = 0.0,
        moe_expert_dropout_prob: float = 0.0,
        rope_theta: float = 500000.0,
        max_trained_length: int = 8192,
        init_scale: float = 0.1,
        use_swiglu: bool = True,
        gradient_checkpointing: bool = False,
        use_flash_attention: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        num_experts_per_tok: int = 1,
        router_aux_loss_coef: float = 0.001,
        moe_layer_experts: Optional[List[int]] = None,
        moe_use_residual: bool = True,
        moe_capacity_factor: float = 1.25,
        moe_eval_capacity_factor: float = 2.0,
        moe_min_capacity: int = 4,
        moe_noisy_gate_policy: Optional[str] = None,
        moe_drop_tokens: bool = True,
        moe_use_rts: bool = True,
        gate_softmax_over_all_experts: bool = True,
        moe_dispatch: str = "sorted",
        species_emb_dim: Optional[int] = None,
        prepend_species_token: bool = False,
        num_species: int = 0,
        species_encoder_embedding_dim: int = 32,
        species_encoder_pooling_strategy: str = "eos",
        species_encoder_repo: Optional[str] = None,
        species_encoder_revision: Optional[str] = None,
        species_encoder_subfolder: Optional[str] = None,
        species_vocab_sha256: Optional[str] = None,
        use_species_encoder: bool = False,
        **kwargs,
    ):
        if use_species_encoder:  # a training-code config: the species encoder is bundled inside the checkpoint
            logger.warning_once(
                "This checkpoint bundles a copy of the species encoder, which this code ignores. Species names "
                "and vectors work unchanged. To infer species, attach the Species-encoder with "
                "model.load_species_encoder()."
            )
            if species_encoder_repo is None:
                species_encoder_repo, species_encoder_subfolder = "athanzli/MicroGlot", "species_encoder"
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, bos_token_id=bos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        if intermediate_size is None:
            if use_swiglu:
                hidden_dim = (8 * hidden_size) // 3
                intermediate_size = 256 * ((hidden_dim + 255) // 256)
            else:
                intermediate_size = 4 * hidden_size
        self.intermediate_size = intermediate_size
        self.attention_dropout_prob = attention_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.attn_output_dropout_prob = attn_output_dropout_prob
        self.moe_output_dropout_prob = moe_output_dropout_prob
        self.ffn_dropout_prob = ffn_dropout_prob
        self.moe_expert_dropout_prob = moe_expert_dropout_prob
        self.rope_theta = rope_theta
        self.max_trained_length = max_trained_length
        self.init_scale = init_scale
        self.use_swiglu = use_swiglu
        self.gradient_checkpointing = gradient_checkpointing
        self.use_flash_attention = use_flash_attention
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef
        self.moe_layer_experts = moe_layer_experts
        self.use_moe = moe_layer_experts is not None and any(n > 0 for n in moe_layer_experts)
        self.moe_use_residual = moe_use_residual
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_eval_capacity_factor = moe_eval_capacity_factor
        self.moe_min_capacity = moe_min_capacity
        self.moe_noisy_gate_policy = moe_noisy_gate_policy
        self.moe_drop_tokens = moe_drop_tokens
        self.moe_use_rts = moe_use_rts
        self.gate_softmax_over_all_experts = gate_softmax_over_all_experts
        self.moe_dispatch = moe_dispatch
        self.species_emb_dim = species_emb_dim
        self.use_species_encoder = False  # a training-code flag; the encoder is attached separately
        self.prepend_species_token = prepend_species_token
        self.num_species = num_species
        self.species_encoder_embedding_dim = species_encoder_embedding_dim
        self.species_encoder_pooling_strategy = species_encoder_pooling_strategy
        self.species_encoder_repo = species_encoder_repo
        self.species_encoder_revision = species_encoder_revision
        self.species_encoder_subfolder = species_encoder_subfolder
        self.species_vocab_sha256 = species_vocab_sha256
        if prepend_species_token and species_emb_dim is None:
            raise ValueError("species_emb_dim must be set when prepend_species_token=True.")
        if num_species and species_emb_dim is None:
            raise ValueError("num_species > 0 needs species_emb_dim (the width of the species table).")
        assert hidden_size % num_query_heads == 0
        self.head_dim = hidden_size // num_query_heads
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        self.num_kv_groups = num_query_heads // num_kv_heads
        self._validate_moe_config()

    def _validate_moe_config(self):
        if not self.use_moe:
            return
        if len(self.moe_layer_experts) != self.num_hidden_layers:
            raise ValueError("moe_layer_experts must have one entry per layer (0 = dense).")
        for i, n in enumerate(self.moe_layer_experts):
            if not isinstance(n, int) or n < 0:
                raise ValueError(f"moe_layer_experts[{i}] = {n} is invalid.")
            if 0 < n < self.num_experts_per_tok:
                raise ValueError(f"moe_layer_experts[{i}] = {n} < num_experts_per_tok.")

    def get_num_experts_for_layer(self, layer_idx: int) -> int:
        """Number of experts in layer `layer_idx` (0 = dense feed-forward)."""
        if not self.use_moe:
            return 0
        return self.moe_layer_experts[layer_idx]

    def get_parameter_counts(self) -> "MoEParameterCounts":
        """Total and activated parameter counts for this configuration."""
        return compute_moe_parameter_counts(self)

    def print_parameter_summary(self) -> None:
        """Print a detailed parameter summary for this configuration."""
        print_model_parameter_summary(self)

    def to_dict(self):
        # save_pretrained() copies this code file next to config.json (see register_for_auto_class below), so
        # keep every auto_map entry local: a saved model then runs the code it was saved with, not the Hub's
        # latest (transformers rewrites Hub-loaded entries to "athanzli/MicroGlot--modeling_microglot...").
        out = super().to_dict()
        if isinstance(out.get("auto_map"), dict):
            out["auto_map"] = {k: v.split("--")[-1] if isinstance(v, str) else v for k, v in out["auto_map"].items()}
        return out

    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        config_dict, kwargs = super().get_config_dict(*args, **kwargs)
        if config_dict.get("model_type") == "genomic_lm":  # the training code's name of this model type
            config_dict["model_type"] = cls.model_type
        return config_dict, kwargs


GenomicModelConfig = MicroGlotConfig  # the training code's name, used in the layer code's type hints


class MicroGlotSpeciesEncoderConfig(MicroGlotConfig):
    """Configuration of the Species-encoder: `MicroGlotConfig` under its own model type, so that the three
    models of the repository, which share this code file, keep distinct Auto classes."""

    model_type = "microglot_species_encoder"

    def __init__(self, **kwargs):  # explicit: transformers 5 would otherwise turn the subclass into a dataclass
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        config_dict, kwargs = super().get_config_dict(*args, **kwargs)
        if config_dict.get("model_type") in ("microglot", "genomic_lm"):  # encoder folders of the training code
            config_dict["model_type"] = cls.model_type
        return config_dict, kwargs


class GenomicAttention(nn.Module):
    """Grouped Query Attention (GQA) with RoPE and Flash Attention 2.0."""
    
    def __init__(self, config: GenomicModelConfig):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_query_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = config.num_kv_groups
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout_prob
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_query_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.attn_output_dropout_prob = config.attn_output_dropout_prob
        self.attn_output_dropout = nn.Dropout(config.attn_output_dropout_prob) if config.attn_output_dropout_prob > 0 else nn.Identity()
        
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            base=config.rope_theta,
            interleaved=False
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply causal (autoregressive) Flash Attention with RoPE."""
        batch_size, seq_len, _ = hidden_states.shape
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=hidden_states.device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_query_heads, self.head_dim).contiguous()
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).contiguous()
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).contiguous()
        
        qkv_states = self.rotary_emb(
            qkv = torch.cat([query_states, key_states, value_states], dim=2),
            num_heads_q = self.num_query_heads,
        )
        query_states = qkv_states[:, :, :self.num_query_heads, :].contiguous()
        key_states = qkv_states[:, :, self.num_query_heads:self.num_query_heads + self.num_kv_heads, :].contiguous()
        value_states = qkv_states[:, :, self.num_query_heads + self.num_kv_heads:, :].contiguous()
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # SDPA picks the fastest kernel the GPU and dtype support: flash attention for bf16/fp16 on
        # Ampere or newer, another kernel otherwise (forcing flash left fp32 and older GPUs without one).
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_query_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        attn_output = self.attn_output_dropout(attn_output)
        return attn_output


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    
    def __init__(self, config: GenomicModelConfig, dropout_prob: float = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.dropout_prob = dropout_prob if dropout_prob is not None else getattr(config, 'ffn_dropout_prob', 0.0)
        self.dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        out = self.down_proj(F.silu(gate) * up)
        return self.dropout(out)


class TopKGate(nn.Module):
    """Top-k gating network for Mixture of Experts."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 1,
        noisy_gate_policy: Optional[str] = None,
        species_emb_dim: Optional[int] = None,
        gate_softmax_over_all_experts: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gate_policy = noisy_gate_policy
        self.gate_softmax_over_all_experts = gate_softmax_over_all_experts
        
        self.wg = nn.Linear(hidden_size, num_experts, bias=False)
        
        self.use_species_film = species_emb_dim is not None
        if self.use_species_film:
            self.film_gamma = nn.Linear(species_emb_dim, num_experts, bias=True)
            self.film_beta = nn.Linear(species_emb_dim, num_experts, bias=True)
            nn.init.trunc_normal_(self.film_gamma.weight, std=0.02)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.trunc_normal_(self.film_beta.weight, std=0.02)
            nn.init.zeros_(self.film_beta.bias)
        else:
            self.film_gamma = None
            self.film_beta = None
    
    def _compute_aux_loss(
        self,
        logits: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GShard-style load balancing auxiliary loss."""
        num_tokens = logits.shape[0]
        
        flat_indices = top_k_indices.reshape(-1)
        expert_counts = torch.zeros(self.num_experts, device=logits.device, dtype=logits.dtype)
        expert_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=logits.dtype))
        f = expert_counts / num_tokens
        
        probs = F.softmax(logits, dim=-1)
        p = probs.mean(dim=0)
        
        if self.top_k == 1:
            aux_loss = self.num_experts * torch.sum(f * p)
        else:
            aux_loss = (self.num_experts ** 2) * torch.mean(f * p) / self.top_k
        
        return aux_loss
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        species_emb: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute top-k gate scores."""
        if self.training and self.noisy_gate_policy == 'Jitter':
            noise = torch.empty_like(hidden_states).uniform_(1.0 - 0.01, 1.0 + 0.01)
            hidden_states = hidden_states * noise
        
        logits = F.linear(hidden_states, self.wg.weight, self.wg.bias).float()

        if self.use_species_film and species_emb is not None and self.film_gamma is not None:
            gamma = F.linear(species_emb, self.film_gamma.weight, self.film_gamma.bias).float()
            beta = F.linear(species_emb, self.film_beta.weight, self.film_beta.bias).float()
            
            gamma_expanded = gamma.repeat_interleave(seq_len, dim=0)
            beta_expanded = beta.repeat_interleave(seq_len, dim=0)
            
            logits = (1.0 + gamma_expanded) * logits + beta_expanded
        
        if self.gate_softmax_over_all_experts:
            all_probs = F.softmax(logits, dim=-1)
            gate_scores, top_k_indices = torch.topk(all_probs, self.top_k, dim=-1)
            if self.top_k > 1:
                gate_scores = gate_scores / gate_scores.sum(dim=-1, keepdim=True)
        else:
            top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
            gate_scores = F.softmax(top_k_logits, dim=-1)

        aux_loss = self._compute_aux_loss(logits, top_k_indices)

        return gate_scores, top_k_indices, aux_loss


class MoELayer(nn.Module):
    """Native PyTorch Mixture of Experts layer."""
    
    def __init__(self, config: GenomicModelConfig, num_experts: int):
        """Initialize native MoE layer."""
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = config.num_experts_per_tok
        self.aux_loss_coef = config.router_aux_loss_coef
        self.moe_dispatch = getattr(config, "moe_dispatch", "loop")
        
        use_species = config.use_species_encoder or (config.species_emb_dim is not None)
        species_dim = config.species_emb_dim if use_species else None
        
        self.gate = TopKGate(
            hidden_size=config.hidden_size,
            num_experts=num_experts,
            top_k=self.top_k,
            noisy_gate_policy=config.moe_noisy_gate_policy,
            species_emb_dim=species_dim,
            gate_softmax_over_all_experts=getattr(config, "gate_softmax_over_all_experts", False),
        )
        
        self.experts = nn.ModuleList([
            SwiGLU(config, dropout_prob=config.moe_expert_dropout_prob)
            for _ in range(num_experts)
        ])
        
        self.use_residual = config.moe_use_residual
        if self.use_residual:
            self.residual_mlp = SwiGLU(config, dropout_prob=config.moe_expert_dropout_prob)
            self.residual_mix = nn.Linear(config.hidden_size, 2, bias=True)
        
        self.moe_output_dropout_prob = config.moe_output_dropout_prob
        self.moe_output_dropout = (
            nn.Dropout(config.moe_output_dropout_prob)
            if config.moe_output_dropout_prob > 0
            else nn.Identity()
        )
    
    def _dispatch_experts(self, flat_hidden, gate_scores, expert_indices):
        mode = self.moe_dispatch
        if mode == "grouped":
            try:
                return self._dispatch_grouped(flat_hidden, gate_scores, expert_indices)
            except (RuntimeError, NotImplementedError) as e:
                if not getattr(self, "_grouped_warned", False):
                    print(f"[MoELayer] torch._grouped_mm unusable ({type(e).__name__}: "
                          f"{str(e)[:80]}); falling back to moe_dispatch='sorted'.")
                    self._grouped_warned = True
                self.moe_dispatch = "sorted"
                return self._dispatch_sorted(flat_hidden, gate_scores, expert_indices)
        if mode == "sorted":
            return self._dispatch_sorted(flat_hidden, gate_scores, expert_indices)
        return self._dispatch_loop(flat_hidden, gate_scores, expert_indices)

    def _dispatch_loop(self, flat_hidden, gate_scores, expert_indices):
        """Legacy per-expert Python loop."""
        output = torch.zeros_like(flat_hidden)
        for k in range(self.top_k):
            indices_k = expert_indices[:, k]
            scores_k = gate_scores[:, k]
            for expert_idx in range(self.num_experts):
                mask = (indices_k == expert_idx)
                if not mask.any():
                    continue
                token_indices = mask.nonzero(as_tuple=True)[0]
                expert_input = flat_hidden[token_indices]
                expert_output = self.experts[expert_idx](expert_input)
                weighted_output = expert_output * scores_k[token_indices].unsqueeze(-1)
                output[token_indices] += weighted_output
        output = output + sum(
            p.flatten()[0] * 0 for expert in self.experts for p in expert.parameters()
        )
        return output

    def _flatten_assignments(self, gate_scores, expert_indices):
        """(token, slot) pairs sorted by expert; shared by sorted/grouped dispatch."""
        num_tokens, top_k = expert_indices.shape
        flat_expert = expert_indices.reshape(-1)
        flat_score = gate_scores.reshape(-1)
        flat_token = torch.arange(num_tokens, device=expert_indices.device).repeat_interleave(top_k)
        order = torch.argsort(flat_expert, stable=True)
        sorted_token = flat_token[order]
        sorted_score = flat_score[order]
        counts = torch.bincount(flat_expert, minlength=self.num_experts)
        return sorted_token, sorted_score, counts

    def _dispatch_sorted(self, flat_hidden, gate_scores, expert_indices):
        """Sort tokens by expert, run each expert on its contiguous slice, scatter-add."""
        sorted_token, sorted_score, counts = self._flatten_assignments(gate_scores, expert_indices)
        sorted_input = flat_hidden.index_select(0, sorted_token)
        sizes = counts.tolist()
        chunks = torch.split(sorted_input, sizes, dim=0)
        expert_out = torch.cat(
            [self.experts[e](chunks[e]) for e in range(self.num_experts)], dim=0
        )
        expert_out = expert_out * sorted_score.to(expert_out.dtype).unsqueeze(-1)
        output = torch.zeros_like(flat_hidden)
        output.index_add_(0, sorted_token, expert_out.to(output.dtype))
        return output

    def _dispatch_grouped(self, flat_hidden, gate_scores, expert_indices):
        """Fully batched dispatch via torch._grouped_mm over stacked expert weights."""
        sorted_token, sorted_score, counts = self._flatten_assignments(gate_scores, expert_indices)
        sorted_input = flat_hidden.index_select(0, sorted_token).contiguous()
        offs = torch.cumsum(counts, 0).to(torch.int32)
        w_gate_up = torch.stack([e.gate_up_proj.weight for e in self.experts])
        w_down = torch.stack([e.down_proj.weight for e in self.experts])
        gate_up = torch._grouped_mm(
            sorted_input, w_gate_up.transpose(-2, -1), offs=offs
        )
        gate, up = gate_up.chunk(2, dim=-1)
        act = (F.silu(gate) * up).contiguous()
        down = torch._grouped_mm(
            act, w_down.transpose(-2, -1), offs=offs
        )
        down = down * sorted_score.to(down.dtype).unsqueeze(-1)
        output = torch.zeros_like(flat_hidden)
        output.index_add_(0, sorted_token, down.to(output.dtype))
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        species_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MoE layer with sparse index-based routing."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        input_dtype = hidden_states.dtype
        
        num_tokens = batch_size * seq_len
        flat_hidden = hidden_states.reshape(num_tokens, hidden_size)
        
        gate_scores, expert_indices, aux_loss = self.gate(
            flat_hidden, species_emb=species_emb, seq_len=seq_len
        )
        
        gate_scores = gate_scores.to(input_dtype)
        
        output = self._dispatch_experts(flat_hidden, gate_scores, expert_indices)

        if self.use_residual:
            residual_output = self.residual_mlp(flat_hidden)
            
            mix_logits = F.linear(
                flat_hidden,
                self.residual_mix.weight,
                self.residual_mix.bias,
            ).float()
            mix_coef = F.softmax(mix_logits, dim=-1).to(input_dtype)
            
            output = mix_coef[:, 0:1] * output + mix_coef[:, 1:2] * residual_output
        
        output = output.reshape(batch_size, seq_len, hidden_size)
        
        if self.moe_output_dropout_prob > 0 and self.training:
            output = self.moe_output_dropout(output)
        
        l_aux = aux_loss.float() * self.aux_loss_coef
        
        return output, l_aux


class GenomicDecoderLayer(nn.Module):
    """Single decoder layer for the Genomic Language Model."""
    def __init__(self, config: GenomicModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = GenomicAttention(config)
        self.input_layernorm = nn.RMSNorm(normalized_shape=config.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(normalized_shape=config.hidden_size)
        
        if not config.use_swiglu:
            raise NotImplementedError("Only SwiGLU is supported.")
        
        self.num_experts = config.get_num_experts_for_layer(layer_idx)
        self.use_moe = self.num_experts > 0
        
        if self.use_moe:
            self.ffn = MoELayer(config, num_experts=self.num_experts)
        else:
            self.ffn = SwiGLU(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        species_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        moe_loss = None
        if self.use_moe:
            hidden_states, moe_loss = self.ffn(hidden_states, species_emb=species_emb)
        else:
            hidden_states = self.ffn(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states, moe_loss


class MicroGlotSpeciesTable(nn.Module):
    """The pretraining species' 32-d Poincare vectors, addressed by `species_ids` (cf. VITS's speaker table).

    `weight` is a float32 buffer `[num_species, 32]` that follows device moves but never dtype casts
    (`torch_dtype=...`, `.to(torch.bfloat16)`, `.half()`): the prior is normalised in float32 and then cast.
    Row i belongs to line i of the tokenizer's species_vocab.txt.
    """

    def __init__(self, num_species: int, dim: int):
        super().__init__()
        self.register_buffer("weight", torch.zeros(num_species, dim, dtype=torch.float32))

    def _apply(self, fn, recurse=True):
        moved = fn(self.weight)
        self._buffers["weight"] = moved if moved.dtype == torch.float32 else self.weight.to(moved.device)
        return self

    def forward(self, species_ids: torch.LongTensor) -> torch.Tensor:
        return self.weight[species_ids]


class MicroGlotHead(nn.Module):
    """dense -> GELU -> out_proj: the Species-encoder's projection."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.dense = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.out_proj(self.act(self.dense(x)))


def _rope_modules(module: nn.Module):
    """The flash-attn rotary-embedding modules inside `module`."""
    return [m for m in module.modules() if isinstance(m, RotaryEmbedding)]


class MicroGlotPreTrainedModel(PreTrainedModel):
    """Base class of the MicroGlot models: weight initialisation, loading, and the species-encoder calls
    (delegated to the `MicroGlotModel` backbone)."""

    config_class = MicroGlotConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GenomicDecoderLayer"]
    _keys_to_ignore_on_load_missing = [r"rotary_emb"]
    # training-code checkpoints bundle the species encoder (here a separate model); base loads of a causal-LM
    # checkpoint carry lm_head.
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)species_encoder\.", r"^lm_head\."]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            std = math.sqrt(self.config.init_scale / fan_in)
            nn.init.trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, RotaryEmbedding):
            # inv_freq is a non-persistent buffer (not in the checkpoint): transformers>=5 re-creates it
            # uninitialised after loading and relies on _init_weights to fill it.
            dim, base = module.dim, float(module.base)
            module.inv_freq.copy_(1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, species_encoder=None, **kwargs):
        """The standard `from_pretrained`, plus `species_encoder`:

        species_encoder (`bool` or `str`, *optional*): `True` also loads and attaches the Species-encoder, as
            `model.load_species_encoder()` does; a repo id or a local path loads the encoder found there. To
            attach an encoder object you already loaded, call `model.set_species_encoder(encoder)`.
        """
        if species_encoder is not None and not isinstance(species_encoder, (bool, str, _os.PathLike)):
            raise TypeError(
                "species_encoder= takes True (this model's Species-encoder), a repo id or a local path. "
                "To attach an encoder object you already loaded, call model.set_species_encoder(encoder)."
            )
        if _DTYPE_KWARG == "torch_dtype" and "dtype" in kwargs:  # accept the newer `dtype=` before transformers 4.56
            kwargs.setdefault("torch_dtype", kwargs.pop("dtype"))
        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded  # output_loading_info=True: (model, info)
        # where this model came from: load_species_encoder() looks for the encoder there, at the same revision
        if pretrained_model_name_or_path is not None:  # None: built from config= and state_dict=
            source = _os.fspath(pretrained_model_name_or_path)
            if _os.path.isdir(source):
                source = _os.path.abspath(source)
            revision = getattr(model.config, "_commit_hash", None) or kwargs.get("revision")
            for m in (model, getattr(model, "model", None)):
                if isinstance(m, MicroGlotModel):
                    m.__dict__["_microglot_source"] = (source, revision)
        if species_encoder:
            hub = {k: kwargs[k] for k in ("cache_dir", "force_download", "local_files_only", "token", "proxies")
                   if k in kwargs}
            model.load_species_encoder(None if species_encoder is True else species_encoder, **hub)
        return loaded

    def to_bfloat16(self) -> "MicroGlotPreTrainedModel":
        """Convert model weights to bfloat16 (the species table stays float32)."""
        return self.to(torch.bfloat16)

    # -- species-encoder composition, delegated to the backbone (MicroGlotModel implements it) ----------
    def _microglot_backbone(self):
        return getattr(self, "model", None)

    @property
    def species_encoder(self):
        """The attached species encoder, or None."""
        return self._microglot_backbone().species_encoder

    def set_species_encoder(self, encoder, match_backbone: bool = True):
        """Attach a species encoder you loaded yourself (None detaches); see `MicroGlotModel.set_species_encoder`."""
        self._microglot_backbone().set_species_encoder(encoder, match_backbone=match_backbone)

    def get_species_encoder(self):
        """The attached species encoder, or None."""
        return self._microglot_backbone().species_encoder

    def load_species_encoder(self, *args, **kwargs):
        """Load and attach the species encoder; see `MicroGlotModel.load_species_encoder`."""
        return self._microglot_backbone().load_species_encoder(*args, **kwargs)

    def get_species_embeds(self, *args, **kwargs):
        """The species vectors the model would use; see `MicroGlotModel.get_species_embeds`."""
        return self._microglot_backbone().get_species_embeds(*args, **kwargs)

    def encode_species(self, species, unknown_species: str = "raise") -> torch.LongTensor:
        """Species name(s) -> `species_ids` on the model's device: a 0-d tensor for one name, `[n]` for a list.

        A thin wrapper over this repository's tokenizer (`tokenizer.convert_species_to_ids`; `None` -> -1,
        loose matching, unknown names raise `KeyError` unless `unknown_species="infer"`), loaded once from
        where the model was loaded and checked against `config.species_vocab_sha256`.
        """
        if not self.config.num_species:
            raise ValueError("This checkpoint has no species table (MicroGlot-plain or the species encoder).")
        tok = self.__dict__.get("_species_tokenizer")
        if tok is None:
            from transformers import AutoTokenizer

            source = self.config._name_or_path
            commit = getattr(self.config, "_commit_hash", None)
            try:
                tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True,
                                                    **({"revision": commit} if commit else {}))
            except (OSError, ValueError):
                tok = None
            if getattr(tok, "species_vocab_sha256", None) is None:
                raise ValueError(
                    f"Could not load the MicroGlot tokenizer from {source!r}. Load it with AutoTokenizer."
                    "from_pretrained('athanzli/MicroGlot', trust_remote_code=True) and call "
                    "tokenizer.convert_species_to_ids(names)."
                )
            if tok.species_vocab_sha256 != self.config.species_vocab_sha256:
                raise ValueError("The tokenizer's species list does not match this model's species table.")
            self.__dict__["_species_tokenizer"] = tok
        ids = tok.convert_species_to_ids(species, unknown_species=unknown_species)
        return torch.tensor(ids, dtype=torch.long, device=self.device)


class MicroGlotModel(MicroGlotPreTrainedModel):
    """Decoder-only backbone (`AutoModel`). Returns token states aligned with `input_ids`.

    Species inputs: `species_ids` (rows of the built-in table; -1 = infer), `species_embeds` (your own
    vectors), or nothing (every row inferred by the attached species encoder). The encoder is held by
    reference: not a submodule, so it is never in `parameters()`, `state_dict()` or a saved checkpoint and
    never trained, but `.to()/.half()/.cuda()/.float()` reach it.
    """

    def __init__(self, config: MicroGlotConfig, species_encoder=None):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GenomicDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(normalized_shape=config.hidden_size)
        self.embed_species = (
            MicroGlotSpeciesTable(config.num_species, config.species_emb_dim) if config.num_species else None
        )
        self.species_token_proj = (
            nn.Linear(config.species_emb_dim, config.hidden_size, bias=False)
            if config.prepend_species_token
            else None
        )
        self.__dict__["_species_encoder"] = None
        self.__dict__["_species_encoder_follow"] = True
        self.post_init()
        if species_encoder is not None:
            self.set_species_encoder(species_encoder)

    def _microglot_backbone(self):
        return self

    @property
    def species_encoder(self):
        """The attached species encoder, or None."""
        return self.__dict__.get("_species_encoder")

    def set_species_encoder(self, encoder, match_backbone: bool = True):
        """Attach (or with None detach) the encoder that infers species for rows without one.

        The encoder is put in eval mode with `requires_grad_(False)`.
        match_backbone=True (default): move/cast it to the backbone's device and dtype and give it exact rotary
        frequencies in the backbone buffer's dtype, so the pair computes exactly what the training code's
        combined model computed; later .to()/.half()/.cuda() calls on this model reach it too. match_backbone=False: leave it
        where it is (e.g. another GPU); inputs are moved to it and its output is cast back.
        """
        if encoder is not None:
            if self.config.species_emb_dim is None:
                raise ValueError(
                    "This checkpoint takes no species information (MicroGlot-plain, or the species encoder's own "
                    "backbone), so it has no use for a species encoder."
                )
            if not hasattr(encoder, "projection") or getattr(
                encoder.config, "species_encoder_embedding_dim", None
            ) != self.config.species_emb_dim:
                raise ValueError(
                    f"Expected the Species-encoder (athanzli/MicroGlot, subfolder='species_encoder') or another "
                    f"MicroGlotSpeciesEncoder "
                    f"with {self.config.species_emb_dim}-d output."
                )
            encoder.eval().requires_grad_(False)
            if match_backbone:
                ref = self.embed_tokens.weight
                encoder.to(device=ref.device, dtype=ref.dtype)
                ref_rope = _rope_modules(self)
                if ref_rope:
                    # The cast above also rounds the encoder's rotary frequencies, which flash-attn reads as
                    # they are when they are float32. Rebuild them exactly and give them the backbone buffer's
                    # dtype, as the encoder inside the training code's combined model had.
                    ref_inv = ref_rope[0].inv_freq
                    for m in _rope_modules(encoder):
                        inv_freq = 1.0 / (m.base ** (torch.arange(0, m.dim, 2, dtype=torch.float32) / m.dim))
                        m.inv_freq = inv_freq.to(device=ref_inv.device, dtype=ref_inv.dtype)
                        m._cos_cached, m._seq_len_cached = None, 0
        self.__dict__["_species_encoder"] = encoder
        self.__dict__["_species_encoder_follow"] = bool(match_backbone)

    def get_species_encoder(self):
        """The attached species encoder, or None."""
        return self.species_encoder

    def load_species_encoder(self, pretrained_model_name_or_path=None, *, subfolder=None, revision=None,
                             match_backbone=True, **kwargs):
        """Load the Species-encoder and attach it (see `set_species_encoder`); returns the encoder.

        Default: the folder `config.species_encoder_subfolder` ("species_encoder") of the repository or local
        folder this model was loaded from, at the same revision (or `config.species_encoder_repo` at
        `config.species_encoder_revision` if set), in the model's dtype. Other keyword arguments go to
        `from_pretrained` (e.g. `cache_dir`, `token`). It uses the encoder class in this code file.
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "Load the species encoder outside torch.inference_mode(): weights created inside it are "
                "inference tensors and cannot be moved or cast afterwards."
            )
        if self.config.species_emb_dim is None:
            raise ValueError("MicroGlot-plain takes no species information, so it has no species encoder.")
        path = pretrained_model_name_or_path
        if path is None:
            if self.config.species_encoder_repo is not None:
                path, revision = self.config.species_encoder_repo, revision or self.config.species_encoder_revision
            else:
                source, source_revision = self.__dict__.get("_microglot_source") or (self.config._name_or_path, None)
                path, revision = source, revision or source_revision
            subfolder = subfolder if subfolder is not None else self.config.species_encoder_subfolder
            if not path:
                raise ValueError("Pass the encoder's location, e.g. "
                                 "model.load_species_encoder('athanzli/MicroGlot', subfolder='species_encoder').")
        if subfolder:
            kwargs["subfolder"] = subfolder
        if revision is not None:
            kwargs["revision"] = revision
        if "torch_dtype" not in kwargs and "dtype" not in kwargs:
            kwargs[_DTYPE_KWARG] = self.embed_tokens.weight.dtype
        try:
            encoder = MicroGlotSpeciesEncoder.from_pretrained(path, **kwargs)
        except OSError as e:
            raise OSError(
                f"No Species-encoder found at {path!r}" + (f" (subfolder {subfolder!r})" if subfolder else "")
                + ". Load it from the Hub: model.load_species_encoder('athanzli/MicroGlot', subfolder='species_encoder')."
            ) from e
        self.set_species_encoder(encoder, match_backbone=match_backbone)
        return encoder

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        encoder = self.__dict__.get("_species_encoder")
        if encoder is not None and self.__dict__.get("_species_encoder_follow", True):
            encoder._apply(fn, recurse)
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    # -- species resolution -------------------------------------------------------------------------
    @torch.no_grad()
    def get_species_embeds(self, input_ids=None, attention_mask=None, species_ids=None, species_embeds=None,
                           normalize_species_embeds: bool = True):
        """The [batch, 32] unit vectors this model would condition on, in its dtype, without running the
        backbone (prior rows from the table, custom vectors, inferred rows from the attached encoder).

        Takes the same species arguments as `forward`; `input_ids` (and `attention_mask`) are only needed for
        rows the encoder infers. Use it to inspect inferred species or to precompute a `species_embeds`
        column (pass it back with `normalize_species_embeds=False` to reuse the exact vectors).
        """
        if input_ids is None:
            ref = species_ids if species_ids is not None else species_embeds
            if ref is None:
                raise ValueError("Pass input_ids, species_ids or species_embeds.")
            ref = torch.as_tensor(ref)
            n = (ref.shape[0] if ref.dim() > 0 else 1) if species_ids is not None else (
                ref.shape[0] if ref.dim() > 1 else 1)
            if species_ids is not None and bool((ref < 0).any()):
                raise ValueError("Rows to infer (species_ids == -1) need input_ids.")
            input_ids = torch.full((n, 1), self.config.bos_token_id, device=self.embed_tokens.weight.device)
        return self._species_vectors(input_ids, attention_mask, species_ids, species_embeds,
                                     normalize_species_embeds)

    def _species_vectors(self, input_ids, attention_mask, species_ids, species_embeds, normalize=True):
        """Unit-norm species vectors [batch, species_emb_dim] in the model dtype, or None (plain)."""
        dim = self.config.species_emb_dim
        if dim is None:
            if species_ids is not None or species_embeds is not None:
                raise ValueError(
                    "This checkpoint (MicroGlot-plain) takes no species information: drop the species "
                    "inputs, or load athanzli/MicroGlot to condition on a species."
                )
            return None
        if species_ids is not None and species_embeds is not None:
            raise ValueError("You cannot specify both species_ids and species_embeds at the same time.")
        batch, device, dtype = input_ids.shape[0], input_ids.device, self.embed_tokens.weight.dtype

        if species_embeds is not None:  # the training code's `species_emb=` path (plus a no-op device move)
            species_embeds = torch.as_tensor(species_embeds)
            if species_embeds.dim() == 1:
                species_embeds = species_embeds.unsqueeze(0)
            if species_embeds.dim() != 2 or species_embeds.shape[-1] != dim:
                raise ValueError(
                    f"species_embeds must be [{dim}] or [batch, {dim}], got {tuple(species_embeds.shape)}."
                )
            if species_embeds.shape[0] == 1 and batch > 1:
                species_embeds = species_embeds.expand(batch, -1)
            elif species_embeds.shape[0] != batch:
                raise ValueError(f"species_embeds has {species_embeds.shape[0]} rows for {batch} sequences.")
            species_embeds = species_embeds.to(device=device)
            if normalize:
                return F.normalize(species_embeds.to(torch.float32), dim=-1).to(dtype)
            norms = species_embeds.float().norm(dim=-1)
            if bool(((norms - 1).abs() > 1e-2).any()):
                raise ValueError(
                    "normalize_species_embeds=False expects unit vectors that are used as they are (e.g. "
                    "out.species_embeds or encoder(...).species_embeds)."
                )
            return species_embeds.to(dtype)

        vectors, unknown = None, None
        if species_ids is not None:
            species_ids = torch.as_tensor(species_ids, device=device)
            if species_ids.dim() == 0:
                species_ids = species_ids.expand(batch)
            if species_ids.shape != (batch,) or species_ids.dtype not in (torch.int64, torch.int32):
                raise ValueError(f"species_ids must be an integer tensor of shape [{batch}].")
            species_ids = species_ids.long()
            if self.embed_species is None:
                raise ValueError("This checkpoint has no species table; pass species_embeds instead.")
            n = self.embed_species.weight.shape[0]
            if bool(((species_ids < -1) | (species_ids >= n)).any()):
                raise ValueError(f"species_ids must lie in [-1, {n}); -1 marks an unknown species.")
            prior = self.embed_species(species_ids.clamp(min=0))
            vectors = F.normalize(prior.to(torch.float32), dim=-1).to(dtype)
            unknown = species_ids < 0
            if not bool(unknown.any()):
                return vectors

        encoder = self.species_encoder
        if encoder is None:
            raise ValueError(
                "No species given for some sequences and no species encoder is attached. Pass species= to "
                "the tokenizer (or species_ids / species_embeds), or attach the encoder with "
                "model.load_species_encoder(). For no species information, use MicroGlot-plain "
                "(subfolder='plain')."
            )
        rows = slice(None) if unknown is None else unknown
        enc_ids = input_ids[rows].to(encoder.device)
        enc_mask = None if attention_mask is None else attention_mask[rows].to(encoder.device)
        with torch.no_grad():
            inferred = encoder(input_ids=enc_ids, attention_mask=enc_mask).species_embeds
        inferred = inferred.to(device=device, dtype=dtype)
        if unknown is None:
            return inferred
        vectors = vectors.clone()
        vectors[unknown] = inferred
        return vectors

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        species_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        species_ids: Optional[torch.LongTensor] = None,
        normalize_species_embeds: bool = True,
        species_emb: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, MicroGlotModelOutput]:
        """Token states for right-padded `input_ids` `[batch, length]` (each row starts with [BOS]).

        Species (MicroGlot only; at most one of the two):
            species_ids (`LongTensor` `[batch]`, or 0-d for the whole batch): rows of the species table
                (`tokenizer(dna, species=...)` returns them); -1 = infer this row with the attached encoder.
            species_embeds (`FloatTensor` `[batch, 32]` or `[32]`): your own vectors, any float dtype and norm;
                normalised in float32 and cast to the model dtype. With `normalize_species_embeds=False` they
                must already be unit vectors and are used as they are (exact replay of `out.species_embeds`).
                `species_emb` (the training code's name) is also accepted.
            Neither: every row is inferred by the attached species encoder (ValueError if none is attached).
        output_hidden_states: also return the 24 hidden states. output_router_logits: compute `moe_loss`
            (default: on). return_dict=False returns the tuple `(last_hidden_state, hidden_states, moe_loss)`.
        The positional order is the training code's (the third argument is the species vector).
        """
        if not (input_ids.is_cuda and self.embed_tokens.weight.is_cuda):
            raise RuntimeError(
                "MicroGlot runs on an NVIDIA GPU (flash-attn's rotary kernel is CUDA-only): move the "
                "model and the inputs to a CUDA device, e.g. model.to('cuda')."
            )
        if species_emb is not None:
            if species_embeds is not None:
                raise ValueError("Pass species_embeds or species_emb (the same argument), not both.")
            species_embeds = species_emb
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None
            else self.config.use_moe
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if attention_mask is not None and not bool(attention_mask[:, 0].all()):
            raise ValueError(
                "MicroGlot needs right padding (every sequence starts with [BOS] at position 0); "
                "tokenize with padding_side='right'."
            )
        if attention_mask is None and self.config.pad_token_id is not None and bool(
            (input_ids == self.config.pad_token_id).any()
        ):
            raise ValueError("input_ids contain [PAD] but no attention_mask was given; pass the tokenizer's mask.")
        if (
            self.species_token_proj is not None
            and (species_ids is not None or species_embeds is not None or self.species_encoder is not None)
            and bool((input_ids[:, 0] != self.config.bos_token_id).any())
        ):
            raise ValueError(
                "With a species, every sequence must start with [BOS] (the species token goes right after it); "
                "tokenize with add_special_tokens=True."
            )
        seq_len = input_ids.shape[1]
        max_trained = getattr(self.config, "max_trained_length", 8192)
        if seq_len > max_trained:
            warnings.warn(
                f"Input sequence length ({seq_len} tokens) exceeds the maximum length seen during "
                f"pretraining ({max_trained} tokens); split long sequences into windows "
                "(tokenizer(..., truncation=True, max_length=8192, return_overflowing_tokens=True)).",
                UserWarning,
                stacklevel=3,
            )
        if torch.is_grad_enabled():
            # flash-attn rebuilds a cos/sin cache made under torch.inference_mode() only in training mode;
            # in eval mode it cannot be saved for backward. Drop it (it is rebuilt with the same values).
            for layer in self.layers:
                rope = layer.self_attn.rotary_emb
                if rope._cos_cached is not None and rope._cos_cached.is_inference():
                    rope._cos_cached = None

        species_emb = self._species_vectors(input_ids, attention_mask, species_ids, species_embeds,
                                            normalize_species_embeds)

        # ---- from here on: the forward used in training and benchmarking, unchanged ---------------
        hidden_states = self.embed_tokens(input_ids).to(self.embed_tokens.weight.dtype)

        if self.species_token_proj is not None and species_emb is not None:
            batch_size = hidden_states.size(0)
            device = hidden_states.device

            species_token = self.species_token_proj(species_emb).unsqueeze(1).to(hidden_states.dtype)

            bos_embedding = hidden_states[:, :1, :]
            rest_embeddings = hidden_states[:, 1:, :]

            hidden_states = torch.cat([bos_embedding, species_token, rest_embeddings], dim=1)

            if attention_mask is not None:
                species_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
                bos_mask = attention_mask[:, :1]
                rest_mask = attention_mask[:, 1:]
                attention_mask = torch.cat([bos_mask, species_mask, rest_mask], dim=1)

        all_hidden_states = () if output_hidden_states else None
        total_moe_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32) if output_router_logits else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    species_emb,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(hidden_states, attention_mask=attention_mask, species_emb=species_emb)

            hidden_states, moe_loss = layer_outputs

            if output_router_logits and moe_loss is not None:
                total_moe_loss = total_moe_loss + moe_loss

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        species_was_prepended = (
            self.species_token_proj is not None and species_emb is not None
        )
        if species_was_prepended:
            hidden_states = torch.cat(
                [hidden_states[:, :1, :], hidden_states[:, 2:, :]], dim=1
            )
            if all_hidden_states is not None:
                all_hidden_states = tuple(
                    torch.cat([hs[:, :1, :], hs[:, 2:, :]], dim=1)
                    for hs in all_hidden_states
                )

        if not return_dict:
            return (hidden_states, all_hidden_states, total_moe_loss)

        return MicroGlotModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            moe_loss=total_moe_loss,
            species_embeds=species_emb,
        )


class MicroGlotForCausalLM(MicroGlotPreTrainedModel):
    """MicroGlot with its next-token head (`AutoModelForCausalLM`, the released checkpoints' class).

    `lm_head` is tied to the input embeddings. Sequence generation is not supported.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # dict: required by transformers 5

    def __init__(self, config: MicroGlotConfig, species_encoder=None):
        super().__init__(config)
        self.model = MicroGlotModel(config, species_encoder=species_encoder)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def tie_weights(self, missing_keys=None, recompute_mapping=True, **kwargs):
        """Always tie lm_head to the input embeddings, as in training (also for configs without
        tie_word_embeddings); under transformers 5 also clear the tied key from the load report."""
        self.lm_head.weight = self.model.embed_tokens.weight
        if missing_keys is not None:
            missing_keys.discard("lm_head.weight")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        species_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        skip_lm_head: bool = False,
        species_ids: Optional[torch.LongTensor] = None,
        normalize_species_embeds: bool = True,
        species_emb: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, MicroGlotCausalLMOutput]:
        """Next-token logits `[batch, length, vocab]`; the species arguments are those of `MicroGlotModel.forward`.

        labels (`LongTensor` `[batch, length]`, -100 = ignored): `loss` = mean next-token cross-entropy, plus the
            router loss (`moe_loss`) unless `output_router_logits=False`. With a species token the first
            prediction (made at [BOS], before the model has seen the species) is excluded, as in pretraining.
        skip_lm_head: return only the final hidden state (in `hidden_states`).
        return_dict=False returns the tuple `([loss,] logits, hidden_states, moe_loss)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_router_logits = (
            output_router_logits if output_router_logits is not None
            else self.config.use_moe
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            species_ids=species_ids,
            species_embeds=species_embeds,
            species_emb=species_emb,
            normalize_species_embeds=normalize_species_embeds,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        if skip_lm_head:
            return MicroGlotCausalLMOutput(loss=None, logits=None, hidden_states=hidden_states)

        logits = self.lm_head(hidden_states)
        moe_aux_loss = outputs.moe_loss

        loss = None
        if labels is not None:
            species_was_prepended = self.config.prepend_species_token and outputs.species_embeds is not None
            if species_was_prepended:
                shift_logits = logits[..., 1:-1, :].contiguous()
                shift_labels = labels[..., 2:].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            ).float()

            if self.config.use_moe and moe_aux_loss is not None:
                loss = loss + moe_aux_loss

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.moe_loss)
            return (loss,) + output if loss is not None else output

        return MicroGlotCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            species_embeds=outputs.species_embeds,
        )



class MicroGlotSpeciesEncoder(MicroGlotPreTrainedModel):
    """Predicts a species embedding from DNA (folder species_encoder/ of athanzli/MicroGlot, `AutoModel`).

    A MicroGlot backbone without species input, pooled at the last real token (needs right padding), then a
    Linear -> GELU -> Linear projection to 32 dimensions (`embeddings`), L2-normalised in the model dtype
    (`species_embeds`, the vector MicroGlot is conditioned on).
    """

    config_class = MicroGlotSpeciesEncoderConfig

    def __init__(self, config: MicroGlotSpeciesEncoderConfig):
        super().__init__(config)
        if config.species_emb_dim is not None or config.prepend_species_token:
            raise ValueError("The species encoder's backbone takes no species input (species_emb_dim=None).")
        self.embedding_dim = config.species_encoder_embedding_dim
        self.pooling_strategy = config.species_encoder_pooling_strategy
        self.model = MicroGlotModel(config)
        self.projection_dropout = nn.Dropout(config.classifier_dropout_prob)
        self.projection = MicroGlotHead(config.hidden_size, config.hidden_size, config.species_encoder_embedding_dim)
        self.post_init()

    def load_species_encoder(self, *args, **kwargs):
        """Not available: this is the species encoder itself."""
        raise TypeError("This is the species encoder; it takes no species encoder of its own.")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MicroGlotSpeciesEncoderOutput]:
        """Species vectors for right-padded `input_ids` `[batch, length]` (one per sequence).

        return_dict=False returns `(species_embeds, embeddings, pooler_output, last_hidden_state,
        hidden_states)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_router_logits=self.config.use_moe,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        if self.pooling_strategy == "eos":
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                seq_lengths = seq_lengths.clamp(min=0)
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                pooled = hidden_states[batch_indices, seq_lengths]
            else:
                pooled = hidden_states[:, -1, :]
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)

        pooled = self.projection_dropout(pooled)
        raw = self.projection(pooled)
        species_embeds = F.normalize(raw, dim=-1)

        if not return_dict:
            return (species_embeds, raw, pooled, hidden_states, outputs.hidden_states)
        return MicroGlotSpeciesEncoderOutput(
            species_embeds=species_embeds,
            embeddings=raw,
            pooler_output=pooled,
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            moe_loss=outputs.moe_loss,
        )


# Training-code names (configs and auto_maps written by the training code keep resolving).
GenomicPreTrainedModel = MicroGlotPreTrainedModel
GenomicModel = MicroGlotModel
GenomicLMForCausalLM = MicroGlotForCausalLM
GenomicSpeciesEmbeddingEncoder = MicroGlotSpeciesEncoder
GenomicModelOutput = MicroGlotModelOutput

# save_pretrained() then copies this file and writes local auto_map entries, also for Hub-loaded models.
MicroGlotConfig.register_for_auto_class("AutoConfig")
MicroGlotSpeciesEncoderConfig.register_for_auto_class("AutoConfig")
MicroGlotModel.register_for_auto_class("AutoModel")
MicroGlotForCausalLM.register_for_auto_class("AutoModelForCausalLM")
MicroGlotSpeciesEncoder.register_for_auto_class("AutoModel")
