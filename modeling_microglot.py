"""Genomic language model: decoder-only transformer architecture."""

import math
import warnings
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass

import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
try:
    from flash_attn.layers.rotary import RotaryEmbedding

    _MICROGLOT_FLASH_ROPE = True
except ImportError:  # pragma: no cover
    _MICROGLOT_FLASH_ROPE = False

    class RotaryEmbedding(nn.Module):
        """Pure-PyTorch drop-in for flash_attn.layers.rotary.RotaryEmbedding."""

        def __init__(self, dim, base=10000.0, interleaved=False, scale_base=None, device=None):
            super().__init__()
            if interleaved or scale_base is not None:
                raise NotImplementedError(
                    "MicroGlot's RoPE fallback supports interleaved=False and scale_base=None only"
                )
            self.dim, self.base, self.interleaved, self.scale = dim, float(base), False, None
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None

        def _update(self, seqlen, device):
            if (
                self._cos_cached is not None
                and seqlen <= self._seq_len_cached
                and self._cos_cached.device == device
            ):
                return
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
            self._cos_cached = torch.cos(freqs)
            self._sin_cached = torch.sin(freqs)

        @staticmethod
        def _rotate(x, cos, sin):
            half = x.shape[-1] // 2
            xf = x.float()
            x1, x2 = xf[..., :half], xf[..., half:]
            c = cos[None, :, None, :].float()
            s = sin[None, :, None, :].float()
            return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)

        def forward(self, qkv, kv=None, seqlen_offset=0, max_seqlen=None, num_heads_q=None):
            if kv is not None or num_heads_q is None:
                raise NotImplementedError(
                    "MicroGlot's RoPE fallback only supports the packed-qkv call with num_heads_q"
                )
            seqlen = qkv.shape[1]
            offset = seqlen_offset if isinstance(seqlen_offset, int) else 0
            self._update(max_seqlen or (seqlen + offset), qkv.device)
            cos = self._cos_cached[offset : offset + seqlen]
            sin = self._sin_cached[offset : offset + seqlen]
            n_q = num_heads_q
            n_kv = (qkv.shape[2] - n_q) // 2
            q, k, v = qkv[:, :, :n_q], qkv[:, :, n_q : n_q + n_kv], qkv[:, :, n_q + n_kv :]
            return torch.cat([self._rotate(q, cos, sin), self._rotate(k, cos, sin), v], dim=2)

from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutputWithPast,
    ModelOutput,
)


@dataclass
class GenomicModelOutput(ModelOutput):
    """Output type for GenomicModel with optional MoE auxiliary loss."""
    last_hidden_state: torch.FloatTensor = None
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


class GenomicModelConfig(PretrainedConfig):
    """Configuration for the Genomic Language Model (Decoder-Only, Causal LM)."""
    
    model_type = "genomic_lm"

    auto_map = {
        "AutoConfig": "model.GenomicModelConfig",
        "AutoModel": "model.GenomicModel",
        "AutoModelForCausalLM": "model.GenomicLMForCausalLM",
    }

    keys_to_ignore_at_inference = []
    
    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 1024,
        num_hidden_layers: int = 12,
        num_query_heads: int = 16,
        num_kv_heads: int = 2,
        intermediate_size: Optional[int] = None,
        attention_dropout_prob: float = 0.0,
        classifier_dropout_prob: float = 0.1,
        attn_output_dropout_prob: float = 0.0,
        moe_output_dropout_prob: float = 0.0,
        ffn_dropout_prob: float = 0.0,
        moe_expert_dropout_prob: float = 0.0,
        rope_theta: float = 500000.0,
        max_trained_length: int = 8192,
        init_scale: float = 0.1,
        use_swiglu: bool = True,
        gradient_checkpointing: bool = True,
        use_flash_attention: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        num_experts_per_tok: int = 1,
        router_aux_loss_coef: float = 0.01,
        moe_layer_experts: Optional[List[int]] = None,
        moe_use_residual: bool = True,
        moe_capacity_factor: float = 1.25,
        moe_eval_capacity_factor: float = 2.0,
        moe_min_capacity: int = 4,
        moe_noisy_gate_policy: Optional[str] = None,
        moe_drop_tokens: bool = True,
        moe_use_rts: bool = True,
        gate_softmax_over_all_experts: bool = False,
        moe_dispatch: str = "loop",
        species_emb_dim: Optional[int] = None,
        use_species_encoder: bool = False,
        prepend_species_token: bool = False,
        species_encoder_embedding_dim: int = 32,
        species_encoder_pooling_strategy: str = "eos",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs
        )
        
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
        self.use_species_encoder = use_species_encoder
        self.prepend_species_token = prepend_species_token
        self.species_encoder_embedding_dim = species_encoder_embedding_dim
        self.species_encoder_pooling_strategy = species_encoder_pooling_strategy

        if self.prepend_species_token and self.species_emb_dim is None:
            raise ValueError(
                "species_emb_dim must be specified when prepend_species_token=True. "
                "Set species_emb_dim to the dimension of your species embeddings."
            )
        
        if self.use_species_encoder and self.species_emb_dim is None:
            raise ValueError(
                "species_emb_dim must be specified when use_species_encoder=True. "
                "Set species_emb_dim to the desired species embedding dimension."
            )
        if self.use_species_encoder and self.species_emb_dim != self.species_encoder_embedding_dim:
            raise ValueError(
                f"When use_species_encoder=True, species_emb_dim ({self.species_emb_dim}) must "
                f"equal species_encoder_embedding_dim ({self.species_encoder_embedding_dim})."
            )
        
        assert hidden_size % num_query_heads == 0
        self.head_dim = hidden_size // num_query_heads
        
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        self.num_kv_groups = num_query_heads // num_kv_heads
        
        self._validate_moe_config()
    
    def _validate_moe_config(self):
        """Validate MoE configuration for consistency and correctness."""
        if not self.use_moe:
            return

        if len(self.moe_layer_experts) != self.num_hidden_layers:
            raise ValueError(
                f"moe_layer_experts length ({len(self.moe_layer_experts)}) must equal "
                f"num_hidden_layers ({self.num_hidden_layers}). "
                f"Each layer needs an expert count specification (0 for dense, >0 for MoE)."
            )

        for i, num_exp in enumerate(self.moe_layer_experts):
            if not isinstance(num_exp, int) or num_exp < 0:
                raise ValueError(
                    f"moe_layer_experts[{i}] = {num_exp} is invalid. "
                    f"Must be a non-negative integer (0 for dense, >0 for MoE)."
                )

        for i, num_exp in enumerate(self.moe_layer_experts):
            if num_exp > 0 and num_exp < self.num_experts_per_tok:
                raise ValueError(
                    f"moe_layer_experts[{i}] = {num_exp} is less than num_experts_per_tok "
                    f"({self.num_experts_per_tok}). Each MoE layer needs at least "
                    f"{self.num_experts_per_tok} experts for top-k routing."
                )
    
    def get_num_experts_for_layer(self, layer_idx: int) -> int:
        """Get the number of experts for a specific layer."""
        if not self.use_moe:
            return 0

        if layer_idx < 0 or layer_idx >= len(self.moe_layer_experts):
            raise IndexError(
                f"layer_idx {layer_idx} is out of range for moe_layer_experts "
                f"(length {len(self.moe_layer_experts)})"
            )
        
        return self.moe_layer_experts[layer_idx]
    
    def get_parameter_counts(self) -> "MoEParameterCounts":
        """Get parameter counts for this configuration."""
        return compute_moe_parameter_counts(self)
    
    def print_parameter_summary(self) -> None:
        """Print a detailed parameter summary for this configuration."""
        print_model_parameter_summary(self)

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
        
        if self.config.use_flash_attention and query_states.is_cuda:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=True,
                    enable_gqa=True,
                )
        else:
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

class GenomicPreTrainedModel(PreTrainedModel):
    """Base class for Genomic models, providing weight initialization and common utilities."""
    
    config_class = GenomicModelConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GenomicDecoderLayer"]
    _keys_to_ignore_on_load_missing = [r"rotary_emb"]
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            std = math.sqrt(self.config.init_scale / fan_in)
            nn.init.trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
    
    def to_bfloat16(self) -> "GenomicPreTrainedModel":
        """Convert model weights to bfloat16 for memory-efficient training."""
        return self.to(torch.bfloat16)

class GenomicModel(GenomicPreTrainedModel):
    """Genomic Language Model - decoder-only transformer backbone."""
    def __init__(self, config: GenomicModelConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )
        self.layers = nn.ModuleList([
            GenomicDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(normalized_shape=config.hidden_size)
        
        if config.use_species_encoder:
            self.species_encoder = GenomicSpeciesEmbeddingEncoder(config)
            for param in self.species_encoder.parameters():
                param.requires_grad = False
        else:
            self.species_encoder = None
        
        if config.prepend_species_token:
            self.species_token_proj = nn.Linear(config.species_emb_dim, config.hidden_size, bias=False)
        else:
            self.species_token_proj = None
        
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        species_emb: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """Forward pass through the decoder-only transformer."""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None 
            else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None
            else self.config.use_moe
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        seq_len = input_ids.shape[1]
        max_trained = getattr(self.config, "max_trained_length", 8192)
        if seq_len > max_trained:
            warnings.warn(
                f"Input sequence length ({seq_len} tokens) exceeds the maximum length "
                f"seen during pretraining ({max_trained} tokens). RoPE positional "
                f"encodings will extrapolate beyond the trained range, which may "
                f"silently degrade output quality. Consider using "
                f"model.chunk_and_encode() to process long sequences in chunks.",
                UserWarning,
                stacklevel=3,
            )

        hidden_states = self.embed_tokens(input_ids).to(self.embed_tokens.weight.dtype)

        if species_emb is not None:
            expected_dim = self.config.species_emb_dim
            if species_emb.dim() == 1:
                species_emb = species_emb.unsqueeze(0)
            if species_emb.dim() != 2:
                raise ValueError(
                    f"species_emb must be 1-D [{expected_dim}] or 2-D [batch, {expected_dim}], "
                    f"but has shape {tuple(species_emb.shape)}."
                )
            if species_emb.shape[-1] != expected_dim:
                raise ValueError(
                    f"species_emb must have exactly {expected_dim} dimensions, but got "
                    f"{species_emb.shape[-1]} (shape {tuple(species_emb.shape)})."
                )
            batch_size_in = input_ids.shape[0]
            if species_emb.shape[0] == 1 and batch_size_in > 1:
                species_emb = species_emb.expand(batch_size_in, -1)
            elif species_emb.shape[0] != batch_size_in:
                raise ValueError(
                    f"species_emb has batch size {species_emb.shape[0]} but input_ids has "
                    f"{batch_size_in}. Pass one vector per sequence, or a single vector to "
                    f"apply to the whole batch."
                )
            species_emb = F.normalize(
                species_emb.to(torch.float32), dim=-1
            ).to(self.embed_tokens.weight.dtype)

        if species_emb is None and self.species_encoder is not None:
            with torch.no_grad():
                encoder_output = self.species_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            species_emb = F.normalize(encoder_output["embeddings"], dim=-1)

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
        
        return GenomicModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            moe_loss=total_moe_loss,
        )

class GenomicLMForCausalLM(GenomicPreTrainedModel, GenerationMixin):
    """Genomic Language Model for causal (autoregressive) language modeling."""
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: GenomicModelConfig):
        super().__init__(config)
        self.model = GenomicModel(config)
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
    
    def tie_weights(self, **kwargs):
        """Tie input embeddings to output lm_head weights."""
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        species_emb: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        skip_lm_head: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass with optional causal LM loss computation."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        output_router_logits = (
            output_router_logits if output_router_logits is not None
            else self.config.use_moe
        )
        
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            species_emb=species_emb,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state

        if skip_lm_head:
            return CausalLMOutputWithPast(loss=None, logits=None, hidden_states=hidden_states)

        logits = self.lm_head(hidden_states)

        moe_aux_loss = outputs.moe_loss if return_dict else None
        if not return_dict and len(outputs) > 2:
            moe_aux_loss = outputs[2] if outputs[2] is not None else None

        loss = None
        if labels is not None:
            species_was_prepended = self.config.prepend_species_token and (
                species_emb is not None or self.model.species_encoder is not None
            )
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
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if return_dict else None,
        )
    
    @torch.no_grad()
    def chunk_and_encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        species_emb: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode a sequence of any length into a single fixed-size embedding."""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"chunk_and_encode expects input_ids of shape [1, seq_len], "
                f"got {list(input_ids.shape)}"
            )

        if max_length is None:
            max_length = getattr(self.config, "max_trained_length", 8192)

        seq_len = input_ids.shape[1]
        device = input_ids.device
        bos_id = self.config.bos_token_id
        eos_id = self.config.eos_token_id

        def _mean_pool_content(hidden_states, chunk_len):
            """Mean-pool over content positions (exclude BOS at 0, keep EOS at end)."""
            if chunk_len <= 1:
                return hidden_states.mean(dim=1)
            content = hidden_states[:, 1:, :]
            return content.mean(dim=1)

        if seq_len <= max_length:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                species_emb=species_emb,
                return_dict=True,
            )
            return _mean_pool_content(outputs.last_hidden_state, seq_len)

        all_ids = input_ids[0]
        has_bos = (all_ids[0].item() == bos_id)
        has_eos = (all_ids[-1].item() == eos_id)

        content_start = 1 if has_bos else 0
        content_end = seq_len - 1 if has_eos else seq_len
        content_ids = all_ids[content_start:content_end]

        content_per_chunk = max_length - 2
        if content_per_chunk < 1:
            raise ValueError(f"max_length ({max_length}) must be >= 3 to fit BOS + content + EOS")

        chunk_embeddings = []
        for i in range(0, len(content_ids), content_per_chunk):
            chunk_content = content_ids[i:i + content_per_chunk]

            chunk_ids = torch.cat([
                torch.tensor([bos_id], device=device),
                chunk_content,
                torch.tensor([eos_id], device=device),
            ]).unsqueeze(0)

            chunk_mask = torch.ones_like(chunk_ids)

            outputs = self.model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                species_emb=species_emb,
                return_dict=True,
            )
            emb = _mean_pool_content(outputs.last_hidden_state, chunk_ids.shape[1])
            chunk_embeddings.append(emb)

        embedding = torch.stack(chunk_embeddings, dim=0).mean(dim=0)
        return embedding

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Prepare inputs for generation (used by HuggingFace generate())."""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class GenomicSpeciesEmbeddingEncoder(GenomicPreTrainedModel):
    """Genomic Species Embedding Encoder - predicts species embeddings from sequences."""

    def __init__(self, config: GenomicModelConfig):
        super().__init__(config)
        self.embedding_dim = config.species_encoder_embedding_dim
        self.pooling_strategy = config.species_encoder_pooling_strategy

        import copy
        inner_config = copy.deepcopy(config)
        inner_config.use_species_encoder = False
        inner_config.prepend_species_token = False
        self.model = GenomicModel(inner_config)

        self.projection_dropout = nn.Dropout(config.classifier_dropout_prob)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.species_encoder_embedding_dim),
        )

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass to get predicted species embeddings."""
        return_dict = return_dict if return_dict is not None else True

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
        embeddings = self.projection(pooled)

        moe_loss = outputs.moe_loss if hasattr(outputs, 'moe_loss') else None

        return {
            "embeddings": embeddings,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
            "moe_loss": moe_loss,
        }


