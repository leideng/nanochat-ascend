"""
Unified Flash Attention interface with automatic FA3 / NPU FA / SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly. Backends:
- CUDA Hopper: Flash Attention 3 (FA3) when available.
- Ascend NPU (torch_npu 2.9+): npu_prompt_flash_attention for training (op-plugin); inference uses SDPA (contiguous cache).
- Else: PyTorch SDPA.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: FA3 (CUDA only) and NPU flash attention (Ascend)
# =============================================================================
def _load_flash_attention_3():
    """FA3 requires CUDA Hopper GPUs and is not available on Ascend NPU."""
    return None


def _load_npu_flash_attention():
    """
    Ascend NPU flash attention (torch_npu 2.9+ / op-plugin 7.3+).
    - Training: torch_npu.npu_prompt_flash_attention (preferred), else _npu_flash_attention
    - Inference: _npu_flash_attention_qlens (paged KV cache; we keep SDPA for contiguous cache)
    """
    try:
        import torch_npu
        if not torch.npu.is_available():
            return None
        # Prefer public API from op-plugin: npu_prompt_flash_attention
        if hasattr(torch_npu, "npu_prompt_flash_attention"):
            return torch_npu
        if hasattr(torch_npu, "_npu_flash_attention"):
            return torch_npu
        return None
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
_npu = _load_npu_flash_attention()
HAS_FA3 = _fa3 is not None
HAS_NPU_FA = _npu is not None
# Preferred NPU training op (op-plugin); see https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_prompt_flash_attention.md
NPU_PROMPT_FA = getattr(_npu, "npu_prompt_flash_attention", None) if _npu else None
# NPU inference with paged cache uses _npu_flash_attention_qlens (key_cache, value_cache, block_table, ...)
NPU_FA_QLENS = getattr(_npu, "_npu_flash_attention_qlens", None) if _npu else None

# Override for testing: set to 'fa3', 'npu', 'sdpa', or None (auto)
_override_impl = None


def _use_fa3():
    """Determine whether to use FA3 based on availability and override."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    return HAS_FA3  # auto


def _use_npu_fa(tensor):
    """Use NPU flash attention when inputs are on NPU and torch_npu provides it."""
    if _override_impl == 'sdpa':
        return False
    if _override_impl == 'npu':
        assert HAS_NPU_FA, "Cannot override to NPU FA: torch_npu flash attention not available"
        return True
    return HAS_NPU_FA and tensor.device.type == "npu"


# =============================================================================
# NPU flash attention (training: npu_prompt_flash_attention preferred, else _npu_flash_attention)
# =============================================================================
def _npu_flash_attn_func(q, k, v, causal, window_size):
    """
    Call torch_npu.npu_prompt_flash_attention (op-plugin) or _npu_flash_attention for training.
    q, k, v: (B, T, H, D). Sliding window: use SDPA unless pre_tokens/next_tokens supported.
    """
    B, T, H, D = q.shape
    num_heads = H
    num_kv_heads = k.size(2)
    scale_value = 1.0 / (D ** 0.5)
    # BNSD = (B, N, S, D): batch, num_heads, seq, head_dim
    q_bnsd = q.transpose(1, 2)   # (B, H, T, D)
    k_bnsd = k.transpose(1, 2)
    v_bnsd = v.transpose(1, 2)

    # npu_prompt_flash_attention (op-plugin 7.3): query, key, value, attn_mask=None, ..., num_heads, scale_value, pre_tokens, next_tokens, input_layout='BNSD', num_key_value_heads=0)
    if NPU_PROMPT_FA is not None:
        try:
            # Causal: pre_tokens = all previous (large), next_tokens = 0. Sliding window: pre_tokens=left, next_tokens=right.
            pre_tokens = 2147483647 if window_size[0] < 0 else min(window_size[0], 2147483647)
            next_tokens = 0 if window_size[1] < 0 else window_size[1]
            kv_heads = 0 if num_kv_heads == num_heads else num_kv_heads
            out = NPU_PROMPT_FA(
                q_bnsd, k_bnsd, v_bnsd,
                attn_mask=None,
                num_heads=num_heads,
                scale_value=scale_value,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                input_layout="BNSD",
                num_key_value_heads=kv_heads,
            )
            return out.transpose(1, 2)  # (B, N, S, D) -> (B, T, H, D)
        except (TypeError, AttributeError, RuntimeError):
            pass

    # Fallback: _npu_flash_attention (no sliding window)
    if window_size[0] >= 0 or window_size[1] >= 0:
        return None
    device = q.device
    dtype = q.dtype
    if causal:
        row = torch.arange(T, device=device, dtype=torch.int32).unsqueeze(1)
        col = torch.arange(T, device=device, dtype=torch.int32).unsqueeze(0)
        mask = (col <= row).to(dtype)
        mask = torch.where(mask, torch.zeros(1, device=device, dtype=dtype), torch.full((1,), float("-inf"), device=device, dtype=dtype))
        mask = mask.unsqueeze(0).unsqueeze(0)
    else:
        mask = None
    for attempt in [
        lambda: _npu._npu_flash_attention(q, k, v, attn_mask=mask, scale=scale_value),
        lambda: _npu._npu_flash_attention(q, k, v, attn_mask=mask),
        lambda: _npu._npu_flash_attention(q, k, v, scale_value=scale_value),
        lambda: _npu._npu_flash_attention(q, k, v),
    ]:
        try:
            return attempt()
        except (TypeError, AttributeError):
            continue
    return None


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # Ascend NPU: torch_npu._npu_flash_attention (no sliding window; SDPA used if window set)
    if _use_npu_fa(q):
        out = _npu_flash_attn_func(q, k, v, causal, window_size)
        if out is not None:
            return out

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.
    On NPU, torch_npu._npu_flash_attention_qlens exists but expects paged KV cache
    (block_table, key_cache/value_cache in block layout); we use SDPA with contiguous cache.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)

# -----------------------------------------------------------------------------
# Ascend NPU (torch_npu 2.9+ / op-plugin 7.3+) flash-attention-style ops (not FA3; FA3 is CUDA-only):
# - Training: torch_npu.npu_prompt_flash_attention(query, key, value, ..., num_heads, scale_value,
#       pre_tokens, next_tokens, input_layout='BNSD', num_key_value_heads). See op-plugin docs:
#       https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_prompt_flash_attention.md
# - Fallback: torch_npu._npu_flash_attention(query, key, value, attn_mask, scale)
# - Inference (paged KV): torch_npu._npu_flash_attention_qlens(...). We use SDPA with contiguous cache.
