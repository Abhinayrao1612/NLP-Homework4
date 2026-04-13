"""
=============================================================
CS5760 Natural Language Processing - Spring 2026
Homework 4 - Q3: Scaled Dot-Product Attention
Student : KOMATLAPALLI VENKATA NAGA SRI
ID      : 700773763
=============================================================

Description:
    Implements the core scaled dot-product attention mechanism:

        Attention(Q, K, V) = softmax( QKᵀ / sqrt(d_k) ) * V

    This is the fundamental building block of the Transformer architecture
    (Vaswani et al., "Attention Is All You Need", 2017).

    Two tests are run:

    TEST 1 — Basic Attention (no mask):
      Random Q, K, V of shape (4, 8) with manual_seed=42.
      Prints raw vs scaled score statistics (softmax stability check),
      full attention weight matrix (rows verified to sum to 1.0),
      and output vectors (weighted sum of values).

    TEST 2 — Causal (Decoder) Masked Attention:
      Same Q, K, V with a lower-triangular causal mask applied.
      Positions where mask==0 are set to -inf before softmax → weight=0.
      Demonstrates the autoregressive property: token i attends only to
      positions <= i (cannot see future tokens).
      Row sums still equal 1.0 even with masking.

    Key formula steps:
      1. raw_scores    = Q @ Kᵀ
      2. scaled_scores = raw_scores / sqrt(d_k)   ← prevents softmax saturation
      3. (optional) mask future positions to -inf
      4. attn_weights  = softmax(scaled_scores, dim=-1)
      5. output        = attn_weights @ V
"""

import torch
import torch.nn.functional as F
import math

# ─────────────────────────────────────────────────────────
# Core Function: Scaled Dot-Product Attention
# ─────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Formula:
        Attention(Q, K, V) = softmax( QKᵀ / sqrt(d_k) ) * V

    Args:
        Q    (Tensor): Query  matrix, shape (..., seq_len, d_k)
        K    (Tensor): Key    matrix, shape (..., seq_len, d_k)
        V    (Tensor): Value  matrix, shape (..., seq_len, d_v)
        mask (Tensor): Optional boolean mask (int). Positions where mask==0
                       are filled with -inf before softmax, giving weight≈0.
                       Used for causal / decoder masking.

    Returns:
        output        (Tensor): Attended output,         shape (..., seq_len, d_v)
        attn_weights  (Tensor): Attention weight matrix, shape (..., seq_len, seq_len)
        raw_scores    (Tensor): Unscaled QKᵀ scores,    same shape as attn_weights
        scaled_scores (Tensor): Scores after /sqrt(d_k), same shape
    """
    d_k = Q.size(-1)   # key/query dimension

    # Step 1: Raw dot products — QKᵀ
    # Each query is compared against all keys. High dot product = high relevance.
    raw_scores = torch.matmul(Q, K.transpose(-2, -1))   # (..., seq_len, seq_len)

    # Step 2: Scale by 1/sqrt(d_k)
    # Without scaling, dot products grow with variance d_k, pushing softmax
    # into saturation (near-zero gradients). Dividing by sqrt(d_k) keeps
    # variance ~1 regardless of key dimension size.
    scaled_scores = raw_scores / math.sqrt(d_k)

    # Step 3: Apply causal mask (optional)
    # For decoder self-attention, future positions must be blocked so the model
    # cannot peek ahead. mask==0 positions become -inf → softmax gives weight≈0.
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))

    # Step 4: Softmax → attention weights
    # Applied along the key dimension (dim=-1). Each row sums to 1.0.
    attn_weights = F.softmax(scaled_scores, dim=-1)   # (..., seq_len, seq_len)

    # Step 5: Weighted sum of values
    # Each output position is a convex combination of all value vectors,
    # weighted by how strongly each query attends to each key.
    output = torch.matmul(attn_weights, V)   # (..., seq_len, d_v)

    return output, attn_weights, raw_scores, scaled_scores


# ─────────────────────────────────────────────────────────
# TEST 1: Basic Attention with Random Inputs (no mask)
# ─────────────────────────────────────────────────────────
torch.manual_seed(42)   # fixed seed for reproducibility

SEQ_LEN = 4   # number of tokens in the sequence
D_K     = 8   # key / query dimension (d_k)
D_V     = 8   # value dimension (d_v)

Q = torch.randn(SEQ_LEN, D_K)
K = torch.randn(SEQ_LEN, D_K)
V = torch.randn(SEQ_LEN, D_V)

output, attn_weights, raw_scores, scaled_scores = scaled_dot_product_attention(Q, K, V)

print("=" * 60)
print("TEST 1 — Basic Scaled Dot-Product Attention")
print("=" * 60)
print(f"\nInput shapes:")
print(f"  Q = {tuple(Q.shape)}  (seq_len={SEQ_LEN}, d_k={D_K})")
print(f"  K = {tuple(K.shape)}  (seq_len={SEQ_LEN}, d_k={D_K})")
print(f"  V = {tuple(V.shape)}  (seq_len={SEQ_LEN}, d_v={D_V})")
print(f"\nScale factor: 1 / sqrt({D_K}) = {1/math.sqrt(D_K):.4f}")

# ── Softmax stability check ──────────────────────────────
print("\n─── Softmax Stability Check ────────────────────────────")
print(f"Raw QKᵀ scores  — max: {raw_scores.max().item():.4f},  "
      f"min: {raw_scores.min().item():.4f},  "
      f"std: {raw_scores.std().item():.4f}")
print(raw_scores.detach().numpy().round(4))

print(f"\nScaled QKᵀ/sqrt(d_k) — max: {scaled_scores.max().item():.4f},  "
      f"min: {scaled_scores.min().item():.4f},  "
      f"std: {scaled_scores.std().item():.4f}")
print(scaled_scores.detach().numpy().round(4))

print("\nObservation: scaling reduces std by factor sqrt(d_k), keeping softmax")
print("in a well-distributed regime with healthy gradients during training.")

# ── Attention weight matrix ──────────────────────────────
print("\n─── Attention Weight Matrix (after softmax) ────────────")
print(attn_weights.detach().numpy().round(4))

row_sums = attn_weights.sum(dim=-1).detach().numpy().round(6)
print(f"\nRow sums (all must equal 1.0): {row_sums}")
assert all(abs(s - 1.0) < 1e-5 for s in row_sums), "Row sums must be 1.0!"
print("✓ All rows sum to 1.0")

# ── Output vectors ───────────────────────────────────────
print("\n─── Output Vectors (weighted sum of Values) ────────────")
print(output.detach().numpy().round(4))


# ─────────────────────────────────────────────────────────
# TEST 2: Causal (Decoder) Masked Attention
# ─────────────────────────────────────────────────────────
# A causal mask is a lower-triangular boolean matrix:
#   mask[i, j] = 1  if j <= i  (token i CAN attend to token j)
#   mask[i, j] = 0  if j >  i  (token i CANNOT attend to future token j)
# This enforces autoregressive generation in the Transformer decoder.
print("\n" + "=" * 60)
print("TEST 2 — Causal (Decoder) Masked Attention")
print("=" * 60)

causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).int()
print("\nCausal mask (lower-triangular):  1 = can attend, 0 = blocked (→ -inf)")
print(causal_mask.numpy())

output_m, attn_m, raw_m, scaled_m = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

print("\nScaled scores after applying mask (upper triangle = -inf → weight=0):")
print(scaled_m.detach().numpy().round(4))

print("\nAttention weights with causal mask:")
print(attn_m.detach().numpy().round(4))

row_sums_m = attn_m.sum(dim=-1).detach().numpy().round(6)
print(f"\nRow sums (masked): {row_sums_m}")
print("✓ Rows still sum to 1.0 even with masking")

print("\n─── Masked Output Vectors ──────────────────────────────")
print(output_m.detach().numpy().round(4))

print("\nObservation:")
print("  Token 0 attends only to itself (weight = 1.0).")
print("  Token 1 attends to positions 0 and 1 only.")
print("  Token 3 (last) attends to all positions — full context.")
print("  Upper-triangular entries are exactly 0, enforcing the")
print("  autoregressive property required in Transformer decoders.")
