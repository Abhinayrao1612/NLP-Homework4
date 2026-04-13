"""
=============================================================
CS5760 Natural Language Processing - Spring 2026
Homework 4 - Q2: Mini Transformer Encoder for Sentences
Student : KOMATLAPALLI VENKATA NAGA SRI
ID      : 700773763
=============================================================

Description:
    Implements a single Transformer encoder block from scratch in PyTorch
    and processes a batch of 10 short sentences.

    Pipeline:
      1. Define 10 short NLP sentences as the dataset.
      2. Build a word-level vocabulary; index 0 is reserved for PAD.
      3. Tokenize and pad all sentences to MAX_LEN (length of longest sentence).
      4. Pass token IDs through an Embedding layer (d_model = 32).
      5. Add sinusoidal positional encoding:
             PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
             PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
      6. Multi-Head Self-Attention (2 heads) — Q = K = V = x.
      7. Add & Norm: residual connection + LayerNorm after attention.
      8. Feed-Forward sublayer: Linear(32→128) → ReLU → Linear(128→32).
      9. Add & Norm after feed-forward.
     10. Print contextual embeddings for sentence 1.
     11. Print text-based attention heatmap for sentence 1.

    Note on attention heatmap row sums:
      MAX_LEN = 5 (longest sentence: "attention is all you need").
      Sentence 1 ("i love nlp") has only 3 real tokens + 2 PAD positions.
      The heatmap prints only the 3 real token columns; the remaining
      attention weight is distributed to the 2 PAD positions (not shown).
      Full row sums across all 5 columns equal exactly 1.0.

    Key concepts demonstrated:
      - Sinusoidal PE: position info without learned parameters
      - Multi-head attention: each head attends in a different subspace
      - Add & Norm: residual + LayerNorm for gradient flow and stability
      - Feed-forward: position-wise MLP with inner dim = 4 x d_model
"""

import torch
import torch.nn as nn
import math

# ─────────────────────────────────────────────────────────
# Step 1: Dataset — 10 Short Sentences
# ─────────────────────────────────────────────────────────
sentences = [
    "i love nlp",
    "deep learning is fun",
    "transformers are powerful",
    "attention is all you need",
    "language models predict text",
    "nlp solves real problems",
    "embeddings capture meaning",
    "gradient descent trains models",
    "softmax gives probabilities",
    "words have context",
]

# ─────────────────────────────────────────────────────────
# Step 2: Vocabulary and Tokenization
# ─────────────────────────────────────────────────────────
# Index 0 is reserved for PAD; all real words start at index 1.
# padding_idx=0 in the Embedding ensures PAD vectors are always zero.
all_words  = sorted(set(" ".join(sentences).split()))
word2idx   = {w: i + 1 for i, w in enumerate(all_words)}   # 0 = PAD
vocab_size = len(word2idx) + 1   # +1 for PAD

# MAX_LEN = 5 ("attention is all you need" has 5 words)
MAX_LEN = max(len(s.split()) for s in sentences)

def tokenize(sentence):
    """
    Convert a sentence string to padded integer token IDs.

    Returns:
        ids   : list of int, length MAX_LEN (right-padded with 0)
        tokens: list of str (original words, no padding)
    """
    tokens = sentence.split()
    ids    = [word2idx[w] for w in tokens]
    ids   += [0] * (MAX_LEN - len(ids))   # right-pad with PAD index
    return ids, tokens

tokenized = [tokenize(s) for s in sentences]
ids_batch = torch.tensor([t[0] for t in tokenized], dtype=torch.long)   # (10, MAX_LEN)

print("=" * 55)
print("Input Tokens (first 3 sentences):")
for i in range(3):
    print(f"  S{i+1}: {tokenized[i][1]}")

# ─────────────────────────────────────────────────────────
# Step 3: Token Embedding
# ─────────────────────────────────────────────────────────
# padding_idx=0: PAD embeddings are always zero and receive no gradient.
D_MODEL   = 32
embedding = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
x = embedding(ids_batch)   # (10, MAX_LEN, D_MODEL)

# ─────────────────────────────────────────────────────────
# Step 4: Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────
def sinusoidal_pe(max_len, d_model):
    """
    Compute a fixed sinusoidal positional encoding matrix.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))   -- even dims
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))   -- odd  dims

    Args:
        max_len : maximum sequence length
        d_model : embedding dimension

    Returns:
        Tensor of shape (1, max_len, d_model) — broadcastable over batch
    """
    pe       = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()   # (max_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions → sin
    pe[:, 1::2] = torch.cos(position * div_term)   # odd  dimensions → cos
    return pe.unsqueeze(0)   # (1, max_len, d_model)

pe = sinusoidal_pe(MAX_LEN, D_MODEL)
x  = x + pe   # broadcast adds PE to every sentence in the batch

# ─────────────────────────────────────────────────────────
# Step 5: Multi-Head Self-Attention (2 heads) + Add & Norm
# ─────────────────────────────────────────────────────────
# Self-attention: Q = K = V = x (every token attends to all others).
# 2 heads each operate in a d_model/2 = 16-dim subspace, potentially
# specialising in different linguistic relationships (syntax, semantics).
N_HEADS = 2
mha     = nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=N_HEADS, batch_first=True)

# Forward: returns attended output + averaged attention weight matrix
attn_out, attn_weights = mha(x, x, x)
# attn_out    : (10, MAX_LEN, D_MODEL)
# attn_weights: (10, MAX_LEN, MAX_LEN)  — attention scores between all token pairs

# Add & Norm: residual stabilises gradient flow; LayerNorm stabilises activations
norm1 = nn.LayerNorm(D_MODEL)
x     = norm1(x + attn_out)

# ─────────────────────────────────────────────────────────
# Step 6: Feed-Forward Sublayer + Add & Norm
# ─────────────────────────────────────────────────────────
# Position-wise MLP applied identically at every token position.
# Inner dim = 4 x D_MODEL follows the original Transformer paper.
ff = nn.Sequential(
    nn.Linear(D_MODEL, D_MODEL * 4),   # expand to richer representation
    nn.ReLU(),                          # non-linearity
    nn.Linear(D_MODEL * 4, D_MODEL)    # project back to model dimension
)
ff_out = ff(x)

norm2 = nn.LayerNorm(D_MODEL)
x     = norm2(x + ff_out)   # final contextual embeddings: (10, MAX_LEN, D_MODEL)

# ─────────────────────────────────────────────────────────
# Step 7: Print Contextual Embeddings (Sentence 1)
# ─────────────────────────────────────────────────────────
print(f"\nFinal Contextual Embeddings — Sentence 1 (all tokens):")
print(f"  Sentence: \"{sentences[0]}\"")
for i, word in enumerate(tokenized[0][1]):
    vec = x[0, i].detach().numpy().round(3)
    print(f"  '{word}': {vec}")

# ─────────────────────────────────────────────────────────
# Step 8: Text-Based Attention Heatmap (Sentence 1)
# ─────────────────────────────────────────────────────────
# attn_weights[0] is the (MAX_LEN x MAX_LEN) attention matrix for sentence 1.
# Each row i shows how much token i attends to every other token j.
#
# IMPORTANT — row sums:
#   Sentence 1 "i love nlp" has 3 real tokens and 2 PAD positions.
#   MAX_LEN = 5, so attention is distributed across all 5 columns.
#   We only print the 3 real token columns below; the missing weight
#   goes to the 2 PAD positions. Full row sums across all 5 columns = 1.0.
words_s1 = tokenized[0][1]                     # ['i', 'love', 'nlp']
attn_s1  = attn_weights[0].detach().numpy()    # (MAX_LEN, MAX_LEN)
L        = len(words_s1)                        # 3

print(f"\nAttention Heatmap — Sentence 1: \"{sentences[0]}\"")
print("(row = query token, col = key token, value = attention weight)")
print("Note: 2 PAD columns not shown; remaining weight goes to PAD positions.")
print()

header = "         " + "  ".join(f"{w[:5]:>5}" for w in words_s1)
print(header)
for i, w in enumerate(words_s1):
    shown_sum = sum(attn_s1[i, j] for j in range(L))
    row = f"{w[:7]:>7}: " + "  ".join(f"{attn_s1[i, j]:.2f}" for j in range(L))
    row += f"  (shown sum={shown_sum:.2f}, full row sum=1.00)"
    print(row)
