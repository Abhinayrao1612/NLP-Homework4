"""
=============================================================
CS5760 Natural Language Processing - Spring 2026
Homework 4 - Q1: Character-Level RNN Language Model
Student : KOMATLAPALLI VENKATA NAGA SRI
ID      : 700773763
=============================================================

Description:
    Trains a character-level LSTM language model from scratch in PyTorch.
    The model learns to predict the next character given previous characters.

    Pipeline:
      1. Define a toy corpus of short repeated words (hello, help, world, here).
      2. Build a character-level vocabulary and encode the corpus as integers.
      3. Create a sliding-window CharDataset / DataLoader (seq_len=20).
      4. Define CharRNN: Embedding → LSTM → Linear (logits over vocab).
      5. Train with teacher forcing: true previous char fed as input each step.
         Loss = CrossEntropyLoss, Optimizer = Adam, gradient clipping max_norm=5.
      6. Generate 300 characters autoregressively at three temperatures:
             tau=0.7  →  confident, repetitive output
             tau=1.0  →  raw model distribution
             tau=1.2  →  diverse but noisier output
      7. Save training loss curve as rnn_loss_curve.png.

    Key concepts:
      - Teacher forcing vs. autoregressive inference (exposure bias)
      - Temperature scaling of logits before softmax
      - Gradient clipping to prevent exploding gradients in RNNs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────
# Step 1: Define Toy Corpus
# ─────────────────────────────────────────────────────────
# A small corpus of similar words so the model can learn repeating patterns.
corpus = (
    "hello world hello help here hello hello help"
    " hello world help here hello"
)

# ─────────────────────────────────────────────────────────
# Step 2: Build Character Vocabulary
# ─────────────────────────────────────────────────────────
# sorted(set(...)) gives a deterministic ordering every run.
chars      = sorted(set(corpus))
char2idx   = {c: i for i, c in enumerate(chars)}   # char → integer index
idx2char   = {i: c for c, i in char2idx.items()}   # integer index → char
vocab_size = len(chars)

print(f"Vocabulary ({vocab_size} chars): {chars}")

# ─────────────────────────────────────────────────────────
# Step 3: Encode Corpus as Integer Indices
# ─────────────────────────────────────────────────────────
encoded = [char2idx[c] for c in corpus]

# ─────────────────────────────────────────────────────────
# Step 4: Dataset — Sliding Window of Length SEQ_LEN
# ─────────────────────────────────────────────────────────
# x[i] = chars at [i, i+SEQ_LEN)        → input sequence
# y[i] = chars at [i+1, i+SEQ_LEN+1)   → target sequence (shifted by 1)
SEQ_LEN    = 20
BATCH_SIZE = 16

class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data    = data
        self.seq_len = seq_len

    def __len__(self):
        # Each window needs seq_len input + 1 look-ahead target char
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.seq_len],     dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y

dataset = CharDataset(encoded, SEQ_LEN)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ─────────────────────────────────────────────────────────
# Step 5: Define the CharRNN Model
# ─────────────────────────────────────────────────────────
# Architecture: Embedding → LSTM → Linear (logits over vocab)
# The LSTM hidden state h carries context forward through time.
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
        super(CharRNN, self).__init__()
        # Embedding maps each integer index to a dense vector
        self.embedding   = nn.Embedding(vocab_size, embed_dim)
        # LSTM processes sequences, maintaining hidden and cell states
        self.lstm        = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        # Linear projects each hidden state to logits over all characters
        self.fc          = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

    def forward(self, x, hidden=None):
        emb         = self.embedding(x)            # (batch, seq, embed_dim)
        out, hidden = self.lstm(emb, hidden)        # (batch, seq, hidden_size)
        logits      = self.fc(out)                  # (batch, seq, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        # LSTM needs both h (hidden state) and c (cell state), initialised to zeros
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h, c)

# ─────────────────────────────────────────────────────────
# Step 6: Training Hyperparameters and Setup
# ─────────────────────────────────────────────────────────
EMBED_DIM   = 32     # character embedding dimension
HIDDEN_SIZE = 128    # LSTM hidden state size (memory capacity)
EPOCHS      = 20     # full passes over training data
LR          = 0.003  # Adam learning rate

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = CharRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()   # combines LogSoftmax + NLLLoss

print(f"\nTraining on: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────────────────────
# Step 7: Training Loop with Teacher Forcing
# ─────────────────────────────────────────────────────────
# Teacher forcing: feed the true previous character as input at every step.
# This speeds convergence but creates a training-inference mismatch
# (exposure bias) since at inference the model feeds its own predictions.
train_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Fresh hidden state per batch — prevents cross-batch backprop
        hidden = model.init_hidden(x_batch.size(0), device)

        optimizer.zero_grad()

        logits, _ = model(x_batch, hidden)   # (batch, seq, vocab_size)

        # CrossEntropyLoss expects (N, C) predictions and (N,) targets
        loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))

        loss.backward()

        # Gradient clipping: rescale gradient if norm > 5 to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:>2}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ─────────────────────────────────────────────────────────
# Step 8: Temperature-Controlled Text Generation
# ─────────────────────────────────────────────────────────
# At inference the model runs autoregressively — each generated character
# is fed back as the next input (no teacher forcing).
# Temperature τ scales the logits before softmax:
#   τ < 1  →  sharper distribution, more repetitive/conservative
#   τ = 1  →  raw model distribution
#   τ > 1  →  flatter distribution, more creative but noisier
def generate(model, start_char, length=300, temperature=1.0):
    """
    Generate a character sequence autoregressively.

    Args:
        model       : trained CharRNN
        start_char  : seed character (must be in vocabulary)
        length      : number of characters to generate
        temperature : sampling temperature (tau)

    Returns:
        str: generated text starting with start_char
    """
    model.eval()
    with torch.no_grad():
        idx    = torch.tensor([[char2idx[start_char]]], dtype=torch.long).to(device)
        hidden = model.init_hidden(1, device)
        result = [start_char]

        for _ in range(length):
            logits, hidden = model(idx, hidden)      # (1, 1, vocab_size)
            logits = logits[0, 0] / temperature      # apply temperature scaling
            probs  = torch.softmax(logits, dim=-1)   # convert to probabilities
            # Multinomial sampling (not greedy argmax) for diversity
            next_idx = torch.multinomial(probs, 1).item()
            result.append(idx2char[next_idx])
            idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return "".join(result)

print("\n" + "=" * 55)
print("Temperature-Controlled Text Generation (seed = 'h')")
print("=" * 55)
for tau in [0.7, 1.0, 1.2]:
    text = generate(model, "h", length=300, temperature=tau)
    print(f"\ntau = {tau}:")
    print(text)

# ─────────────────────────────────────────────────────────
# Step 9: Plot and Save Training Loss Curve
# ─────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o", label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Character-Level RNN – Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("rnn_loss_curve.png", dpi=150)
plt.show()
print("\nLoss curve saved → rnn_loss_curve.png")
