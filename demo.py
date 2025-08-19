# demo.py
import numpy as np
import os, sys
sys.path.append('..')
from core.attention import MultiHeadAttention
from core.positional import PositionalEncoding
from core.norm import LayerNorm
from core.ffn import FeedForward
from core.transformer import TransformerBlock
from models.gpt import NanoGPT
from utils.tokenizer import SimpleTokenizer

# Corpus
corpus = [
    "Alice was beginning to get very tired of sitting by her sister on the bank",
    "What is a Caucus-race? said Alice; not that she wanted to know",
    "The Rabbit started violently and vanished into a hole"
]

# Initialize tokenizer
tokenizer = SimpleTokenizer(corpus)
vocab_size = len(tokenizer.vocab)

# Prepare dataset
def prepare_dataset(corpus, tokenizer, seq_len):
    inputs, targets = [], []
    for text in corpus:
        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens) - seq_len):
            inputs.append(tokens[i:i + seq_len])
            targets.append(tokens[i + 1:i + seq_len + 1])
    return np.array(inputs), np.array(targets)

# Cross-entropy loss
def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    log_probs = -np.log(softmax[np.arange(len(targets_flat)), targets_flat] + 1e-10)
    return np.mean(log_probs)

# Initialize model
seq_len = 5
model = NanoGPT(
    vocab_size=vocab_size,
    d_model=64,
    num_heads=4,
    d_ff=256,
    num_layers=2,
    max_len=seq_len,
    
)

# Training
num_epochs = 1000
learning_rate = 0.001
batch_size = 2

inputs, targets = prepare_dataset(corpus, tokenizer, seq_len)
print("Inputs shape:", inputs.shape, "Targets shape:", targets.shape)

for epoch in range(num_epochs):
    indices = np.random.permutation(len(inputs))
    total_loss = 0
    for i in range(0, len(inputs), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]
        
        # Create causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        
        # Forward pass
        logits = model.forward(batch_inputs, mask=mask)
        
        # Compute loss
        loss = cross_entropy_loss(logits, batch_targets)
        total_loss += loss
        
        # Backward pass (simplified)
        model.backward(logits, batch_targets, learning_rate)
        
        print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Loss: {loss:.4f}")
    
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / (len(inputs) // batch_size):.4f}")

# Test input
input_text = "Alice was beginning to"
input_ids = np.array([tokenizer.encode(input_text)])
mask = np.triu(np.ones((input_ids.shape[1], input_ids.shape[1])), k=1)
logits = model.forward(input_ids, mask=mask)
probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
probs = probs / np.sum(probs, axis=-1, keepdims=True)
next_token = np.argmax(probs[0, -1, :])
print("Input text:", input_text)
if 0 <= next_token < len(tokenizer.vocab):
    print("Next token:", list(tokenizer.vocab)[next_token])
else:
    print("Error: next_token index out of bounds", next_token)

# Test individual components
X = np.random.randn(1, len(input_ids[0]), 64) / np.sqrt(64)
attn = MultiHeadAttention(d_model=64, num_heads=4)
pos_enc = PositionalEncoding(d_model=64, max_len=seq_len)
norm = LayerNorm(d_model=64)
ffn = FeedForward(d_model=64, d_ff=256)
block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)

print("Attention output shape:", attn.forward(X).shape)
print("Positional encoding output shape:", pos_enc.forward(X).shape)
print("Layer norm output shape:", norm.forward(X).shape)
print("FFN output shape:", ffn.forward(X).shape)
print("Transformer block output shape:", block.forward(X).shape)