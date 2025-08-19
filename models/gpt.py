# models/gpt.py
import numpy as np
from core.positional import PositionalEncoding
from core.transformer import TransformerBlock
from core.norm import LayerNorm

class NanoGPT:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512):
        scale = np.sqrt(vocab_size) * 0.1
        self.embedding = np.random.randn(vocab_size, d_model) / scale
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.norm = LayerNorm(d_model)
        self.output = np.random.randn(d_model, vocab_size) / scale
        # Store gradients
        self.grad_embedding = np.zeros_like(self.embedding)
        self.grad_output = np.zeros_like(self.output)

    def forward(self, X, mask=None):
        self.X = X  # Save input for backward pass
        X = self.embedding[X]
        X = np.clip(X, -10.0, 10.0)
        X = X / (np.sqrt(np.mean(X**2, axis=-1, keepdims=True) + 1e-6) * 10.0)
        self.X_embed = X  # Save for backward
        X = self.pos_encoding.forward(X)
        for layer in self.layers:
            X = layer.forward(X, mask)
        X = self.norm.forward(X)
        self.X_final = X  # Save for backward
        logits = np.matmul(X, self.output)
        return logits

    def backward(self, logits, targets, learning_rate=0.01):
        # Simplified backward pass for embedding and output layers
        batch, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Softmax and cross-entropy gradient
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        grad_logits = probs.copy()
        grad_logits[np.arange(grad_logits.shape[0]), targets_flat] -= 1
        grad_logits /= batch * seq_len  # Average over batch and seq_len
        
        # Gradient for output layer
        grad_logits = grad_logits.reshape(batch, seq_len, vocab_size)
        self.grad_output = np.matmul(self.X_final.transpose(0, 2, 1), grad_logits).sum(axis=0)
        grad_X_final = np.matmul(grad_logits, self.output.T)
        
        # Gradient for embedding (simplified, assumes no backprop through layers)
        grad_X_embed = grad_X_final  # Approximate (skipping transformer layers)
        grad_indices = self.X.reshape(-1)
        self.grad_embedding *= 0  # Reset gradients
        for i, idx in enumerate(grad_indices):
            self.grad_embedding[idx] += grad_X_embed[i // seq_len, i % seq_len]
        
        # Update weights
        self.output -= learning_rate * self.grad_output
        self.embedding -= learning_rate * self.grad_embedding