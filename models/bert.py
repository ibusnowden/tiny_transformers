# Encoder-only model
import os, sys
sys.path.append('..')
#print(os.getcwd())
#print(sys.path)

import numpy as np
from core.positional import PositionalEncoding
from core.transformer import TransformerBlock
from core.norm import LayerNorm

class NanoBERT:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512):
        self.embedding = np.random.randn(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.norm = LayerNorm(d_model)
        self.cls_head = np.random.randn(d_model, 2)  # Binary classification (e.g., sentiment)
    
    def forward(self, X):
        X = self.embedding[X]
        X = self.pos_encoding.forward(X)
        for layer in self.layers:
            X = layer.forward(X)  # No mask for bidirectional attention
        X = self.norm.forward(X)
        cls_output = X[:, 0, :]  # Use [CLS] token (first token)
        logits = np.matmul(cls_output, self.cls_head)
        return logits