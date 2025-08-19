# Decoder-only model
import os, sys
sys.path.append('..')
#print(os.getcwd())
#print(sys.path)

import numpy as np
from core.positional import PositionalEncoding
from core.transformer import TransformerBlock
from core.norm import LayerNorm

class NanoGPT:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512):
        self.embedding = np.random.randn(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.norm = LayerNorm(d_model)
        self.output = np.random.randn(d_model, vocab_size)

    def forward(self, X, mask=None):
        # X shape: (batch, seq_len) (token indices)
        X = self.embedding[X] #(batch, seq_len, d_model)
        X = self.pos_encoding.forward(X)
        for layer in self.layers:
            X = layer.forward(X, mask)
        X = self.norm.forward(X)
        logits = np.matmu(X, self.output)

        return logits

    

        