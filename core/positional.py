# Positional encodings
import numpy as np 

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, -1)
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div) # even
        pe[:, 1::2] = np.cos(position * div) # odd 

    def forward(self, X):
        # X shape: (batch, seq_len, d_model)
        return X + self.pe[:, :X.shape[1], :]