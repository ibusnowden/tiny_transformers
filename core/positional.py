# core/positional.py
import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_len, eps=1e-6):
        self.pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)

    def forward(self, X):
        return X + self.pe[:X.shape[1], :]