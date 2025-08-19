# Feed-forward network
import numpy as np

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        # X shape: (batch, seq_len, d_model)
        out = np.matmul(X, self.W1) + self.b1
        out = np.maximum(0, out)  # ReLU
        out = np.matmul(out, self.W2) + self.b2 

        return out
