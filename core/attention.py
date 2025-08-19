# Multi-head self-attention
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize weights for Q, K, V
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model) 

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch, seq_len, d_k)
        scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask * -1e9
        attn = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        return np.matmul(attn, V)
    
    def forward(self, X, mask=None):
        # X shape: (batch, seq_len, d_model)
        batch, seq_len, _ = X.shape

        # Linear projections
        Q = np.matmul(X, self.W_q).reshape(batch, seq_len, self.num_heads, self.d_k)
        K = np.matmul(X, self.W_k).reshape(batch, seq_len, self.num_heads, self.d_k)
        V = np.matmul(X, self.W_v).reshape(batch, seq_len, self.num_heads, self.d_k)

        # Transpose to (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = K.transpose(0, 2, 1, 3)

        # Attention
        out = self.scaled_dot_product_attention(Q, K, V, mask)

        # concatenate heads and project
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        out = np.matmul(out, self.W_o)

        return out
        