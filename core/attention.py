# core/attention.py
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Tighter initialization
        scale = np.sqrt(d_model) * 0.1  # Reduce scale
        self.W_q = np.random.randn(d_model, d_model) / scale
        self.W_k = np.random.randn(d_model, d_model) / scale
        self.W_v = np.random.randn(d_model, d_model) / scale
        self.W_o = np.random.randn(d_model, d_model) / scale

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            mask = mask[np.newaxis, np.newaxis, :, :]
            scores = scores + mask * -1e9
        attention = softmax(scores, axis=-1)
        context = np.matmul(attention, V)
        return context

    def forward(self, X, mask=None):
        batch, seq_len, d_model = X.shape
        Q = np.matmul(X, self.W_q).reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.matmul(X, self.W_k).reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.matmul(X, self.W_v).reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        context = self.scaled_dot_product_attention(Q, K, V, mask)
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        out = np.matmul(context, self.W_o)
        return out