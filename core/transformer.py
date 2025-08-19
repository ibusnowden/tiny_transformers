# Transformer block
from .attention import MultiHeadAttention
from .norm import LayerNorm
from .ffn import FeedForward

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        
    def forward(self, X, mask=None):
        attn_out = self.attention.forward(X, mask)
        X = self.norm1.forward(X + attn_out) # Residual connection
        ffn_out = self.ffn.forward(X)
        X = self.norm2.forward(X + ffn_out) # Residual connection

        return X