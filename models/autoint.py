import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (B, F, D)
        B, F, D = x.shape
        H = self.num_heads

        Q = self.query(x).view(B, F, H, -1).transpose(1, 2)  # (B, H, F, d_k)
        K = self.key(x).view(B, F, H, -1).transpose(1, 2)
        V = self.value(x).view(B, F, H, -1).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, F, F)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, H, F, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, F, D)  # (B, F, D)

        return self.fc(attn_output)


class AutoInt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['input_dim']
        self.embed_dim = config.get('embedding_dim', 8)
        self.num_heads = config.get('num_heads', 2)
        self.num_layers = config.get('num_layers', 2)
        self.num_classes = config['num_classes']

        # feature embedding
        self.embedding = nn.Embedding(self.input_dim, self.embed_dim)

        # Self-attention layers
        self.attn_layers = nn.ModuleList([
            MultiHeadSelfAttention(self.embed_dim, self.num_heads)
            for _ in range(self.num_layers)
        ])

        # Output layer
        self.fc = nn.Linear(self.input_dim * self.embed_dim, self.num_classes)

    def forward(self, x):
        # x: (B, input_dim)
        indices = torch.arange(self.input_dim).to(x.device)
        embed_weights = self.embedding(indices)  # (input_dim, embed_dim)
        x_embed = x.unsqueeze(2) * embed_weights  # (B, input_dim, embed_dim)

        out = x_embed
        for attn in self.attn_layers:
            out = out + attn(out)  # Residual connection

        out_flat = out.view(x.shape[0], -1)  # Flatten
        logits = self.fc(out_flat)
        return F.log_softmax(logits, dim=-1)
