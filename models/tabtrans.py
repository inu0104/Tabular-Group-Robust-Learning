import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Core components ---

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mult * 2)
        self.act = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * mult, dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.out = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.heads, -1).transpose(1, 2) for t in qkv]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, ff_mult=4, attn_dropout=0.0, ff_dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, attn_dropout)
        self.ff = FeedForward(dim, ff_mult, ff_dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_numerical, dim))
        self.bias = nn.Parameter(torch.randn(num_numerical, dim))

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, N, 1)
        return x * self.weight + self.bias  # (B, N, D)

# --- TabTransformer ---

class TabTransformer(nn.Module):
    def __init__(self, categories, num_continuous, dim=32, depth=6, heads=8, dim_head=16,
            num_classes=1, attn_dropout=0.0, ff_dropout=0.0):
        super().__init__()
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.dim = dim

        total_tokens = sum(categories)
        offset = [0] + list(torch.cumsum(torch.tensor(categories[:-1]), dim=0))
        self.register_buffer('category_offsets', torch.tensor(offset))

        self.emb = nn.Embedding(total_tokens, dim)
        if num_continuous > 0:
            self.num_embed = NumericalEmbedder(dim, num_continuous)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes)
        )

    def get_representation(self, x):
        x_cat, x_num = x  # unpack tuple
        B = len(x_cat)
        x_cat = x_cat + self.category_offsets.to(x_cat.device)
        x_cat = self.emb(x_cat)

        tokens = [x_cat]
        if self.num_continuous > 0 and x_num is not None:
            x_num = self.num_embed(x_num)
            tokens.append(x_num)

        x = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)
        return x[:, 0]  # (B, D) — CLS token representation

    def forward(self, x):
        cls_rep = self.get_representation(x)
        return self.head(cls_rep)
