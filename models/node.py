import torch
import torch.nn as nn
import torch.nn.functional as F

class ObliviousTree(nn.Module):
    def __init__(self, input_dim, depth):
        super().__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.num_leaf = 2 ** depth

        self.feature_selector = nn.Linear(input_dim, 1)
        self.thresholds = nn.Parameter(torch.randn(depth))  # (depth,)
        self.leaf_scores = nn.Parameter(torch.randn(self.num_leaf))  # (2^depth,)

        # Binary routing table: (num_leaf, depth)
        # e.g., [[0,0,0], [0,0,1], ..., [1,1,1]]
        binary_codes = torch.arange(self.num_leaf)
        binary_matrix = ((binary_codes[:, None] >> torch.arange(depth-1, -1, -1)) & 1).float()
        self.register_buffer('binary_matrix', binary_matrix)  # non-trainable tensor

    def forward(self, x):  # x: (B, input_dim)
        B = x.size(0)

        feature = self.feature_selector(x)  # (B, 1)
        feature = feature.repeat(1, self.depth)  # (B, depth)

        thresholds = self.thresholds.unsqueeze(0)  # (1, depth)
        decisions = torch.sigmoid(feature - thresholds)  # (B, depth)

        # Soft routing: (B, num_leaf)
        # binary_matrix: (num_leaf, depth) —> (1, num_leaf, depth)
        # decisions: (B, 1, depth)
        bm = self.binary_matrix.unsqueeze(0)           # (1, num_leaf, depth)
        dec = decisions.unsqueeze(1)                   # (B, 1, depth)
        path_probs = dec * bm + (1 - dec) * (1 - bm)   # (B, num_leaf, depth)
        path_probs = path_probs.prod(dim=-1)           # (B, num_leaf)

        # Weighted leaf output
        output = torch.matmul(path_probs, self.leaf_scores)  # (B,)
        return output.unsqueeze(1)  # (B, 1)


class NODEBlock(nn.Module):
    def __init__(self, input_dim, num_trees, depth):
        super().__init__()
        self.trees = nn.ModuleList([
            ObliviousTree(input_dim, depth) for _ in range(num_trees)
        ])
        self.output_dim = num_trees

    def forward(self, x):  # (B, input_dim)
        out = [tree(x) for tree in self.trees]  # list of (B, 1)
        return torch.cat(out, dim=1)  # (B, num_trees)


class NODE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_dim = kwargs["input_dim"]
        num_classes = kwargs["num_classes"]
        num_trees = kwargs.get("num_trees", 5)
        depth = kwargs.get("depth", 3)
        hidden_dim = kwargs.get("hidden_dim", 64)
        dropout = kwargs.get("dropout", 0.0)
        self.node_block = NODEBlock(input_dim, num_trees, depth)
        self.fc = nn.Sequential(
            nn.Linear(self.node_block.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def get_representation(self, x):
        x = self.node_block(x)  # (B, num_trees)
        x = self.fc[0](x)       # Linear
        x = self.fc[1](x)       # ReLU
        return x

    def forward(self, x):
        x = self.get_representation(x)  # (B, num_trees)
        return self.fc[3](x)
