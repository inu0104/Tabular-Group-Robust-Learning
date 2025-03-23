import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['input_dim']
        embedding_dim = config.get('embedding_dim', 8)
        hidden_dims = config.get('hidden_dims', [64, 32])
        num_classes = config.get('num_classes', 2)

        # FM 1st part(bias + linear)
        self.linear = nn.Linear(input_dim, 1)

        # FM 2nd part (factorized interaction)
        self.feature_embedding = nn.Embedding(input_dim, embedding_dim)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Deep part (MLP)
        deep_input_dim = input_dim * embedding_dim
        layers = []
        in_dim = deep_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.deep = nn.Sequential(*layers)

        self.output = nn.Linear(1 + 1 + hidden_dims[-1], num_classes)

    def forward(self, x):
        # x: (batch_size, input_dim)

        # FM first
        linear_part = self.linear(x)  # (batch_size, 1)

        # FM second (embedding 기반)
        # setting index to 0~input_dim-1 for use embd
        indices = torch.arange(self.input_dim).to(x.device)
        embeds = self.feature_embedding(indices)  # (input_dim, embed_dim)
        x_embed = x.unsqueeze(2) * embeds  # (batch_size, input_dim, embed_dim)

        sum_square = torch.sum(x_embed, dim=1) ** 2  # (batch_size, embed_dim)
        square_sum = torch.sum(x_embed ** 2, dim=1)  # (batch_size, embed_dim)
        second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # (batch_size, 1)

        # Deep part
        deep_input = x_embed.view(x.shape[0], -1)  # (batch_size, input_dim * embed_dim)
        deep_output = self.deep(deep_input)  # (batch_size, hidden)

        combined = torch.cat([linear_part, second_order, deep_output], dim=1)  # (batch_size, total_dim)
        logits = self.output(combined)  # (batch_size, num_classes)
        return F.log_softmax(logits, dim=-1)
