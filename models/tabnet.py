import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU_Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        x_proj = self.fc(x)
        x_a, x_b = x_proj.chunk(2, dim=-1)
        return x_a * torch.sigmoid(x_b)


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.block1 = GLU_Block(input_dim, hidden_dim)
        self.block2 = GLU_Block(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, prior):
        x = self.fc(x)
        x = self.bn(x)
        mask = torch.softmax(x * prior, dim=-1)
        return mask


class TabNet(nn.Module):
    def __init__(self, config):
        super(TabNet, self).__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_steps = config.get('n_steps', 3)
        self.num_classes = config['num_classes']
        
        self.initial_transform = FeatureTransformer(self.input_dim, self.hidden_dim)
        self.attentive = nn.ModuleList([
            AttentiveTransformer(self.hidden_dim, self.input_dim) for _ in range(self.n_steps)
        ])
        self.transformers = nn.ModuleList([
            FeatureTransformer(self.input_dim, self.hidden_dim) for _ in range(self.n_steps)
        ])
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):
        out_agg = 0
        prior = torch.ones_like(x)
        x_raw = x.clone()  
        
        x_transformed = self.initial_transform(x_raw)
        
        for step in range(self.n_steps):
            mask = self.attentive[step](x_transformed, prior)  
            x_masked = x_raw * mask 
            step_output = self.transformers[step](x_masked)
            out_agg = out_agg + step_output
            prior = prior * (1 - mask)

        logits = self.fc(out_agg)
        return F.log_softmax(logits, dim=-1)

if __name__ == "__main__":
    model = TabNet(input_dim=14, hidden_dim=64, n_steps=3, num_classes=2)
    sample_input = torch.randn(32, 14)  # Batch 32, Feature 14
    output = model(sample_input)
    print("Output shape:", output.shape)  # Expected: (32, 2)
