import torch
import torch.nn as nn
import torch.nn.functional as F

def sparsemax(input, dim=-1):
    dim = dim if dim >= 0 else input.ndim + dim  

    input = input - input.max(dim=dim, keepdim=True).values
    sorted_input, _ = torch.sort(input, descending=True, dim=dim)
    cumsum = torch.cumsum(sorted_input, dim=dim)

    shape = [1] * input.ndim
    shape[dim] = -1
    rhos = torch.arange(1, input.shape[dim] + 1, device=input.device).view(shape)

    support = sorted_input * rhos > (cumsum - 1)
    k = support.sum(dim=dim, keepdim=True)
    k_safe = torch.clamp(k, min=1)
    index = (k_safe - 1).expand_as(k_safe)

    tau = (cumsum.gather(dim, index) - 1) / k_safe
    output = torch.clamp(input - tau, min=0)
    return output

class GLU_Block(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.bn = nn.LayerNorm(output_dim * 2)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x_a, x_b = x.chunk(2, dim=-1)
        return self.dropout(x_a * torch.sigmoid(x_b))


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_layers=None, dropout=0.0):
        super().__init__()
        self.shared = shared_layers
        if shared_layers is None:
            self.block1 = GLU_Block(input_dim, hidden_dim, dropout)
        self.block2 = GLU_Block(hidden_dim, hidden_dim, dropout)

    def forward(self, x):
        if self.shared is not None:
            x = self.shared(x)
        else:
            x = self.block1(x)
        return self.block2(x)


class AttentiveTransformer(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, input_dim) 
        self.bn = nn.LayerNorm(input_dim)

    def forward(self, x, prior):
        x = self.fc(x)
        x = self.bn(x)
        x = x * prior
        return sparsemax(x, dim=-1)

class TabNet(nn.Module):
    def __init__(self, input_dim, num_classes, n_steps=3, hidden_dim=64, gamma=1.5, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.num_classes = num_classes
        self.gamma = gamma

        self.shared_ft = GLU_Block(input_dim, hidden_dim, dropout)

        self.initial_transform = FeatureTransformer(input_dim, hidden_dim, self.shared_ft, dropout)

        self.attentive = nn.ModuleList([
            AttentiveTransformer(hidden_dim, input_dim) for _ in range(n_steps)
        ])
        self.transformers = nn.ModuleList([
            FeatureTransformer(input_dim, hidden_dim, self.shared_ft, dropout) for _ in range(n_steps)
        ])

        self.fc = nn.Linear(hidden_dim, num_classes)

    def get_representation(self, x):
        B, D = x.size()
        prior = torch.ones(B, D, device=x.device)
        out_agg = 0
        M_loss = 0
        x_transformed = self.initial_transform(x)

        for step in range(self.n_steps):
            mask = self.attentive[step](x_transformed, prior)
            prior = prior * (self.gamma - mask)
            x_masked = x * mask
            x_transformed = self.transformers[step](x_masked)
            out_agg = out_agg + x_transformed

            M_loss += torch.mean(torch.sum(-mask * torch.log(mask + 1e-15), dim=1))

        return out_agg, M_loss

    def forward(self, x, return_loss=False):
        out_agg, M_loss = self.get_representation(x)
        logits = self.fc(out_agg)
        
        if return_loss:
            return logits, M_loss  
        else:
            return logits    