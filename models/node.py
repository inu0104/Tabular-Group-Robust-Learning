import torch
import torch.nn as nn
import torch.nn.functional as F

class NODEBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_trees, depth=2):
        super(NODEBlock, self).__init__()
        self.num_trees = num_trees
        self.depth = depth
        self.hidden_dim = hidden_dim

        # 결정 함수 (Oblivious Decision Tree)
        self.feature_selection = nn.Linear(input_dim, num_trees)
        self.thresholds = nn.Parameter(torch.randn(num_trees, depth))

        # 가중치 행렬 수정 (리프 수에 맞춰 크기 조정)
        self.num_leaves = num_trees * (2 ** depth)  # 리프 수 계산
        self.leaf_weights = nn.Linear(self.num_leaves, hidden_dim)

    def forward(self, x):
        feature_scores = self.feature_selection(x)  # (batch_size, num_trees)
        feature_decisions = torch.sigmoid(feature_scores.unsqueeze(-1) - self.thresholds)  # (batch_size, num_trees, depth)

        # 트리 리프 생성 (각 트리의 가능한 리프 수 반영)
        leaves = feature_decisions.prod(dim=-1)  # (batch_size, num_trees)
        leaves = torch.cat([leaves, 1 - leaves], dim=-1)  # (batch_size, num_trees * 2)

        # 리프 개수를 num_trees * 2에서 num_trees * (2 ** depth)로 변경
        leaves = leaves.repeat(1, 2 ** (self.depth - 1))  # (batch_size, num_leaves)

        # 리프 가중치 적용
        out = self.leaf_weights(leaves)  # (batch_size, hidden_dim)
        return out


class NODE(nn.Module):
    def __init__(self, config):
        super(NODE, self).__init__()
        self.node_block = NODEBlock(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_trees=config['num_trees'],
            depth=config.get('depth', 2)
        )
        self.fc = nn.Linear(config['hidden_dim'], config['num_classes'])

    def forward(self, x):
        x = self.node_block(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

if __name__ == "__main__":
    # 모델 테스트
    model = NODE(input_dim=14, hidden_dim=64, num_trees=5, depth=3, num_classes=2)
    sample_input = torch.randn(32, 14)  # Batch 32, Feature 14
    output = model(sample_input)
    print("Output shape:", output.shape)  # Expected: (32, 2)
