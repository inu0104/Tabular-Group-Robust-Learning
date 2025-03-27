import copy

from models.tabnet import TabNet
from models.autoint import AutoInt
from models.deepfm import DeepFM
from models.node import NODE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split, Subset, DataLoader

def split_train_set(train_loader, alpha=0.1, seed=42):
    dataset = train_loader.dataset
    n_total = len(dataset)
    n_held = int(n_total * alpha)
    n_remain = n_total - n_held

    torch.manual_seed(seed)
    held_indices, remain_indices = random_split(range(n_total), [n_held, n_remain])

    held_dataset = Subset(dataset, held_indices)
    remain_dataset = Subset(dataset, remain_indices)

    held_loader = DataLoader(held_dataset, batch_size=train_loader.batch_size, shuffle=False)
    remain_loader = DataLoader(remain_dataset, batch_size=train_loader.batch_size, shuffle=True)

    return held_loader, remain_loader

def split_validation_set(val_loader, seed=42):
    dataset = val_loader.dataset
    n_total = len(dataset)
    n_half = n_total // 2

    torch.manual_seed(seed)
    part1, part2 = random_split(dataset, [n_half, n_total - n_half])

    target_loader = DataLoader(part1, batch_size=val_loader.batch_size, shuffle=False)
    val_loader_final = DataLoader(part2, batch_size=val_loader.batch_size, shuffle=False)
    return target_loader, val_loader_final

###########################################################################

class TabNetFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.initial_transform = model.initial_transform
        self.attentive = model.attentive
        self.transformers = model.transformers
        self.n_steps = model.n_steps

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

        return out_agg


class AutoIntFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding = model.embedding
        self.attn_layers = model.attn_layers
        self.input_dim = model.input_dim
        self.embed_dim = model.embed_dim

    def forward(self, x):
        indices = torch.arange(self.input_dim).to(x.device)
        embed_weights = self.embedding(indices)  # (input_dim, embed_dim)
        x_embed = x.unsqueeze(2) * embed_weights  # (B, input_dim, embed_dim)

        out = x_embed
        for attn in self.attn_layers:
            out = out + attn(out)  # residual
        return out.view(x.shape[0], -1)  # flattened


class DeepFMFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feature_embedding = model.feature_embedding
        self.input_dim = model.input_dim
        self.embedding_dim = model.embedding_dim
        self.deep = model.deep

    def forward(self, x):
        indices = torch.arange(self.input_dim).to(x.device)
        embeds = self.feature_embedding(indices)  # (input_dim, embed_dim)
        x_embed = x.unsqueeze(2) * embeds  # (batch, input_dim, embed_dim)
        deep_input = x_embed.view(x.shape[0], -1)
        return self.deep(deep_input)


class NODEFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.node_block = model.node_block 

    def forward(self, x):
        return self.node_block(x)


def get_feature_extractor(model):
    
    if isinstance(model, TabNet):
        return TabNetFeatureExtractor(model)

    elif isinstance(model, AutoInt):
        return AutoIntFeatureExtractor(model)

    elif isinstance(model, DeepFM):
        return DeepFMFeatureExtractor(model)

    elif isinstance(model, NODE):
        return NODEFeatureExtractor(model)

    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
###########################################################################

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


def train_last_layer(feature_extractor, classifier, data_loader, w, device, weight_decay=1e-3):
    classifier = classifier.to(device)
    classifier.train()
    feature_extractor.eval()

    optimizer = optim.LBFGS(classifier.parameters(), lr=1.0)
    criterion = nn.NLLLoss(reduction="none")

    all_x, all_y = [], []
    for x, y, _, _ in data_loader:
        all_x.append(x)
        all_y.append(y)
    x_all = torch.cat(all_x).to(device)
    y_all = torch.cat(all_y).to(device)
    w = w.to(device)

    with torch.no_grad():
        feats = feature_extractor(x_all)

    def closure():
        optimizer.zero_grad()
        out = classifier(feats)
        loss_vec = criterion(out, y_all)
        base_loss = torch.sum(w * loss_vec)
        l2_reg = sum(torch.norm(p) ** 2 for p in classifier.parameters())
        loss = base_loss + weight_decay * l2_reg
        loss.backward()
        return loss

    optimizer.step(closure)
    return classifier

###########################################################################

def update_gamma(gamma, classifier, feature_extractor, target_loader, num_groups, tau, device):
    target_group_losses = {}
    with torch.no_grad():
        for x, y, g, _ in target_loader:
            x, y, g = x.to(device), y.to(device), g.to(device)
            out = classifier(feature_extractor(x))
            loss_vec = F.nll_loss(out, y, reduction="none")
            for i in range(len(x)):
                group_id = g[i].item()
                if group_id not in target_group_losses:
                    target_group_losses[group_id] = [loss_vec[i].item()]
                else:
                    target_group_losses[group_id].append(loss_vec[i].item())

    for g in range(num_groups):
        loss = sum(target_group_losses.get(g, [0.0])) / max(len(target_group_losses.get(g, [1])), 1)
        gamma[g] = gamma[g] * torch.exp(torch.tensor(loss / tau, device=gamma.device))

    gamma = gamma / gamma.sum()
    return gamma


def update_gamma_plus(gamma, model, target_loader, num_groups, tau, device):
    model.eval()
    group_loss_sum = torch.zeros(num_groups, device=device)
    group_count = torch.zeros(num_groups, device=device)

    with torch.no_grad():
        for x, y, g, _ in target_loader:
            x, y, g = x.to(device), y.to(device), g.to(device)
            out = model(x)
            loss_vec = F.nll_loss(out, y, reduction="none")
            for i in range(len(x)):
                group_id = g[i]
                group_loss_sum[group_id] += loss_vec[i]
                group_count[group_id] += 1

    for g in range(num_groups):
        avg_loss = group_loss_sum[g] / max(group_count[g], 1.0)
        gamma[g] = gamma[g] * torch.exp(avg_loss / tau)

    gamma = gamma / gamma.sum()
    return gamma

###########################################################################

def compute_aggregated_influence(feature_extractor, classifier, train_loader_held, target_loader, gamma, w, device, method):
    # 1. ξ 초기화 (sample 개수와 같은 길이의 0 vector)
    n = len(train_loader_held.dataset)
    xi = torch.zeros(n, device=device)
    
    target_x, target_y, target_group, target_id = [], [], [], []
    
    for x, y, g, sid in target_loader:
        target_x.append(x)
        target_y.append(y)
        target_group.append(g)
        target_id.append(sid)

    target_x = torch.cat(target_x).to(device)
    target_y = torch.cat(target_y).to(device)
    target_group = torch.cat(target_group).to(device)
    target_id = torch.cat(target_id).to(device)
    
    unique_groups = torch.unique(target_group)
    
    # 2. 각 그룹 g에 대해 반복

    for g in unique_groups:
        g = g.item()
  
        if method == 'gsr-hf':
            influence = compute_group_influence_hf(
                feature_extractor, classifier,
                train_loader_held,
                target_x, target_y,
                g, 
                target_group,
                w, 
                device
            )
        elif method == 'gsr':
        # 2-2. influence 계산 (Vanilla GSR 방식)
            influence = compute_group_influence(
                feature_extractor, classifier,
                train_loader_held,
                target_x, target_y,
                g, 
                target_group,
                w, 
                device
            )
        elif method == 'gsr-plus':
            influence = compute_group_influence_plus(
                classifier,  # classifier == model in gsr_plus
                train_loader_held,
                target_x, target_y,
                g, target_group, w, device
            )
        else:
            raise ValueError(f"Unknown method: {method}")  
        
        #print(g, group_size)
        influence = influence / (influence.abs().max() + 1e-8)
        influence = torch.clamp(influence, min=-1.0, max=1.0)

        
        # 2-3. ξ += gamma[g] * influence
        xi += gamma[g] * influence
        
    return xi


def compute_group_influence_hf(
    feature_extractor, classifier,
    train_loader_held,
    target_x, target_y,
    group_id, 
    target_group,
    w,
    device
):
    
    classifier.eval()
    feature_extractor.eval()

    group_mask = (target_group == group_id)
    group_x = target_x[group_mask]
    group_y = target_y[group_mask]
    
    # 1. held-out 전체 feature와 loss
    all_x, all_y = [], []
    for x, y, _, _ in train_loader_held:
        all_x.append(x)
        all_y.append(y)
    x_held = torch.cat(all_x).to(device)
    y_held = torch.cat(all_y).to(device)
    w = w.to(device)

    feats_held = feature_extractor(x_held)
    loss_vec = F.nll_loss(classifier(feats_held), y_held, reduction="none")

    grads_held = []
    for i in range(len(x_held)):
        grad_i = torch.autograd.grad(loss_vec[i], classifier.parameters(), retain_graph=True)
        grad_i_flat = torch.cat([g.view(-1) for g in grad_i])
        grads_held.append(grad_i_flat)
    grads_held = torch.stack(grads_held)  # (n, d)

    # 2. target 그룹 g에 대한 평균 loss
    feats_target = feature_extractor(group_x)
    target_loss_vec = F.nll_loss(classifier(feats_target), group_y, reduction="none")
    target_loss = target_loss_vec.mean()

    grad_target = torch.autograd.grad(target_loss, classifier.parameters(), create_graph=True)
    grad_target_vec = torch.cat([g.view(-1) for g in grad_target])

    # 3. HVP 근사: identity로 근사 (i.e., influence = -gᵢᵀ · ∇L_target)
    influence_scores = -torch.matmul(grads_held, grad_target_vec.detach())

    return influence_scores.detach()


def compute_group_influence(
    feature_extractor, classifier,
    train_loader_held,
    target_x, target_y,
    group_id, 
    target_group,
    w,
    device
):
    """
    기본 GSR 방식에서 group influence를 계산하는 함수 (Hessian-Free 아님).
    구현 방식은 필요에 따라 대체 가능.
    """
    classifier.eval()
    feature_extractor.eval()

    group_mask = (target_group == group_id)
    group_x = target_x[group_mask]
    group_y = target_y[group_mask]

    all_x, all_y = [], []
    for x, y, _, _ in train_loader_held:
        all_x.append(x)
        all_y.append(y)
    x_held = torch.cat(all_x).to(device)
    y_held = torch.cat(all_y).to(device)
    w = w.to(device)

    feats_held = feature_extractor(x_held)
    loss_vec = F.nll_loss(classifier(feats_held), y_held, reduction="none")

    grads_held = []
    for i in range(len(x_held)):
        grad_i = torch.autograd.grad(loss_vec[i], classifier.parameters(), retain_graph=True)
        grad_i_flat = torch.cat([g.view(-1) for g in grad_i])
        grads_held.append(grad_i_flat)
    grads_held = torch.stack(grads_held)

    feats_target = feature_extractor(group_x)
    target_loss_vec = F.nll_loss(classifier(feats_target), group_y, reduction="none")
    target_loss = target_loss_vec.mean()

    grad_target = torch.autograd.grad(target_loss, classifier.parameters(), create_graph=True)
    grad_target_vec = torch.cat([g.view(-1) for g in grad_target])

    # Identity Hessian 가정
    influence_scores = -torch.matmul(grads_held, grad_target_vec.detach())

    return influence_scores.detach()


def compute_group_influence_plus(
    model,
    train_loader_held,
    target_x, target_y,
    group_id, 
    target_group,
    w,
    device,
    damping=0.1,
    scale=0.01,
    depth=30
):
    model.eval()
    params = list(model.parameters())

    group_mask = (target_group == group_id)
    group_x = target_x[group_mask]
    group_y = target_y[group_mask]

    # held-out 데이터 준비
    x_held, y_held = [], []
    for x, y, _, _ in train_loader_held:
        x_held.append(x)
        y_held.append(y)
    x_held = torch.cat(x_held).to(device)
    y_held = torch.cat(y_held).to(device)

    # gradients per sample (train held-out)
    outputs_held = model(x_held)
    loss_vec_held = F.nll_loss(outputs_held, y_held, reduction='none')
    grads_held = []
    for i in range(len(x_held)):
        grad_i = torch.autograd.grad(loss_vec_held[i], params, retain_graph=True)
        grad_i_flat = torch.cat([g.view(-1) for g in grad_i])
        grads_held.append(grad_i_flat)
    grads_held = torch.stack(grads_held)  # (n, d)

    # target 그룹의 평균 loss의 gradient
    outputs_group = model(group_x)
    loss_group = F.nll_loss(outputs_group, group_y)
    grad_group = torch.autograd.grad(loss_group, params, create_graph=True)
    grad_group_vec = torch.cat([g.view(-1) for g in grad_group])

    # LiSSA 방식으로 Hessian-inverse @ grad 계산
    def hvp(vec):
        out = model(x_held)
        loss = F.nll_loss(out, y_held)
        grad_params = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grad_params])
        grad_dot_vec = torch.dot(flat_grad, vec)
        hv = torch.autograd.grad(grad_dot_vec, params, retain_graph=True)
        hv_flat = torch.cat([g.contiguous().view(-1) for g in hv])
        return hv_flat + damping * vec

    cur_estimate = grad_group_vec.clone()
    for _ in range(depth):
        hvp_est = hvp(cur_estimate)
        cur_estimate = grad_group_vec + (1 - damping) * cur_estimate - scale * hvp_est
        
        cur_estimate = torch.clamp(cur_estimate, min=-10.0, max=10.0)
    ihvp = cur_estimate.detach()

    # influence 계산 (정확히 논문과 일치)
    influence_scores = -torch.matmul(grads_held, ihvp)

    return influence_scores.detach()

###########################################################################

def update_w(w, lr_w, xi):
    w = w - lr_w * xi
    w = torch.clamp(w, min=0)
    w = w / w.sum()
    return w

###########################################################################

def tuple_evaluate(feature_extractor, classifier, dataloader, dataset_name="Val", device="cuda"):
    feature_extractor.to(device).eval()
    classifier.to(device).eval()

    outputs, labels = [], []
    with torch.no_grad():
        for x, y, _, _ in dataloader:
            x = x.to(device)
            out = classifier(feature_extractor(x))
            outputs.append(out.cpu())
            labels.append(y.cpu())

    outputs = torch.cat(outputs)
    labels = torch.cat(labels)

    criterion = nn.NLLLoss(reduction="none")
    losses = criterion(outputs, labels).squeeze()

    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    total_loss = losses.sum().item() / total

    try:
        groups = dataloader.dataset.groups
    except AttributeError:
        if hasattr(dataloader.dataset, 'dataset') and hasattr(dataloader.dataset, 'indices'):
            full_dataset = dataloader.dataset.dataset
            indices = dataloader.dataset.indices
            groups = full_dataset.groups[indices]
        else:
            raise AttributeError("Cannot extract group information from dataloader.dataset")

    group_correct = {}
    group_total = {}
    group_loss_sum = {}

    for i in range(total):
        group = groups[i].item()

        if group not in group_correct:
            group_correct[group] = 0
            group_total[group] = 0
            group_loss_sum[group] = 0.0

        group_total[group] += 1
        group_loss_sum[group] += losses[i].item()
        if predictions[i] == labels[i]:
            group_correct[group] += 1

    group_losses = {}
    group_accuracies = {}

    for group in sorted(group_total.keys()):
        if group_total[group] > 0:
            group_acc = group_correct[group] / group_total[group]
            group_loss_avg = group_loss_sum[group] / group_total[group]
            group_losses[group] = group_loss_avg
            group_accuracies[group] = group_acc
        else:
            group_losses[group] = None
            group_accuracies[group] = None

    return accuracy, total_loss, group_accuracies, group_losses

def tuple_evaluate_plus(model, dataloader, dataset_name="Val", device="cuda"):
    model.to(device).eval()

    outputs, labels = [], []
    with torch.no_grad():
        for x, y, _, _ in dataloader:
            x = x.to(device)
            out = model(x)
            outputs.append(out.cpu())
            labels.append(y.cpu())

    outputs = torch.cat(outputs)
    labels = torch.cat(labels)

    criterion = nn.NLLLoss(reduction="none")
    losses = criterion(outputs, labels).squeeze()

    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    total_loss = losses.sum().item() / total

    try:
        groups = dataloader.dataset.groups
    except AttributeError:
        if hasattr(dataloader.dataset, 'dataset') and hasattr(dataloader.dataset, 'indices'):
            full_dataset = dataloader.dataset.dataset
            indices = dataloader.dataset.indices
            groups = full_dataset.groups[indices]
        else:
            raise AttributeError("Cannot extract group information from dataloader.dataset")

    group_correct = {}
    group_total = {}
    group_loss_sum = {}

    for i in range(total):
        group = groups[i].item()

        if group not in group_correct:
            group_correct[group] = 0
            group_total[group] = 0
            group_loss_sum[group] = 0.0

        group_total[group] += 1
        group_loss_sum[group] += losses[i].item()
        if predictions[i] == labels[i]:
            group_correct[group] += 1

    group_losses = {}
    group_accuracies = {}

    for group in sorted(group_total.keys()):
        if group_total[group] > 0:
            group_acc = group_correct[group] / group_total[group]
            group_loss_avg = group_loss_sum[group] / group_total[group]
            group_losses[group] = group_loss_avg
            group_accuracies[group] = group_acc
        else:
            group_losses[group] = None
            group_accuracies[group] = None

    return accuracy, total_loss, group_accuracies, group_losses

###########################################################################

def log_group_weights(w, train_loader_held, device):
    group_weights = {}
    group_counts = {}
    all_groups = []

    # 각 sample의 group ID 추출
    for _, _, group, _ in train_loader_held:
        all_groups.append(group)
    all_groups = torch.cat(all_groups).to(device)

    for i, group_id in enumerate(all_groups):
        g = group_id.item()
        group_weights[g] = group_weights.get(g, 0.0) + w[i].item()
        group_counts[g] = group_counts.get(g, 0) + 1

    # 그룹별 평균 계산
    group_avgs = {g: group_weights[g] / group_counts[g] for g in group_weights}

    # 출력 없이 반환만
    return {
        "sum": group_weights,
        "avg": group_avgs,
        "count": group_counts
    }




