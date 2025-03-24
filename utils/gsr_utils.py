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

    held_loader = DataLoader(held_dataset, batch_size=train_loader.batch_size, shuffle=True)
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

def compute_influence_scores(feature_extractor, classifier, train_loader, target_loader, device):
    classifier.eval()
    feature_extractor.eval()

    # 1. train features
    all_train_x, all_train_y = [], []
    for x, y, _, _ in train_loader:
        all_train_x.append(x)
        all_train_y.append(y)
    train_x = torch.cat(all_train_x).to(device)
    train_y = torch.cat(all_train_y).to(device)

    feats_train = feature_extractor(train_x)
    losses_train = nn.NLLLoss(reduction="none")(classifier(feats_train), train_y)
    
    grads_train = []
    for i in range(len(train_x)):
        grad_i = torch.autograd.grad(
            losses_train[i], classifier.parameters(), retain_graph=True
        )
        grad_i_flat = torch.cat([g.view(-1) for g in grad_i])  # flatten
        grads_train.append(grad_i_flat)
    grads_train = torch.stack(grads_train)  # (n_train, d)

    # 2. target set gradient (worst-group loss avg)
    all_target_x, all_target_y = [], []
    for x, y, _, _ in target_loader:
        all_target_x.append(x)
        all_target_y.append(y)
    target_x = torch.cat(all_target_x).to(device)
    target_y = torch.cat(all_target_y).to(device)

    feats_target = feature_extractor(target_x)
    loss_target = nn.NLLLoss()(classifier(feats_target), target_y)
    
    grad_target = torch.autograd.grad(loss_target, classifier.parameters())
    grad_target = torch.cat([g.view(-1) for g in grad_target])

    # 3. influence score = - <grad_target, grad_i>
    influence_scores = -torch.matmul(grads_train, grad_target.detach())
    return influence_scores.detach()

###########################################################################

def update_sample_weights(w, influence_scores, lr=0.1):
    w = w - lr * influence_scores
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