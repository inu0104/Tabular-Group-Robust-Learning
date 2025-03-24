import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils.data_loader import GroupDataset

def compute_grad(model, dataloader, device="cuda"):

    model.eval()

    all_gradients = []
    all_sample_ids = []

    for batch_idx, (X_batch, _, _, sample_ids) in enumerate(dataloader):
        X_batch = X_batch.to(device, non_blocking=True)

        for i in range(X_batch.shape[0]):
            X_single = X_batch[i].unsqueeze(0).to(device)
            X_single.requires_grad_()

            output = model(X_single).float()
            grad_outputs = torch.ones_like(output, device=device)

            grads = torch.autograd.grad(outputs=output,
                                        inputs=[param for param in model.parameters() if param.requires_grad],
                                        grad_outputs=grad_outputs,
                                        retain_graph=False)

            grad_vector = torch.cat([g.flatten() for g in grads], dim=0).to("cpu")

            all_gradients.append(grad_vector)
            all_sample_ids.append(sample_ids[i].item())

            del X_single, output, grads, grad_outputs, grad_vector
            torch.cuda.empty_cache()

    all_gradients_tensor = torch.stack(all_gradients)
    all_sample_ids_tensor = torch.tensor(all_sample_ids)

    return all_gradients_tensor, all_sample_ids_tensor

def tau_vector(train_grad, train_sample, model, test_loader, device):
  
    model.eval()
    print()
    
    sample_ids = train_sample.cpu().numpy()
    train_grad = pd.DataFrame(train_grad.cpu().numpy())
    train_grad.insert(0, "sample_id", sample_ids)

    sample_ids = torch.tensor(train_grad["sample_id"].values, dtype=torch.long)
    all_gradients = torch.tensor(train_grad.drop(columns=["sample_id"]).values, dtype=torch.float32)
    num_train_samples, p = all_gradients.shape
    all_gradients = all_gradients.to(device)

    G_sigma = torch.matmul(all_gradients.T, all_gradients)
    try:
        G_sigma_inv = torch.inverse(G_sigma)
    except RuntimeError:
        print("⚠️ G_sigma is singular, using pseudo-inverse instead.")
        G_sigma_inv = torch.linalg.pinv(G_sigma)

    group_logits = {}  # dict[group_id] = [logits]

    with torch.no_grad():
        for X_batch, _, group_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).cpu()
            for logit, g in zip(logits, group_batch):
                g = int(g.item())
                if g not in group_logits:
                    group_logits[g] = []
                group_logits[g].append(logit)

    # (1 - σ(f(z)))
    group_sigmoid_mean = {}
    for g in group_logits:
        logits_tensor = torch.stack(group_logits[g], dim=0)
        sigma = torch.sigmoid(logits_tensor)
        group_sigmoid_mean[g] = (1 - sigma).mean().to(device)

    # τ
    tau_matrix = []
    with torch.no_grad():
        for i in range(num_train_samples):
            g_z_train = all_gradients[i].view(-1, 1)
            tau_per_group = {}

            for g in group_sigmoid_mean:
                influence = torch.matmul(g_z_train.T, torch.matmul(G_sigma_inv, g_z_train)).squeeze()
                tau_value = influence * group_sigmoid_mean[g]
                tau_per_group[g] = tau_value.item()

            tau_matrix.append(tau_per_group)

    tau_vector_df = pd.DataFrame(tau_matrix)
    tau_vector_df.insert(0, "sample_id", sample_ids.cpu().numpy())

    print("✅ τ vector calculation completed successfully (per group)")
    return tau_vector_df


def alignment_score(tau_vector_df, valid_group_losses, beta=1.0):
    df = tau_vector_df.copy()

    group_ids = sorted(valid_group_losses.keys())

    group_losses = np.array([valid_group_losses[g] for g in group_ids], dtype=np.float32)
    weights = np.exp(beta * group_losses)
    weights /= np.sum(weights)  # softmax-style normalize

    tau_matrix = df[group_ids].values
    alignment_scores = np.dot(tau_matrix, weights)

    df["alignment_score"] = alignment_scores
    print("✅ Group Alignment Score computed from τ matrix")

    return df[["sample_id", "alignment_score"]]


def new_per(alignment_scores_df, train_df, dataset, remove_ratio=0.1):
    new_train_df = train_df.copy()

    alignment_scores_df = alignment_scores_df.sort_values(by="alignment_score", ascending=True)

    num_remove = int(len(alignment_scores_df) * remove_ratio)
    remove_sample_ids = alignment_scores_df.iloc[:num_remove]["sample_id"].values

    new_train_df = new_train_df[~new_train_df["sample_id"].isin(remove_sample_ids)].reset_index(drop=True)

    path = f"./dataset/{dataset}/new_train_processed.csv"
    new_train_df.to_csv(path, index=False)
    
    train_dataset_new = GroupDataset(path)
    train_loader_new = DataLoader(train_dataset_new, batch_size=32, shuffle=True)

    print(f"✅ Removed bottom {remove_ratio*100:.1f}% alignment samples - Remaining samples: {len(train_dataset_new)}")

    return train_loader_new


def new_neg(alignment_scores_df, train_df, dataset):
    new_train_df = train_df.copy()

    # sample_id: A_i < 0
    remove_sample_ids = alignment_scores_df[alignment_scores_df["alignment_score"] < 0]["sample_id"].values

    new_train_df = new_train_df[~new_train_df["sample_id"].isin(remove_sample_ids)].reset_index(drop=True)
    
    path = f"./dataset/{dataset}/new_train_processed.csv"
    new_train_df.to_csv(path, index=False)

    train_dataset_new = GroupDataset(path)
    train_loader_new = DataLoader(train_dataset_new, batch_size=32, shuffle=True)

    print(f"✅ Removed {len(remove_sample_ids)} samples with Aᵢ < 0 - Remaining samples: {len(train_dataset_new)}")

    return train_loader_new