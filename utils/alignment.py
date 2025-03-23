import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils.data_loader import GroupDataset

def compute_group_alignment_score(tau_vector_df, train_loader, valid_group_losses, beta=1.0):

    sample_ids = []  
    group_labels = []

    for _, _, group_id, indices in train_loader:  
        sample_ids.append(indices.cpu().numpy())
        group_labels.append(group_id.cpu().numpy())

    sample_ids = np.concatenate(sample_ids)
    group_labels = np.concatenate(group_labels)

    group_df = pd.DataFrame({"sample_id": sample_ids, "group": group_labels})
    df = pd.merge(tau_vector_df, group_df, on="sample_id", how="inner")

    # group loss
    unique_groups = set(df["group"].unique())
    group_losses = {g: valid_group_losses.get(g, 0.0) for g in unique_groups}

    # exp(β * group_loss) 
    df["group_loss"] = df["group"].map(group_losses)
    df["exp_beta_loss"] = torch.exp(beta * torch.tensor(df["group_loss"].values, dtype=torch.float32)).numpy()

    # Ai 
    num = df["exp_beta_loss"] * df["tau"]
    denom = df.groupby("group")["exp_beta_loss"].transform("sum")

    df["alignment_score"] = num / denom

    print("✅ Group Alignment Score")
    return df[["sample_id", "group", "alignment_score"]]


def create_new_train_dataset_groupwise(alignment_scores_df, train_df, dataset, remove_ratio=0.1):

    new_train_df = train_df.copy() 

    group_sizes = alignment_scores_df["group"].value_counts().to_dict()
    num_remove_per_group = {g: int(group_sizes[g] * remove_ratio) for g in group_sizes}  

    remove_sample_ids = []

    for group_id in num_remove_per_group.keys():
        group_samples = alignment_scores_df[alignment_scores_df["group"] == group_id]
        group_samples = group_samples.sort_values(by="alignment_score", ascending=True)  

        num_remove = num_remove_per_group[group_id]
        remove_sample_ids.extend(group_samples.iloc[:num_remove]["sample_id"].values) 

    new_train_df = new_train_df[~new_train_df["sample_id"].isin(remove_sample_ids)].reset_index(drop=True)

    path = "./dataset/"+dataset+"/new_train_processed.csv"
    new_train_df.to_csv(path, index=False)
    
    train_dataset_new = GroupDataset(path) 
    train_loader_new = DataLoader(train_dataset_new, batch_size=32, shuffle=True)

    print(f"✅ Successfully removed {remove_ratio*100:.1f}% of samples per group - Remaining samples: {len(train_dataset_new)}")

    return train_loader_new