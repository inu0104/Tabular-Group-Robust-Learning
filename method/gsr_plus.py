import copy
import torch
import random

from utils.evaluation import evaluate
from utils.train import train_or_eval_model
from utils.gsr_utils import (
    split_validation_set,
    tuple_evaluate_plus,
    update_gamma_plus,
    compute_aggregated_influence,
    update_w,
    log_group_weights
)
import matplotlib.pyplot as plt
import os
import pandas as pd


def run_gsr_plus(model, train_loader, val_loader, test_loader, train_df, train_params, device, dataset, method):
    print(f"🔥 Running GSR+ Method on {device}...")
    print("1️⃣  Stage 1: Representation Learning...")

    target_loader, val_loader_final = split_validation_set(val_loader.dataset, val_loader.batch_size)

    w = initialize_weight_with_group_scaling2(train_loader, device)
    
    model_rep = copy.deepcopy(model)
    model_rep = train_or_eval_model(
        model=model_rep,
        train_loader=train_loader,
        valid_loader=val_loader_final,
        params=train_params, 
        device=device,
        mode="train",
        sample_weights=w
    )
    
    #evaluate(model_rep, val_loader_final, dataset_name="Valid (ERM)", device=device)
    #evaluate(model_rep, test_loader, dataset_name="TEST (ERM)", device=device)

    print()
    print("2️⃣  Stage 2: Sample Weighting + Full Model Retraining...")

    group_labels = train_loader.dataset.groups
    num_groups = torch.unique(group_labels).numel()
    gamma = torch.ones(num_groups).to(device) / num_groups

    tau = train_params.get("tau", 1.0)
    lr_w = train_params.get("lr_w", 0.01)
    outer_steps = train_params.get("outer_steps", 5)

    _, _, _, val_group_losses = tuple_evaluate_plus(model_rep, val_loader_final, dataset_name="Val", device=device)
    best_val_worst_group_loss = max(val_group_losses.values())
    best_model_final = copy.deepcopy(model_rep)
    
    log_list = []
    for step in range(outer_steps):
        
        logs = log_group_weights(w, train_loader, device)
        log_list.append(logs['sum'])
        
        group_weights_sum = logs['sum']
        sorted_group_ids = sorted(group_weights_sum.keys())
        print(f"Step {step:3d} - Group Weight Sums: ", end="")
        for g in sorted_group_ids:
            print(f"[{g}]:{group_weights_sum[g]:.3f}  ", end="")
        print()
  
        model_tmp = copy.deepcopy(model)
        model_tmp = train_or_eval_model(
            model=model_tmp,
            train_loader=train_loader,
            valid_loader=val_loader_final,
            sample_weights=w,
            params=train_params,
            device=device,
            mode="train"
        )

        gamma = update_gamma_plus(gamma, model_tmp, target_loader, num_groups, tau, device)
        xi = compute_aggregated_influence(model_tmp, model_tmp, train_loader, target_loader, gamma, device, method)
        w = update_w(w, lr_w, xi)

        val_acc, val_loss, val_group_accs, val_group_losses = tuple_evaluate_plus(
            model_tmp, val_loader_final,
            dataset_name="Val", device=device
        )

        val_worst_group_loss = max(val_group_losses.values())
    
        if val_worst_group_loss < best_val_worst_group_loss:
            print(step)
            best_val_worst_group_loss = val_worst_group_loss
            best_model_final = copy.deepcopy(model_tmp)

    #save_group_weight_log(log_list)

    print()
    #evaluate(best_model_final, val_loader_final, dataset_name="Valid (GSR+)", device=device)
    #evaluate(best_model_final, test_loader, dataset_name="TEST (GSR+)", device=device)
    return model_rep, best_model_final



def save_group_weight_log(group_weight_log, output_dir="./result", filename="group_weights.png"):
    """
    그룹별 sample weight 로그를 저장하고, 변화 추이를 그래프로 그립니다.

    Args:
        group_weight_log (list of dict): 각 step마다 {group_id: weight_sum} 형태의 dict
        output_dir (str): 저장할 폴더
        filename (str): 저장할 CSV 파일 이름

    Returns:
        tuple: (csv_path, plot_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 리스트 → DataFrame (step × group_id)
    df = pd.DataFrame(group_weight_log)
    df.index.name = "step"

    # 그래프 저장
    plt.figure(figsize=(10, 6))
    for group_id in df.columns:
        plt.plot(df.index, df[group_id], label=f"Group {group_id}")
    plt.xlabel("Step")
    plt.ylabel("Group Weight Sum")
    plt.title("Group-wise Sample Weight Over Iterations")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def initialize_weight_with_group_scaling(train_loader, device):
    dataset = train_loader.dataset
    groups = dataset.groups.to(device)             # shape: (n_total,)
    sample_ids = dataset.sample_ids.to(device)     # shape: (n_total,)
    n_total = len(groups)

    group_counts = torch.bincount(groups)
    group_inv_freq = 1.0 / group_counts.float()
    group_weights = group_inv_freq[groups]         # 각 sample의 group에 대한 weight

    rand_noise = torch.rand(n_total, device=device)
    w = rand_noise * group_weights
    w = w / w.sum()

    # sample_id와 weight를 묶어서 반환
    w_with_id = torch.stack([sample_ids.float(), w], dim=1)  # shape: (n_total, 2)

    return w_with_id

def initialize_weight_with_group_scaling2(train_loader, device):
    dataset = train_loader.dataset
    sample_ids = dataset.sample_ids.to(device)
    n_total = len(sample_ids)

    uniform_weight = torch.full((n_total,), 1.0 / n_total, device=device)
    w_with_id = torch.stack([sample_ids.float(), uniform_weight], dim=1)

    return w_with_id