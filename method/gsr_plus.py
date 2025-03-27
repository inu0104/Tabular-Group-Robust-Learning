import copy
import torch
import random

from utils.evaluation import evaluate
from utils.train import train_or_eval_model
from utils.gsr_utils import (
    split_train_set,
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

    train_loader_held, train_loader_remain = split_train_set(train_loader, alpha=0.1)
    target_loader, val_loader_final = split_validation_set(val_loader)
    
    model_rep = copy.deepcopy(model)
    model_rep = train_or_eval_model(
        model=model_rep,
        data_loader=train_loader_remain,
        params=train_params, 
        device=device,
        mode="train"
    )
    
    evaluate(model_rep, test_loader, dataset_name="TEST (ERM)", device=device)
    
    print()
    print("2️⃣  Stage 2: Sample Weighting + Full Model Retraining...")

    n = len(train_loader_held.dataset)
    w = torch.ones(n).to(device) / n

    group_labels = train_loader.dataset.groups
    num_groups = torch.unique(group_labels).numel()
    gamma = torch.ones(num_groups).to(device) / num_groups

    tau = train_params.get("tau", 3.0)
    lr_w = train_params.get("lr_w", 0.05)
    outer_steps = train_params.get("outer_steps", 20)
    weight_decay = train_params.get("weight_decay", 1e-3)


    val_acc, val_loss, val_group_accs, val_group_losses = tuple_evaluate_plus(
            model_rep, val_loader_final,
            dataset_name="Val", device=device
        )

    best_val_worst_group_loss = max(val_group_losses.values())
    a = []

    for step in range(outer_steps):
        logs = log_group_weights(w, train_loader_held, device)
        a.append(logs['sum'])

        random_seed = random.randint(0, 1_000_000)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        # 전체 모델을 복사해서 새로 학습
        model_tmp = copy.deepcopy(model)
        model_tmp = train_or_eval_model(
            model=model_tmp,
            data_loader=train_loader_held,
            params={**train_params, "sample_weights": w},
            device=device,
            mode="train"
        )

        gamma = update_gamma_plus(gamma, model_tmp, target_loader, num_groups, tau, device)
        xi = compute_aggregated_influence(model_tmp, model_tmp, train_loader_held, target_loader, gamma, w, device, method)
        xi = xi / (xi.abs().max() + 1e-8)
        w = update_w(w, lr_w, xi)

        group_weights_sum = logs['sum']
        sorted_group_ids = sorted(group_weights_sum.keys())
        print(f"Step {step:3d} - Group Weight Sums: ", end="")
        for g in sorted_group_ids:
            print(f"[{g}]:{group_weights_sum[g]:.3f}  ", end="")
        print()

        val_acc, val_loss, val_group_accs, val_group_losses = tuple_evaluate_plus(
            model_tmp, val_loader_final,
            dataset_name="Val", device=device
        )

        val_worst_group_loss = max(val_group_losses.values())
        
        print(best_val_worst_group_loss)
        print(val_worst_group_loss)
        
        if val_worst_group_loss < best_val_worst_group_loss:
            best_val_worst_group_loss = val_worst_group_loss
            best_model_final = copy.deepcopy(model_tmp)

    save_group_weight_log(a)

    print()
    evaluate(best_model_final, test_loader, dataset_name="TEST (GSR+)", device=device)
    return



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
