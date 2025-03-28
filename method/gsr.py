import copy
import torch

from utils.evaluation import evaluate
from utils.train import train_or_eval_model
from utils.gsr_utils import (
    get_feature_extractor,
    split_train_set,
    split_validation_set,
    train_last_layer,
    Classifier,
    tuple_evaluate,
    update_gamma,
    compute_aggregated_influence,
    update_w,
    log_group_weights
)
import matplotlib.pyplot as plt
import os
import pandas as pd

def run_gsr(model, train_loader, val_loader, test_loader, train_df, train_params, device, dataset, method):

    print(f"🔥 Running GSR Method on {device}...")
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
    
    # ϕ* 
    feature_extractor = get_feature_extractor(model_rep).to(device)
    feature_extractor.eval()
    
    sample_input, *_ = next(iter(train_loader_held))
    with torch.no_grad():
        feat_sample = feature_extractor(sample_input.to(device))
    feature_dim = feat_sample.shape[1]
    num_classes = train_params.get("num_classes", model.fc.out_features if hasattr(model, "fc") else 2)
    
    # ψ*
    fc_layer = model_rep.fc  
    best_classifier = Classifier(input_dim=feature_dim, num_classes=num_classes)
    best_classifier.linear.load_state_dict(fc_layer.state_dict())
    best_classifier = best_classifier.to(device)
    
    print()
    print("2️⃣  Stage 2: Sample Weighting + Last-Layer Retraining...")
    
    n = len(train_loader_held.dataset)
    w = torch.ones(n).to(device) / n
    
    group_labels = train_loader.dataset.groups
    num_groups = torch.unique(group_labels).numel()
    gamma = torch.ones(num_groups).to(device) / num_groups
    
    tau = train_params.get("tau", 5.0)
    lr_w = train_params.get("lr_w", 0.001)
    outer_steps = train_params.get("outer_steps", 30)
    weight_decay = train_params.get("weight_decay", 1e-3)
    
    best_val_worst_group_loss = float("inf")
    
    a = []
    for step in range(outer_steps):
        
        logs = log_group_weights(w, train_loader_held, device)
        a.append(logs['sum'])
        
        classifier = Classifier(input_dim=feature_dim, num_classes=num_classes)
        classifier = train_last_layer(feature_extractor, classifier, train_loader_held, w, device, weight_decay)
        
        gamma = update_gamma(gamma, classifier, feature_extractor, target_loader, num_groups, tau, device)
        xi = compute_aggregated_influence(feature_extractor, classifier, train_loader_held, target_loader, gamma, w, device, method)
        xi = xi / (xi.abs().max() + 1e-8)
        
        w = update_w(w, lr_w, xi)
        
        
        
        group_weights_sum = logs['sum']
        sorted_group_ids = sorted(group_weights_sum.keys())  # 그룹 순서대로 정렬

        print(f"Step {step:3d} - Group Weight Sums: ", end="")
        for g in sorted_group_ids:
            print(f"[{g}]:{group_weights_sum[g]:.3f}  ", end="")
        print()
            
        val_acc, val_loss, val_group_accs, val_group_losses = tuple_evaluate(
            feature_extractor, classifier, val_loader_final,
            dataset_name="Val", device=device
        )
        
        val_worst_group_loss = max(val_group_losses.values())
        
        if val_worst_group_loss < best_val_worst_group_loss:
            print("!!!")
            best_val_worst_group_loss = val_worst_group_loss
            best_classifier = copy.deepcopy(classifier)
    
    save_group_weight_log(a)
    
    # best_classifier를 붙여 최종 모델 생성
    model_final = copy.deepcopy(model_rep)
    model_final.fc.load_state_dict(best_classifier.linear.state_dict())

    # 최종 평가
    evaluate(model_final, test_loader, dataset_name="TEST (GSR)", device=device)
    return


def save_group_weight_log(group_weight_log, output_dir="./result", filename="group_weights.csv"):
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

    # CSV 저장
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path)

    # 그래프 저장
    plt.figure(figsize=(10, 6))
    for group_id in df.columns:
        plt.plot(df.index, df[group_id], label=f"Group {group_id}")
    plt.xlabel("Step")
    plt.ylabel("Group Weight Sum")
    plt.title("Group-wise Sample Weight Over Iterations")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, "group_weight_trend.png")
    plt.savefig(plot_path)
    plt.close()

    return csv_path, plot_path
