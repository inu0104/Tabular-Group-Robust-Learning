import copy
import torch

from utils.evaluation import evaluate
from utils.train import train_or_eval_model
from utils.gsr_utils import (
    get_feature_extractor,
    split_train_set,
    split_validation_set,
    train_last_layer,
    compute_influence_scores, compute_influence_scores_lissa,
    update_sample_weights,
    Classifier,
    tuple_evaluate
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
    
    print()
    print("2️⃣  Stage 2: Sample Weighting + Last-Layer Retraining...")
    n = len(train_loader_held.dataset)
    w = torch.ones(n).to(device) / n
    
    outer_steps = train_params.get("outer_steps", 50)
    lr_w = train_params.get("lr_w", 0.1)
    weight_decay = train_params.get("weight_decay", 1e-3)
    
    sample_input, *_ = next(iter(train_loader_held))
    with torch.no_grad():
        feat_sample = feature_extractor(sample_input.to(device))
    feature_dim = feat_sample.shape[1]
    num_classes = train_params.get("num_classes", model.fc.out_features if hasattr(model, "fc") else 2)

    best_val_score = None
    best_classifier = None
    
    

    group_weight_log1 = []
    group_weight_log2 = []
    for step in range(outer_steps):

        # 1. classifier ψ 
        classifier = Classifier(input_dim=feature_dim, num_classes=num_classes)
        classifier = train_last_layer(feature_extractor, classifier, train_loader_held, w, device)
      
        # 2. influence
        influence_scores = compute_influence_scores_lissa(feature_extractor, classifier, train_loader_held, target_loader, device)
   
        # 3. w update
        w = update_sample_weights(w, influence_scores, lr=lr_w)
        stats = log_group_weights(w, train_loader_held, device)
        group_weight_log1.append(stats["avg"])
        group_weight_log2.append(stats["sum"])

        # 4. evaluation
        val_acc, _, _, _ = tuple_evaluate(
            feature_extractor, classifier, val_loader_final,
            dataset_name="Val", device=device
        )
        
        if best_val_score is None or val_acc > best_val_score:
            best_val_score = val_acc
            best_classifier = copy.deepcopy(classifier)

    save_group_weight_log(group_weight_log1, 'avg', output_dir="./result")
    save_group_weight_log(group_weight_log2, 'sum', output_dir="./result")

    # Final validation
    final_acc, final_loss, final_group_accs, final_group_losses = tuple_evaluate(
        feature_extractor, best_classifier, test_loader,
        dataset_name="TEST", device=device
    )   

    print(f"\n✅ TEST Evaluation - Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
    for group in sorted(final_group_accs.keys()):
        acc = final_group_accs[group]
        if acc is not None:
            print(f"  Group {group} Accuracy: {acc:.4f}")
        else:
            print(f"  Group {group} Accuracy: N/A (No samples)")

    return best_classifier


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


def save_group_weight_log(group_weight_log, name, output_dir="./result", filename="group_weights.csv" ):
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

    plot_path = os.path.join(output_dir, "group_weight_" + name +".png")
    plt.savefig(plot_path)
    plt.close()

    return csv_path, plot_path