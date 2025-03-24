import copy
import torch

from utils.evaluation import evaluate
from utils.train import train_or_eval_model
from utils.gsr_utils import (
    get_feature_extractor,
    split_train_set,
    split_validation_set,
    train_last_layer,
    compute_influence_scores,
    update_sample_weights,
    Classifier,
    tuple_evaluate
)

def run_gsr(model, train_loader, val_loader, test_loader, train_df, train_params, device, dataset):

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

    for step in range(outer_steps):

        # 1. classifier ψ 
        classifier = Classifier(input_dim=feature_dim, num_classes=num_classes)
        classifier = train_last_layer(feature_extractor, classifier, train_loader_held, w, device)

        # 2. influence
        influence_scores = compute_influence_scores(
            feature_extractor, classifier, train_loader_held, target_loader, device
        )

        # 3. w update
        w = update_sample_weights(w, influence_scores, lr=lr_w)

        # 4. evaluation
        val_acc, _, _, _ = tuple_evaluate(
            feature_extractor, classifier, val_loader_final,
            dataset_name="Val", device=device
        )
        
        if best_val_score is None or val_acc > best_val_score:
            best_val_score = val_acc
            best_classifier = copy.deepcopy(classifier)

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
