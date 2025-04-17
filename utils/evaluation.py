import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from utils.train import train_or_eval_model  

def evaluate(model, dataloader, dataset_name="Test", device="cuda"):
    
    outputs, labels = train_or_eval_model(model, dataloader, dataloader, {}, device, mode="eval")

    criterion = nn.NLLLoss(reduction="none")
    losses = criterion(outputs, labels).squeeze() 

    predictions = outputs.argmax(dim=1)
    probs = torch.softmax(outputs, dim=1)[:, 1]
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    total_loss = losses.sum().item() / total

    try:
        overall_f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="binary")
        print(f"\n{dataset_name} Evaluation - Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, F1-score: {overall_f1:.4f}")
    except:
        overall_f1 = None
        print(f"\n{dataset_name} Evaluation - Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, F1-score: N/A")

    try:
        if len(set(labels.tolist())) < 2:
            overall_auc = None
            print(f"  Overall AUC: N/A (only one class)")
        else:
            overall_auc = roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy())
            print(f"  Overall AUC: {overall_auc:.4f}")
    except:
        overall_auc = None
        print(f"  Overall AUC: N/A (need both classes)")

    # Group-wise metrics
    group_correct = {}
    group_total = {}
    group_loss_sum = {}
    group_probs = {}
    group_labels = {}
    group_preds = {}

    for i in range(total):
        group = dataloader.dataset.groups[i].item()

        if group not in group_correct:
            group_correct[group] = 0
            group_total[group] = 0
            group_loss_sum[group] = 0.0
            group_probs[group] = []
            group_labels[group] = []
            group_preds[group] = []

        group_total[group] += 1
        group_loss_sum[group] += losses[i].item()
        group_probs[group].append(probs[i].item())
        group_labels[group].append(labels[i].item())
        group_preds[group].append(predictions[i].item())

        if predictions[i] == labels[i]:
            group_correct[group] += 1

    group_losses = {}
    group_aucs = {}
    group_f1s = {}

    for group in sorted(group_total.keys()):
        if group_total[group] > 0:
            group_acc = group_correct[group] / group_total[group]
            group_loss_avg = group_loss_sum[group] / group_total[group]
            group_losses[group] = group_loss_avg

            # AUC per group
            if len(set(group_labels[group])) < 2:
                group_aucs[group] = None
                auc_str = ", AUC: N/A (only one class)"
            else:
                auc = roc_auc_score(group_labels[group], group_probs[group])
                group_aucs[group] = auc
                auc_str = f", AUC: {auc:.4f}"

            # F1-score per group
            try:
                f1 = f1_score(group_labels[group], group_preds[group], average="binary")
                group_f1s[group] = f1
                f1_str = f", F1-score: {f1:.4f}"
            except:
                group_f1s[group] = None
                f1_str = ", F1-score: N/A"

            print(f"  Group {group} Accuracy: {group_acc:.4f} ({group_correct[group]}/{group_total[group]}){auc_str}{f1_str}")
        else:
            print(f"  Group {group} Accuracy: N/A (No samples)")
            group_losses[group] = None
            group_aucs[group] = None
            group_f1s[group] = None

    return accuracy, group_losses, overall_auc, group_aucs, overall_f1, group_f1s
