import torch
import torch.nn as nn
from utils.train import train_or_eval_model  

def evaluate(model, dataloader, dataset_name="Test", device="cuda"):
    
    outputs, labels = train_or_eval_model(model, dataloader, {}, device, mode="eval")

    criterion = nn.NLLLoss(reduction="none")
    losses = criterion(outputs, labels).squeeze() 

    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    total_loss = losses.sum().item() / total

    print(f"\n{dataset_name} Evaluation - Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    group_correct = {}
    group_total = {}
    group_loss_sum = {}

    for i in range(total):
        group = dataloader.dataset.groups[i].item()  

        if group not in group_correct:
            group_correct[group] = 0
            group_total[group] = 0
            group_loss_sum[group] = 0.0
        
        group_total[group] += 1
        group_loss_sum[group] += losses[i].item()

        if predictions[i] == labels[i]:
            group_correct[group] += 1

    group_losses = {}
    for group in sorted(group_total.keys()):
        if group_total[group] > 0:
            group_acc = group_correct[group] / group_total[group]
            group_loss_avg = group_loss_sum[group] / group_total[group]
            group_losses[group] = group_loss_avg

            print(f"  Group {group} Accuracy: {group_acc:.4f} ({group_correct[group]}/{group_total[group]})")
        else:
            print(f"  Group {group} Accuracy: N/A (No samples)")
            group_losses[group] = None  

    return accuracy, group_losses  
