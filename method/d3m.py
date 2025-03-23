import os
import copy
import torch
import pandas as pd

from utils.evaluation import evaluate 
from utils.train import train_or_eval_model
from utils.compute_tau import compute_tau_vector
from utils.compute_gradients import compute_grad
from utils.alignment import compute_group_alignment_score
from utils.alignment import create_new_train_dataset_groupwise


def run_d3m(model, train_loader, test_loader, train_df, train_params, device, dataset):

    print(f"🔥 Running D3M Method on {device}: Initial Training Phase...")

    model2 = copy.deepcopy(model)
    trained_model = train_or_eval_model(model, train_loader, train_params, device, mode="train")

    #_, _ = evaluate(trained_model, train_loader, dataset_name="Train", device=device)
    _, test_group_losses = evaluate(trained_model, test_loader, dataset_name="Test", device=device)

    #torch.save(trained_model.state_dict(), "model_state/d3m_node.pth")
    
    train_grad, train_sample = compute_grad(trained_model, train_loader, device)
    #test_grad, test_sample = compute_grad(trained_model, test_loader, device)
    
    tau_vector_df = compute_tau_vector(train_grad, train_sample, trained_model, test_loader, device)
    alignment_scores_df = compute_group_alignment_score(tau_vector_df, train_loader, test_group_losses)
    
    remove_ratio = 0.05
    train_loader_new = create_new_train_dataset_groupwise(alignment_scores_df, train_df, dataset, remove_ratio)

    trained_model_new = train_or_eval_model(model2, train_loader_new, train_params, device, mode="train")
    #_, _ = evaluate(trained_model_new, train_loader_new, dataset_name="Train", device=device)
    _, _ = evaluate(trained_model_new, test_loader, dataset_name="Test", device=device)
    
    return 0  
