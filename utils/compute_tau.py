import torch
import pandas as pd

def compute_tau_vector(train_grad, train_sample, model, test_loader, device):
  
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

    all_logits = []
    
    with torch.no_grad():
        for X_batch, _, _, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).cpu()
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)

    # 1 - σ(f(z_valid)) 
    sigma_fz = torch.sigmoid(all_logits)
    one_minus_sigma = (1 - sigma_fz).mean().to(device)

    tau_list = []

    # τ(z_train)
    with torch.no_grad():
        for i in range(num_train_samples):
            g_z_train = all_gradients[i].view(-1, 1)

            tau_i = torch.matmul(g_z_train.T, torch.matmul(G_sigma_inv, g_z_train)).squeeze() * one_minus_sigma
            tau_list.append(tau_i.item())

    tau_vector_df = pd.DataFrame({"sample_id": sample_ids.cpu().numpy(), "tau": tau_list})
    
    print("✅ τ vector calculation completed successfully.")
    return tau_vector_df
