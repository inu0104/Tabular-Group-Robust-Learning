import torch
import os

def compute_grad(model, dataloader, device="cuda"):

    model.eval()

    all_gradients = []
    all_sample_ids = []

    for batch_idx, (X_batch, _, _, sample_ids) in enumerate(dataloader):
        X_batch = X_batch.to(device, non_blocking=True)

        for i in range(X_batch.shape[0]):
            X_single = X_batch[i].unsqueeze(0).to(device)
            X_single.requires_grad_()

            output = model(X_single).float()
            grad_outputs = torch.ones_like(output, device=device)

            grads = torch.autograd.grad(outputs=output,
                                        inputs=[param for param in model.parameters() if param.requires_grad],
                                        grad_outputs=grad_outputs,
                                        retain_graph=False)

            grad_vector = torch.cat([g.flatten() for g in grads], dim=0).to("cpu")

            all_gradients.append(grad_vector)
            all_sample_ids.append(sample_ids[i].item())

            del X_single, output, grads, grad_outputs, grad_vector
            torch.cuda.empty_cache()

    all_gradients_tensor = torch.stack(all_gradients)
    all_sample_ids_tensor = torch.tensor(all_sample_ids)

    return all_gradients_tensor, all_sample_ids_tensor
    
