import torch
import torch.nn as nn
import torch.optim as optim

def train_or_eval_model(model, data_loader, params, device, mode="train", loss_fn=None):
    model.to(device)

    if mode == "train":
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.NLLLoss()
        epochs = params["epochs"]
        
        if loss_fn is None:
            criterion = nn.NLLLoss()
            
            def loss_fn(model, x, y, group=None):
                output = model(x)
                log_probs = torch.log_softmax(output, dim=1)
                return criterion(log_probs, y)
        
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch, group_batch, _ in data_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                group_batch = group_batch.to(device)

                optimizer.zero_grad()
                output = model(x_batch)
                loss = loss_fn(model, x_batch, y_batch, group_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(data_loader):.4f}")

        return model  

    elif mode == "eval":
        model.eval()
        outputs = []
        labels = []
        with torch.no_grad():
            for x_batch, y_batch, _, _ in data_loader:
                x_batch = x_batch.to(device)
                output = model(x_batch)
                outputs.append(output.cpu())
                labels.append(y_batch.cpu())

        return torch.cat(outputs), torch.cat(labels)  
