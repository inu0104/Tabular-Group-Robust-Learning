import torch
import torch.nn as nn
import torch.optim as optim

def train_or_eval_model_gdro(model, train_loader, valid_loader, params, device, loss_fn, sample_weights):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.NLLLoss(reduction='mean')
    epochs = params["epochs"]
    patience = params.get("patience", 10)
    best_val_loss = float("inf")
    no_improve_count = 0
    best_model_state = None

    for epoch in range(epochs):
        model.eval()
        group_losses = {}

        with torch.no_grad():
            for g in group_ids:
                loader = DataLoader(train_tensor_dict[g], batch_size=256, shuffle=False)
                losses = []
                for x_batch, y_batch in loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    output = model(x_batch)
                    log_probs = torch.log_softmax(output, dim=1)
                    loss = criterion(log_probs, y_batch)
                    losses.append(loss.item())
                group_losses[g] = sum(losses) / len(losses) if losses else 0.0

        # worst group 선택
        worst_group = max(group_losses, key=group_losses.get)
        worst_loader = DataLoader(train_tensor_dict[worst_group], batch_size=256, shuffle=True)

        model.train()
        total_loss = 0
        for x_batch, y_batch in worst_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            log_probs = torch.log_softmax(output, dim=1)
            loss = criterion(log_probs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(worst_loader)

        # validation
        model.eval()
        group_val_losses = {}
        with torch.no_grad():
            for g in group_ids:
                group_val_losses[g] = []

            for x_val, y_val, group_val, _, _ in valid_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                group_val = group_val.to(device)
                output = model(x_val)
                log_probs = torch.log_softmax(output, dim=1)
                loss = nn.NLLLoss(reduction='none')(log_probs, y_val)

                for g in group_ids:
                    mask = (group_val == g)
                    if mask.sum() > 0:
                        group_val_losses[g].append(loss[mask].mean().item())

        group_avg_losses = {g: sum(v)/len(v) for g, v in group_val_losses.items() if len(v) > 0}
        worst_val_loss = max(group_avg_losses.values())

        if worst_val_loss < best_val_loss:
            best_val_loss = worst_val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model