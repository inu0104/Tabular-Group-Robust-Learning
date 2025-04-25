import torch
import torch.nn as nn
import torch.optim as optim

def train_or_eval_model(model, train_loader, valid_loader, params, device, mode="train", loss_fn=None, sample_weights=None):
    model.to(device)
    if mode in ["train", "trans"]:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        epochs = params["epochs"]
        patience = params.get("patience", 5) 
        best_val_loss = float("inf")
        no_improve_count = 0
        best_model_state = None
        
        if sample_weights is not None:
            idx_to_weight = {int(sample_id.item()): weight.item() for sample_id, weight in sample_weights}
        
        if loss_fn is None:
            def loss_fn(model, x, y, sample_ids=None):
                output = model(x).squeeze(-1)
                y = y.float()
                losses = criterion(output, y)
                if sample_weights is not None and sample_ids is not None:
                    weights_list = [idx_to_weight[int(i)] for i in sample_ids.tolist()]
                    weights = torch.tensor(weights_list, device=device)
                    return (losses * weights).sum() / train_loader.batch_size * len(train_loader.dataset)
                else:
                    return losses.mean()
        
################## TRAIN ######################################################        
        for epoch in range(epochs):
            total_loss = 0
            model.train()
            
            if mode == 'trans':
          
                for x_categ, x_numer, y_batch, group_batch, sample_id_batch in train_loader:
                    x_categ, x_numer = x_categ.to(device), x_numer.to(device)
                    x_batch = (x_categ, x_numer)
                    y_batch = y_batch.to(device).float()
                    sample_id_batch = sample_id_batch.to(device)

                    optimizer.zero_grad()
                    loss = loss_fn(model, x_batch, y_batch, group_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)

            else : 
                for x_batch, y_batch, group_batch, sample_id_batch, _ in train_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    group_batch = group_batch.to(device)
                    sample_id_batch = sample_id_batch.to(device)

                    optimizer.zero_grad()
                    loss = loss_fn(model, x_batch, y_batch, group_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)

################## VALID ######################################################    
            model.eval()
            val_loss_total = 0
            model.eval()
            with torch.no_grad():
                if mode == 'trans':
                    for x_cat, x_num, y_val, g_val, _ in valid_loader:
                        x_cat = x_cat.to(device)
                        x_num = x_num.to(device)
                        x_val = (x_cat,x_num)
                        y_val = y_val.to(device).float()
                        loss = loss_fn(model, x_val, y_val, g_val)
                        val_loss_total += loss.item()
                else:
                    for x_val, y_val, g_val, _, _ in valid_loader:
                        x_val = x_val.to(device)
                        y_val = y_val.to(device).float()
                        loss = loss_fn(model, x_val, y_val, g_val)
                        val_loss_total += loss.item()

            avg_val_loss = val_loss_total / len(valid_loader)
            
            #print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # early stopping 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    #print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model  

    elif mode == "eval":
        model.eval()
        outputs = []
        labels = []
        with torch.no_grad():
            for x_batch, y_batch, _, _, _ in valid_loader:
                x_batch = x_batch.to(device)
                output = model(x_batch)
                outputs.append(output.cpu())
                labels.append(y_batch.cpu())

        return torch.cat(outputs), torch.cat(labels)


def convert_weights_tensor_to_dict(w_with_id):
    return {int(sample_id.item()): weight.item() for sample_id, weight in w_with_id}