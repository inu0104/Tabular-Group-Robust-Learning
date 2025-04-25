import copy
import torch
import torch.nn.functional as F
from utils.train import train_or_eval_model

def focal_loss(logits, targets, gamma=2.0):
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1 - prob) * (1 - targets)
    loss = - (1 - pt) ** gamma * torch.log(pt + 1e-12)
    return loss.mean()

class GroupDRO:
    def __init__(self, criterion, n_groups, inv_group_ratios, step_size_q=0.01, gamma=2.0, device='cpu'):
        self.criterion = criterion
        self.n_groups = n_groups
        self.step_size_q = step_size_q
        self.gamma = gamma 
        self.q = torch.ones(n_groups, device=device) / n_groups
        self.device = device
        self.inv_group_ratios = inv_group_ratios

    def loss(self, model, x, y, group_idx):
        loss_ls = []

        for g in range(self.n_groups):
            selected = (group_idx == g)
            if selected.sum() > 0:
                x_g = x[selected]
                y_g = y[selected].float().view(-1)
                yhat_g = model(x_g).view(-1)

                # Focal Loss with gamma
                prob = torch.sigmoid(yhat_g)
                pt = prob * y_g + (1 - prob) * (1 - y_g)
                hard_weights = (1 - pt).detach() ** self.gamma
                loss = -hard_weights * torch.log(pt + 1e-12)
                loss = loss.mean()

                loss_ls.append(loss)
            else:
                loss_ls.append(torch.tensor(0.0, device=self.device, requires_grad=True))

        # hardness 기반 group weighting
        q_prime = self.q.clone()
        for g in range(self.n_groups):
            scale = self.inv_group_ratios[g]
            q_prime[g] *= torch.exp(scale * self.step_size_q * loss_ls[g].detach())
        self.q = q_prime / q_prime.sum()

        return sum(self.q[g] * loss_ls[g] for g in range(self.n_groups))


def soft_dro_loss(losses, step_size=0.1):
    loss_tensor = torch.stack(losses)
    weights = torch.exp(step_size * loss_tensor.detach())
    weights = weights / weights.sum()
    total_loss = torch.sum(weights * loss_tensor)
    return total_loss

class SoftGroupDRO:
    def __init__(self, criterion, n_groups, inv_group_ratios, step_size_q=0.1, device='cpu'):
        self.criterion = criterion
        self.n_groups = n_groups
        self.step_size_q = step_size_q
        self.inv_group_ratios = inv_group_ratios
        self.device = device

    def loss(self, model, x, y, group_idx):
        loss_ls = []

        for g in range(self.n_groups):
            selected = (group_idx == g)
            if selected.sum() > 0:
                x_g = x[selected]
                y_g = y[selected].float().view(-1)
                yhat_g = model(x_g).view(-1)
                loss = self.criterion(yhat_g, y_g)
                loss_ls.append(loss)
            else:
                loss_ls.append(torch.tensor(0.0, device=self.device, requires_grad=True))

        # 반영된 soft-DRO loss
        scaled_losses = [self.inv_group_ratios[g] * loss_ls[g] for g in range(self.n_groups)]
        return soft_dro_loss(scaled_losses, step_size=self.step_size_q)


class HardGroupDRO:
    def __init__(self, criterion, n_groups, inv_group_ratios, device='cpu'):
        self.criterion = criterion
        self.n_groups = n_groups
        self.inv_group_ratios = inv_group_ratios
        self.device = device

    def loss(self, model, x, y, group_idx):
        loss_ls = []
        scaled_loss_ls = []

        for g in range(self.n_groups):
            selected = (group_idx == g)
            if selected.sum() > 0:
                x_g = x[selected]
                y_g = y[selected].float().view(-1)
                yhat_g = model(x_g).view(-1)
                loss = self.criterion(yhat_g, y_g)
                loss_ls.append(loss)
                scaled_loss_ls.append(self.inv_group_ratios[g] * loss)
            else:
                dummy = torch.tensor(-float('inf'), device=self.device)
                loss_ls.append(dummy)
                scaled_loss_ls.append(dummy)

        # group weight를 반영한 최악 group 선택
        losses_tensor = torch.stack(scaled_loss_ls)
        worst_group_idx = torch.argmax(losses_tensor).item()
        total_loss = loss_ls[worst_group_idx]

        return total_loss


def get_inverse_group_ratios(train_loader, n_groups, device='cpu'):
    group_counts = torch.zeros(n_groups, device=device)
    total = 0

    for batch in train_loader:
        x, y, group_idx = batch[:3]
        group_idx = group_idx.to(device)

        for g in range(n_groups):
            group_counts[g] += (group_idx == g).sum()

        total += group_idx.size(0)

    group_ratios = group_counts / total
    inv_ratios = 1.0 / (group_ratios + 1e-8)       
    inv_ratios = inv_ratios / inv_ratios.max()     

    return inv_ratios


def run_group_dro_focal(model, train_loader, valid_loader, test_loader, train_df, train_params, device, dataset, method):
    print(f"🔥 Running GroupDRO Method on {device}...")

    n_groups = train_df['group'].nunique()
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    is_tabnet = model.__class__.__name__.lower().startswith("tabnet")
    lambda_sparse = train_params.get("lambda_sparse", 1e-3) if is_tabnet else 0.0
    step_size_q = train_params.get("group_dro_eta", 0.01)
    gamma = train_params.get("gamma", 0.01)

    inv_group_ratios = get_inverse_group_ratios(train_loader, n_groups, device)

    if is_tabnet:
        def loss_fn_erm(model, x, y, group=None, sample_ids=None):
            output, M_loss = model(x, return_loss=True)
            loss = F.binary_cross_entropy_with_logits(output.squeeze(), y.float())
            return loss + lambda_sparse * M_loss
    else:
        loss_fn_erm = None 

    # ##############################
    # # 1. ERM
    # ##############################
    # model_erm = copy.deepcopy(model)
    # model_erm = train_or_eval_model(
    #     model=model_erm,
    #     train_loader=train_loader,
    #     valid_loader=valid_loader,
    #     params=train_params,
    #     device=device,
    #     mode="train",
    #     loss_fn=loss_fn_erm
    # )

    ##############################
    # 2. GroupDRO
    ##############################
    group_dro = GroupDRO(criterion, n_groups=n_groups, inv_group_ratios=inv_group_ratios, step_size_q=0.01, device=device)

    def loss_fn_gdro(model, x, y, group=None, sample_ids=None):
        if is_tabnet:
            output, M_loss = model(x, return_loss=True)
            loss = group_dro.loss(model, x, y, group) 
            
            return loss + lambda_sparse * M_loss
        else:
            return group_dro.loss(model, x, y, group)

    model_gdro = copy.deepcopy(model)
    model_gdro = train_or_eval_model(
        model=model_gdro,
        train_loader=train_loader,
        valid_loader=valid_loader,
        params=train_params,
        device=device,
        mode="train",
        loss_fn=loss_fn_gdro
    )
    return model_gdro
    
    # # ##############################
    # # # 3. SoftGroupDRO
    # # ##############################
    # soft_group_dro = SoftGroupDRO(criterion, n_groups=n_groups, inv_group_ratios=inv_group_ratios, step_size_q=0.1, device=device)

    # def loss_fn_sgdro(model, x, y, group=None, sample_ids=None):
    #     if is_tabnet:
    #         output, M_loss = model(x, return_loss=True)
    #         loss = soft_group_dro.loss(model, x, y, group)
    #         return loss + lambda_sparse * M_loss
    #     else:
    #         return soft_group_dro.loss(model, x, y, group)

    # model_sgdro = copy.deepcopy(model)
    # model_sgdro = train_or_eval_model(
    #     model=model_sgdro,
    #     train_loader=train_loader,
    #     valid_loader=valid_loader,
    #     params=train_params,
    #     device=device,
    #     mode="train",
    #     loss_fn=loss_fn_sgdro
    # )

    # # ##############################
    # # # 4. HardGroupDRO
    # # ##############################
    # hard_group_dro = HardGroupDRO(criterion, n_groups=n_groups, inv_group_ratios=inv_group_ratios, device=device)

    # def loss_fn_hgdro(model, x, y, group=None, sample_ids=None):
    #     if is_tabnet:
    #         output, M_loss = model(x, return_loss=True)
    #         loss = hard_group_dro.loss(model, x, y, group)
    #         return loss + lambda_sparse * M_loss
    #     else:
    #         return hard_group_dro.loss(model, x, y, group)

    # model_hgdro = copy.deepcopy(model)
    # model_hgdro = train_or_eval_model(
    #     model=model_hgdro,
    #     train_loader=train_loader,
    #     valid_loader=valid_loader,
    #     params=train_params,
    #     device=device,
    #     mode="train",
    #     loss_fn=loss_fn_hgdro
    # )

    # return model_erm, model_gdro, model_sgdro, model_hgdro