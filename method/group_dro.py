import copy
import torch
from utils.train import train_or_eval_model
from utils.evaluation import evaluate

class GroupDRO:
    def __init__(self, criterion, n_groups, step_size_q=0.01, device='cpu'):
        self.criterion = criterion              # loss function
        self.n_groups = n_groups                # number of groups
        self.step_size_q = step_size_q          # learning rate for q
        self.q = torch.ones(n_groups, device=device) / n_groups  # group weights
        self.device = device

    def loss(self, model, x, y, group_idx):
        loss_ls = []

        for g in range(self.n_groups):
            selected = (group_idx == g)
            if selected.sum() > 0:
                x_g = x[selected]
                y_g = y[selected]
                yhat_g = model(x_g)
                log_probs = torch.log_softmax(yhat_g, dim=1)
                loss = self.criterion(log_probs, y_g)
                loss_ls.append(loss)
            else:
                loss_ls.append(torch.tensor(0.0, device=self.device, requires_grad=True))

        q_prime = self.q.clone()
        for g in range(self.n_groups):
            q_prime[g] *= torch.exp(self.step_size_q * loss_ls[g].detach())

        self.q = q_prime / q_prime.sum()

        total_loss = sum(self.q[g] * loss_ls[g] for g in range(self.n_groups))
        return total_loss

def run_group_dro(model, train_loader, valid_loader, test_loader, train_df, train_params, device, dataset, method):
    print(f"🔥 Running GroupDRO Method on {device}...")

    n_groups = train_df['group'].nunique()
    criterion = torch.nn.NLLLoss(reduction='mean')
    group_dro = GroupDRO(criterion, n_groups=n_groups, step_size_q=0.01, device=device)

    def loss_fn(model, x, y, group):
        return group_dro.loss(model, x, y, group)

    trained_model = train_or_eval_model(model, train_loader, train_params, device, mode="train", loss_fn=loss_fn)
    _, _ = evaluate(trained_model, test_loader, dataset_name="Test", device=device)

    return trained_model