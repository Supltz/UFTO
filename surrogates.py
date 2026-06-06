import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_center(dist):
    row_mean = dist.mean(dim=1, keepdim=True)
    col_mean = dist.mean(dim=0, keepdim=True)
    grand_mean = dist.mean()
    return dist - row_mean - col_mean + grand_mean


def conditional_distance_correlation(x, y, labels):
    if x.shape[0] < 2:
        return torch.tensor(0.0, device=x.device)
    x = x.float()
    y = y.float()
    labels = labels.to(x.device)
    n = x.shape[0]
    d_cov_sum = torch.tensor(0.0, device=x.device)
    d_var_x_sum = torch.tensor(0.0, device=x.device)
    d_var_y_sum = torch.tensor(0.0, device=x.device)
    has_valid_class = False

    for cls in torch.unique(labels):
        idx = labels == cls
        if idx.sum() < 2:
            continue
        has_valid_class = True
        x_cls = x[idx]
        y_cls = y[idx]
        dist_x = torch.cdist(x_cls, x_cls, p=2)
        dist_y = torch.cdist(y_cls, y_cls, p=2)
        A = _double_center(dist_x)
        B = _double_center(dist_y)
        d_cov_sum += (A * B).sum()
        d_var_x_sum += (A * A).sum()
        d_var_y_sum += (B * B).sum()

    if not has_valid_class:
        return torch.tensor(0.0, device=x.device)

    if d_var_x_sum <= 1e-12 or d_var_y_sum <= 1e-12:
        return torch.tensor(0.0, device=x.device)
    denom = torch.sqrt(d_var_x_sum * d_var_y_sum)
    d_cor_sq = d_cov_sum / denom
    return torch.sqrt(torch.clamp(d_cor_sq, min=0.0, max=1.0))


def conditional_distance_correlation_joint(x, y, labels, sensitive_labels):
    if x.shape[0] < 2:
        return torch.tensor(0.0, device=x.device)
    x = x.float()
    y = y.float()
    labels = labels.to(x.device)
    sensitive_labels = sensitive_labels.to(x.device)
    n = x.shape[0]
    d_cov_sum = torch.tensor(0.0, device=x.device)
    d_var_x_sum = torch.tensor(0.0, device=x.device)
    d_var_y_sum = torch.tensor(0.0, device=x.device)
    has_valid_group = False

    if labels.dim() == 1:
        pairs = torch.stack([labels, sensitive_labels], dim=1)
        unique_pairs = torch.unique(pairs, dim=0)

        def pair_mask(pair):
            return (labels == pair[0]) & (sensitive_labels == pair[1])
    else:
        pairs = torch.cat([labels, sensitive_labels.unsqueeze(1)], dim=1)
        unique_pairs = torch.unique(pairs, dim=0)

        def pair_mask(pair):
            return (labels == pair[:-1]).all(dim=1) & (sensitive_labels == pair[-1])

    for pair in unique_pairs:
        idx = pair_mask(pair)
        if idx.sum() < 2:
            continue
        has_valid_group = True
        x_grp = x[idx]
        y_grp = y[idx]
        dist_x = torch.cdist(x_grp, x_grp, p=2)
        dist_y = torch.cdist(y_grp, y_grp, p=2)
        A = _double_center(dist_x)
        B = _double_center(dist_y)
        d_cov_sum += (A * B).sum()
        d_var_x_sum += (A * A).sum()
        d_var_y_sum += (B * B).sum()

    if not has_valid_group:
        return torch.tensor(0.0, device=x.device)

    if d_var_x_sum <= 1e-12 or d_var_y_sum <= 1e-12:
        return torch.tensor(0.0, device=x.device)
    denom = torch.sqrt(d_var_x_sum * d_var_y_sum)
    d_cor_sq = d_cov_sum / denom
    return torch.sqrt(torch.clamp(d_cor_sq, min=0.0, max=1.0))

