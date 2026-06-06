import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LambdaConditioner(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode(self, lambda_val, batch_size, device):
        if lambda_val is None:
            lam = torch.zeros(batch_size, 1, device=device)
        elif torch.is_tensor(lambda_val):
            lam = lambda_val.to(device=device, dtype=torch.float32)
            if lam.dim() == 0:
                lam = lam.view(1, 1).expand(batch_size, 1)
            elif lam.dim() == 1:
                if lam.numel() == 1:
                    lam = lam.view(1, 1).expand(batch_size, 1)
                elif lam.numel() == batch_size:
                    lam = lam.view(batch_size, 1)
                else:
                    raise ValueError("lambda batch size mismatch.")
        else:
            lam = torch.tensor(lambda_val, device=device, dtype=torch.float32)
            lam = lam.view(1, 1).expand(batch_size, 1)
        lam = lam.clamp(0.0, 1.0)
        return self.mlp(lam)


class ConditionalLayerNorm(nn.Module):
    def __init__(self, ln, embed_dim):
        super().__init__()
        if isinstance(ln.normalized_shape, int):
            self.normalized_shape = (ln.normalized_shape,)
        else:
            self.normalized_shape = tuple(ln.normalized_shape)
        self.hidden_size = self.normalized_shape[-1]
        self.eps = ln.eps
        if ln.elementwise_affine:
            self.base_weight = nn.Parameter(ln.weight.data.clone())
            self.base_bias = nn.Parameter(ln.bias.data.clone())
        else:
            self.base_weight = nn.Parameter(torch.ones(self.hidden_size))
            self.base_bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.to_scale_shift = nn.Linear(embed_dim, 2 * self.hidden_size)
        self.current_cond = None

    def set_condition(self, cond):
        self.current_cond = cond

    def forward(self, x):
        normalized = F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        if x.dim() == 2:
            base_shape = (1, self.hidden_size)
            cond_shape = (x.size(0), self.hidden_size)
        else:
            base_shape = (1,) + (1,) * (x.dim() - 2) + (self.hidden_size,)
            cond_shape = (x.size(0),) + (1,) * (x.dim() - 2) + (self.hidden_size,)
        if self.current_cond is None:
            gamma = self.base_weight.view(*base_shape)
            beta = self.base_bias.view(*base_shape)
            return normalized * gamma + beta
        delta_gamma, delta_beta = self.to_scale_shift(self.current_cond).chunk(2, dim=-1)
        gamma = self.base_weight.unsqueeze(0) + delta_gamma
        beta = self.base_bias.unsqueeze(0) + delta_beta
        return normalized * gamma.view(*cond_shape) + beta.view(*cond_shape)


def set_condition(model, cond):
    for module in model.modules():
        if hasattr(module, "set_condition"):
            module.set_condition(cond)


def replace_layernorms(module, embed_dim):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, ConditionalLayerNorm(child, embed_dim))
        else:
            replace_layernorms(child, embed_dim)


def build_swin_base(num_classes):
    return timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes,
    )


class ConditionalSwinBase(nn.Module):
    def __init__(self, num_classes, lambda_embed_dim=16):
        super().__init__()
        self.base = build_swin_base(num_classes)
        self.lambda_conditioner = LambdaConditioner(lambda_embed_dim)
        if hasattr(self.base, "layers") and self.base.layers:
            last_stage = self.base.layers[-1]
            for block in last_stage.blocks:
                replace_layernorms(block, lambda_embed_dim)

    def forward(self, x, lambda_val=None):
        cond = self.lambda_conditioner.encode(lambda_val, x.size(0), x.device)
        set_condition(self.base, cond)
        return self.base(x)
