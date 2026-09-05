from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class NoPEConfig:
    d_model: int = 512
    n_head: int = 8
    d_head: int = 96
    qk_head_dim: int = 96
    kv_latent: int = 16
    q_latent: int = 16
    dropout: float = 0.3
    eps: float = 1e-5


class NoPEMLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qk_head_dim = cfg.qk_head_dim
        self.v_head_dim = getattr(cfg, "v_head_dim", cfg.d_head)
        self.p_dropout = cfg.dropout
        eps = getattr(cfg, "norm_eps", getattr(cfg, "eps", 1e-5))
        self.kv_down = nn.Linear(cfg.d_model, cfg.kv_latent, bias=False)
        self.kv_norm = nn.RMSNorm(cfg.kv_latent, eps=eps)
        self.k_up = nn.Linear(cfg.kv_latent, cfg.n_head*self.qk_head_dim, bias=False)
        self.v_up = nn.Linear(cfg.kv_latent, cfg.n_head*self.v_head_dim, bias=False)
        self.q_down = nn.Linear(cfg.d_model, cfg.q_latent, bias=False)
        self.q_norm = nn.RMSNorm(cfg.q_latent, eps=eps)
        self.q_up = nn.Linear(cfg.q_latent, cfg.n_head*self.qk_head_dim, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.n_head*self.v_head_dim, bias=False)
        self.W_o = nn.Linear(cfg.n_head*self.v_head_dim, cfg.d_model, bias=False)
        self.apply(self.init_weight)

    def create_mask(self, past_len, total_len, device):
        #q必须在第1维,k必须在第0维(SDPA需要的attention形状为[B,N,q_len,k_len])
        q = torch.arange(past_len, total_len, device=device)[:, None]
        k = torch.arange(total_len, device=device)[None, :]
        return (k <= q)[None, None]

    def forward(self, x, cache=None, padding_mask=None):
        b, length, _ = x.shape
        mask = (torch.ones((b, length), dtype=torch.bool, device=x.device)
                if padding_mask is None else padding_mask.to(x.device, torch.bool))
        x = x * mask.unsqueeze(-1)
        latent = self.kv_norm(self.kv_down(x))
        past_len = 0 if cache is None else cache[0].size(1)
        keys_mask = mask
        if cache is not None:
            latent = torch.cat((cache[0], latent), dim=1)
            keys_mask = torch.cat((cache[1], mask), dim=1)
        total = latent.size(1)
        k = self.k_up(latent).view(b, total, self.n_head, self.qk_head_dim).transpose(1, 2)
        v = self.v_up(latent).view(b, total, self.n_head, self.v_head_dim).transpose(1, 2)
        q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.view(b, length, self.n_head, self.qk_head_dim).transpose(1, 2)
        attn_mask = self.create_mask(past_len, total, x.device) & keys_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.p_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, length, -1)
        # 保留原始变体的线性输出门控，不将它悄悄换为 sigmoid。
        out = self.W_o(out * self.gate(x)) * mask.unsqueeze(-1)
        return out, (latent, keys_mask)

    @staticmethod
    def init_weight(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)
