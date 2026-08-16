import torch
import torch.nn as nn
from dataclasses import dataclass
from rotary_embedding_torch import RotaryEmbedding
import torch.nn.functional as f



@dataclass
class Config:
    d_model = 512
    n_head = 8
    qk_nope = 48
    qk_rope = 16
    v_head_dim = 64
    qk_head_dim = 64
    kv_latent = 16
    q_latent = 16
    dropout = 0.5
    n_layer = 3
class MLA(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qk_nope = cfg.qk_nope
        self.qk_rope = cfg.qk_rope
        self.qk_head_dim = cfg.qk_nope + cfg.qk_rope
        self.v_head_dim = cfg.v_head_dim
        self.dropout = cfg.dropout

        #KV
        self.kv_down = nn.Linear(cfg.d_model,cfg.kv_latent)
        self.k_r = nn.Linear(cfg.d_model, cfg.n_head * cfg.qk_rope)
        self.k_up = nn.Linear(cfg.kv_latent,cfg.n_head*cfg.qk_nope)
        self.v_up = nn.Linear(cfg.kv_latent,cfg.n_head*cfg.v_head_dim)
        self.kv_norm = nn.RMSNorm(cfg.kv_latent)

        #Q
        self.q_down = nn.Linear(cfg.d_model,cfg.q_latent)
        self.q_up = nn.Linear(cfg.q_latent,cfg.n_head*cfg.qk_nope)
        self.q_r = nn.Linear(cfg.q_latent,cfg.n_head*cfg.qk_rope)
        self.q_norm = nn.RMSNorm(cfg.q_latent)


        #rope
        self.rope = RotaryEmbedding(cfg.qk_rope)


        self.W_o = nn.Linear(cfg.n_head * cfg.v_head_dim, cfg.d_model)

        #init
        self.apply(self._init_weights)
    def forward(self,x,kv_cache=None,use_cache=True):
        """

        :param x:
        :param kv_cache: (kv_latent,k_rope)
        :return:
        """
        residual = x
        b, l, d = x.shape

        past_len = 0 if kv_cache is None else kv_cache[0].size(-2)
        total_len= past_len+l

        #KV_press
        #b l n*(kv_latent)
        kv_latent_now = self.kv_down(x)
        kv_latent_now = self.kv_norm(kv_latent_now)


        #k_rope
        k_rope_now = self.k_r(x).view(b, l, self.n_head, self.qk_rope)
        k_rope_now = self.rope.rotate_queries_or_keys(k_rope_now, offset=past_len)
        #k_rope_history
        if kv_cache is not None:
            k_rope = torch.cat([kv_cache[1], k_rope_now], dim=1)
        else:
            k_rope = k_rope_now



        #cat the historic kv
        if kv_cache is not None:
            kv_latent = torch.cat([kv_cache[0], kv_latent_now], dim=1)
        else:
            kv_latent = kv_latent_now
        k_nope = self.k_up(kv_latent).view(b, total_len, self.n_head, self.qk_nope)
        k = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)



        #v
        v = self.v_up(kv_latent)
        v = v.view(b,total_len,self.n_head,self.v_head_dim).transpose(1,2)



        #Q
        q_latent = self.q_down(x)
        q_latent = self.q_norm(q_latent)
        q_nope = self.q_up(q_latent).view(b, l, self.n_head, self.qk_nope)
        q_rope = self.q_r(q_latent).view(b, l, self.n_head, self.qk_rope)
        q_rope = self.rope.rotate_queries_or_keys(q_rope,offset=past_len)
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)



        # 生成q,k的全局位置,判断q可以查看哪些k
        # 只生成当前的q的位置即可,ps:[3,4]
        # 升维 变为(2,1)
        q_pos = torch.arange(past_len, total_len, device=x.device).unsqueeze(1)
        # k生成历史所有的位置,因为q要和所有的k做计算,ps[0,1,2,3,4]
        # 升维变为(1,5)
        k_pos = torch.arange(total_len, device=x.device).unsqueeze(0)
        # 只让q和q位置以前的k进行计算
        causal_mask = k_pos <= q_pos



        #MHA
        dropout_p = self.dropout if self.training else 0.0
        output = f.scaled_dot_product_attention(q, k, v, is_causal=False,attn_mask=causal_mask,dropout_p=dropout_p)
        output = output.transpose(1,2).contiguous().view(b,l,-1)
        output = self.W_o(output)


        if use_cache:
            new_kv_cache = (kv_latent,k_rope)
            return output + residual, new_kv_cache

        return output + residual

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)

            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

if __name__ == "__main__":
    x = torch.randn(1,32,512)
    mla = MLA(Config())
    print(mla(x))