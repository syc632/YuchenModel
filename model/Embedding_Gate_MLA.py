import torch
import torch.nn as nn
from dataclasses import dataclass
from rotary_embedding_torch import RotaryEmbedding
import torch.nn.functional as f
from scipy.special.cython_special import kv


@dataclass
class EGConfig:
    d_model = 512
    n_head = 8
    qk_nope = 48
    qk_rope = 16
    v_head_dim = 64
    qk_head_dim = 64
    kv_latent = 16
    q_latent = 16
    dropout = 0.5
    norm_eps: float = 1e-5
    vocab_size  = 6400
    kv_embd:int = 64
class EG_MLA(nn.Module):
    def __init__(self,cfg:EGConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qk_nope = cfg.qk_nope
        self.qk_rope = cfg.qk_rope
        self.qk_head_dim = cfg.qk_nope + cfg.qk_rope
        self.v_head_dim = cfg.v_head_dim
        self.dropout = cfg.dropout

        #KV
        self.kv_down = nn.Linear(cfg.d_model,cfg.kv_latent)
        self.k_r = nn.Linear(cfg.d_model, cfg.n_head * cfg.qk_rope,bias=False)
        self.k_up = nn.Linear(cfg.kv_latent,cfg.n_head*cfg.qk_nope,bias=False)
        self.v_up = nn.Linear(cfg.kv_latent,cfg.n_head*cfg.v_head_dim,bias=False)
        self.kv_norm = nn.RMSNorm(cfg.kv_latent,eps=cfg.norm_eps)

        #Q
        self.q_down = nn.Linear(cfg.d_model,cfg.q_latent,bias=False)
        self.q_up = nn.Linear(cfg.q_latent,cfg.n_head*cfg.qk_nope,bias=False)
        self.q_r = nn.Linear(cfg.q_latent,cfg.n_head*cfg.qk_rope,bias=False)
        self.q_norm = nn.RMSNorm(cfg.q_latent,eps=cfg.norm_eps)


        #rope
        self.rope = RotaryEmbedding(cfg.qk_rope)


        #Embedding Gate
        self.kv_embd = nn.Embedding(cfg.vocab_size,cfg.kv_embd)
        self.kv_embd_norm = nn.RMSNorm(cfg.n_head*(cfg.qk_nope*cfg.v_head_dim),eps=cfg.norm_eps)
        self.eg_gate_up = nn.Linear(cfg.kv_embd,cfg.n_head*(cfg.qk_nope*cfg.v_head_dim),bias=False)


        self.W_o = nn.Linear(cfg.n_head * cfg.v_head_dim, cfg.d_model,bias=False)

        #init
        self.apply(self._init_weights)
    def forward(self,x,token_ids,cache=None,padding_mask=None):
        """
        :param x: 当前输入，形状为 [B, L, D]
        :param cache: (kv_latent, k_rope, key_padding_mask)
        :param token_ids: 当前输入的 token id，形状为 [B, L]
        :param padding_mask: 当前输入的有效 token 掩码，形状为 [B, L]
        :return: (attention_output, next_cache)
        """
        b, l, d = x.shape


        #padding_mask
        if padding_mask is None:
            padding_mask = torch.ones((b,l),device=x.device,dtype=torch.bool)

        else:
            padding_mask = padding_mask.to(device=x.device,dtype = torch.bool)
        #防止padding进入qkv矩阵
        x = x*padding_mask.unsqueeze(-1)



        past_len = 0 if cache is None else cache[0].size(-2)
        total_len= past_len+l

        #KV_press
        #b l n*(kv_latent)
        kv_latent_now = self.kv_down(x)
        kv_latent_now = self.kv_norm(kv_latent_now)


        #k_rope
        k_rope_now = self.k_r(x).view(b, l, self.n_head, self.qk_rope)
        k_rope_now = self.rope.rotate_queries_or_keys(k_rope_now,offset=past_len,seq_dim=-3)
        #k_rope_history
        if cache is not None:
            k_rope = torch.cat([cache[1], k_rope_now], dim=1)
        else:
            k_rope = k_rope_now



        #cat the historic kv and padding
        if cache is not None:
            kv_latent = torch.cat([cache[0], kv_latent_now], dim=1)
            # 历史token是否为padding是无法从头推导的(forward输入不包含历史token的任何信息),因此需要把padding加入到cache中
            past_padding_mask =  cache[2]
            key_padding_mask = torch.cat([past_padding_mask, padding_mask], dim=1)
        else:
            kv_latent = kv_latent_now
            key_padding_mask = padding_mask



        # Embedding Gate
        eg_gate = self.kv_embd(token_ids)
        eg_gate = self.kv_embd_norm(eg_gate)
        eg_gate = self.eg_gate_up(eg_gate)

        k_nope = self.k_up(kv_latent).view(b, total_len, self.n_head, self.qk_nope)
        k = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        k = k*eg_gate


        #v
        v = self.v_up(kv_latent)
        v = v.view(b,total_len,self.n_head,self.v_head_dim).transpose(1,2)
        v = v*eg_gate




        #Q
        q_latent = self.q_down(x)
        q_latent = self.q_norm(q_latent)
        q_nope = self.q_up(q_latent).view(b, l, self.n_head, self.qk_nope)
        q_rope = self.q_r(q_latent).view(b, l, self.n_head, self.qk_rope)
        q_rope = self.rope.rotate_queries_or_keys(q_rope,offset=past_len,seq_dim=-3)
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
        #&两个位置都为True,结果才是True
        attention_mask = causal_mask.view(1,1,l,total_len) & key_padding_mask.view(b,1,1,total_len)




        #MHA
        dropout_p = self.dropout if self.training else 0.0
        output = f.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
        )
        output = output.transpose(1,2).contiguous().view(b,l,-1)
        #softmax会把padding位置变为非零输出
        output = self.W_o(output)*padding_mask.unsqueeze(-1)

        # MLA 只返回 mixing 结果；残差连接由外层 ModelLayer 统一处理。
        next_cache = (kv_latent,k_rope,key_padding_mask)
        return output,next_cache

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