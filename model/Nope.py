import torch
import torch.nn as nn
from dataclasses import  dataclass
import torch.nn.functional as f
@dataclass
class NoPEConfig:
    d_model:int = 512
    n_head:int = 8
    d_head:int = 96
    qk_head_dim :int = 96
    kv_latent:int = 16
    q_latent:int = 16
    dropout:float = 0.3
    eps:float = 1e-5
    vocab_size = 6400
    kv_embd:int = 64


class GatedMLA(nn.Module):
    def __init__(self,cfg:NoPEConfig):
        super().__init__()

        self.p_dropout= cfg.dropout
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head
        self.q_latent = cfg.q_latent

        #kv
        self.kv_down = nn.Linear(cfg.d_model,cfg.kv_latent,bias=False)
        self.kv_norm = nn.RMSNorm(cfg.kv_latent,eps=cfg.eps)
        self.k_up = nn.Linear(cfg.kv_latent,cfg.n_head*cfg.qk_head_dim,bias=False)
        self.v_up = nn.Linear(cfg.kv_latent,cfg.n_head*cfg.d_head,bias=False)


        #q
        self.q_down = nn.Linear(cfg.d_model,cfg.q_latent,bias=False)
        self.q_norm = nn.RMSNorm(cfg.q_latent,eps=cfg.eps)
        self.q_up = nn.Linear(cfg.q_latent,cfg.n_head*cfg.qk_head_dim,bias=False)


        #gate
        self.gate = nn.Linear(cfg.d_model,cfg.n_head*cfg.d_head,bias=False)
        self.W_o = nn.Linear(cfg.d_head*cfg.n_head,cfg.d_model,bias=False)

        self.apply(self.init_weight)
    def create_mask(self,past_len,total_len,device="cuda"):
        l = total_len-past_len
        #l,1
        q_pos = torch.arange(past_len,total_len,device=device).unsqueeze(-1)
        #1,total_len
        k_pos = torch.arange((total_len),device=device).unsqueeze(0)
        #l,total_len
        causal_mask = q_pos >= k_pos
        attention_mask = causal_mask.view(1,1,l,total_len)
        return attention_mask


    def forward(self,x,cache=None,padding_mask=None):

        b,l,d = x.shape


        if padding_mask is None:
            padding_mask = torch.ones((b,l),dtype=torch.bool,device=x.device)

        x = x*padding_mask.unsqueeze(-1)


        past_len = 0 if cache is None else cache[0].size(-2)
        total_len = past_len + l


        #kv
        kv_latent_now = self.kv_down(x)
        self.kv_latent_now = self.kv_norm(kv_latent_now)
        if cache is not None:
            past_kv_latent = cache[0]
            kv_latent = torch.cat([past_kv_latent,kv_latent_now],dim=1)
            past_padding_mask = cache[1]
            padding_mask = torch.cat([padding_mask,past_padding_mask],dim=1)
        else:
            kv_latent = kv_latent_now

        k = self.k_up(kv_latent).view(b,total_len,self.n_head,self.d_head)
        v = self.k_up(kv_latent).view(b,total_len,self.n_head,self.d_head)

        #q
        q = self.q_up(self.q_norm(self.q_down(x)))

        attention_mask = self.create_mask(past_len,total_len,device=x.device)
        attention_mask = attention_mask & padding_mask.view(b,1,l,total_len)


        drop_p = self.p_dropout if self.training else 0

        output = f.scaled_dot_product_attention(q,k,v,is_causal=False,attn_mask=attention_mask,dropout_p=drop_p)


        output = output.transpose(1,2).view(b,l,-1)
        gate = self.gate(x)
        output = output*gate
        output =self.W_o(output) *padding_mask.unsqueeze(-1)

    @staticmethod
    def init_weight(module:nn.Module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            torch.nn.init.trunc_normal_(module.weight,std=0.02)
        elif isinstance(module,nn.RMSNorm):
            torch.nn.init.ones_(module.weight)
if __name__ == "__main__":
    x = torch.randn((1,1,512))
    mla = GatedMLA(NoPEConfig)
    print(mla(x))