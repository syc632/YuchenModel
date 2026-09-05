import torch
import torch.nn as nn
from dataclasses import dataclass
from rotary_embedding_torch import RotaryEmbedding



@dataclass
class DSAConfig:
    index_n_head:int = 64
    index_head_dim = 128
    index_top_k: int = 50
    d_model = 1024
    n_head = 8
    qk_nope = 48
    qk_rope = 16
    v_head_dim = 128
    qk_head_dim = 64
    kv_latent = 16
    q_latent = 16
    dropout = 0.5
    norm_eps: float = 1e-5



class LightingIndexer(nn.Module):
    def __init__(self,cfg:DSAConfig):
        super().__init__()
        factory_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
        self.index_n_head = cfg.index_n_head
        self.index_head_dim = cfg.index_head_dim
        self.index_top_k = cfg.index_top_k
        self.W_q = nn.Linear(cfg.d_model,cfg.index_n_head*cfg.index_head_dim,bias=False,**factory_kwargs)
        self.W_k = nn.Linear(cfg.d_model,cfg.index_head_dim,bias=False,**factory_kwargs)
        self.W_w = nn.Linear(cfg.d_model,cfg.index_n_head,bias=False,**factory_kwargs)


        self.relu = nn.ReLU()
    def forward(self,x,pask_kv,causal_mask=None):
        b,l,d = x.shape
        _,s,_ = pask_kv

        W = self.W_w(x).unsqueeze(-1) #b l n 1
        q = self.W_q(x).view(b,l,self.index_n_head,self.index_head_dim)
        k = self.W_k(pask_kv).view(b,s,self.index_head_dim).unsqueeze(1)  #b 1 s d_head

        score = (q@k.transpose(-1,-2))/(self.index_head_dim**0.5) #b l n s
        score = self.relu(score)

        score = (W*score).sum(dim=-2) #b l s
        top_indices = torch.topk(score,k=self.index_top_k).indices

        if causal_mask is not None:
            top_indices = top_indices.masked_fill(causal_mask == 0, -1e9)

        return top_indices #b l k

class MLAWithDSA(nn.Module):
    def __init__(self,cfg:DSAConfig):
        super().__init__()

        factory_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}

        self.cfg = cfg
        self.n_head = cfg.n_head
        self.qk_nope = cfg.qk_nope
        self.qk_head_dim = cfg.qk_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.p_dropout = cfg.dropout


        #kv
        self.kv_down = nn.Linear(cfg.d_model,cfg.kv_latent,bias=False,**factory_kwargs)
        self.k_r = nn.Linear(cfg.d_model,cfg.qk_rope,bias=False,**factory_kwargs)
        self.k_up = nn.Linear(cfg.kv_latent,cfg.n_head*cfg.qk_nope,bias=False,**factory_kwargs)
        self.v_up = nn.Linear(cfg.kv_latent,cfg.n_head*cfg.v_head_dim,bias=False,**factory_kwargs)
        self.kv_norm = nn.RMSNorm(cfg.kv_latent,eps=cfg.norm_eps,device=torch.device("cuda"))


        #q
        self.q_down = nn.Linear(cfg.d_model,cfg.q_latent,bias=False,**factory_kwargs)
        self.q_r = nn.Linear(cfg.d_model,cfg.qk_rope,bias=False,**factory_kwargs)
        self.q_up = nn.Linear(cfg.q_latent,cfg.qk_nope*cfg.n_head,bias=False,**factory_kwargs)
        self.q_norm = nn.RMSNorm(cfg.q_latent,eps=cfg.norm_eps)

        #rope
        self.rope = RotaryEmbedding(cfg.qk_rope)

        #LightingIndex
        self.indexer = LightingIndexer(cfg)



        self.W_o = nn.Linear(cfg.v_head_dim*cfg.n_head,cfg.d_model,bias=False,**factory_kwargs)

        self.apply(self.init)
    def forward(self,x,cache=None,padding=None,):
        """

        :param x:
        :param cache: (kv_latent,k_rope,k_embedding_mask)
        :param padding:
        :return:
        """

        x = x.to(self.kv_down.weight.device)
        b,l,d = x.shape


        past_len = cache[0].size(1) if cache is not None else 0
        total_len = past_len + l



        if padding is None:
            padding = torch.ones((b,l),device=x.device,dtype=torch.bool)
        else:
            padding = padding.to(device=x.device)
        x = x*padding.unsqueeze(-1)



        #kv_down
        kv_latent_now = self.kv_down(x)
        kv_latent_now = self.kv_norm(kv_latent_now)



        #q_down
        q_rope = self.q_r(x)
        q_latent = self.q_down(x)
        q_latent = self.q_norm(q_latent)
        q_rope = self.rope.rotate_queries_or_keys(q_rope, offset=past_len, seq_dim=-2)



        #k_r
        k_rope_now = self.k_r(x)
        k_rope_now = self.rope.rotate_queries_or_keys(k_rope_now, offset=past_len, seq_dim=-2)



        if cache is not None:
            past_kv_latent = cache[0]
            kv_latent = torch.cat([past_kv_latent, kv_latent_now], dim=1)
            k_rope = torch.cat([k_rope_now,cache[1]],dim=1)
            key_padding = torch.cat([padding,cache[2]],dim=1)
        else:
            past_kv_latent = None
            kv_latent = kv_latent_now
            k_rope = k_rope_now
            key_padding = padding



        #q_up
        q_nope = self.q_up(q_latent)
        q_nope = q_nope.view(b,total_len,self.n_head,self.qk_nope)
        q_rope = q_rope.unsqueeze(2).expand(-1,-1,self.n_head,-1)
        q = torch.cat([q_nope,q_rope],dim=-1)



        #k_up
        k_nope = self.k_up(kv_latent)
        k_nope = k_nope.view(b,total_len,self.n_head,self.qk_nope)
        k_rope = k_rope.unsqueeze(2).expand(-1,-1,self.n_head,-1)
        k = torch.cat([k_nope,k_rope],dim=-1)


        #v_up
        v = self.v_up(kv_latent).view(b,total_len,self.n_head,self.v_head_dim)


        #attention_mask
        attention_bool = self.create_attention_mask(total_len,past_len,key_padding)
        attention_bool = attention_bool.to(x.device,x.dtype)
        attention_float = torch.zeros_like(attention_bool)
        attention_float.masked_fill(attention_bool, -torch.inf)

        indices = self.indexer(x, past_kv_latent, attention_float) #b l k
        indices = indices.long()  #b l k
        batch_ids = torch.arange(b)[:,None,None]  #b 1 1


        #取出被选择位置对应的 causal/padding mask
        selected_k = k[batch_ids, indices]
        selected_v = v[batch_ids,indices]
        allowed = attention_bool.squeeze(1) #b l total_len
        selected_allow = allowed.gather(1,indices) #b l k


        atten_score = (q@selected_k.transpose(-1,-2))/(self.qk_head_dim**0.5)
        atten_score = atten_score.masked_fill(~selected_allow.unsqueeze(2), 0.0)
        attn_prob = atten_score/atten_score.sum(dim=-1,keepdim=True).clamp_min(1e-9)
        attn_prob = torch.dropout(attn_prob, p=self.p_dropout,train=self.training)


        output = torch.einsum("blhk,blkhv->blhv",attn_prob,selected_v)
        output = output.contiguous().view(b,l,self.n_head*self.v_head_dim)
        output = self.W_o(output)
        output = output*padding.unsqueeze(-1)
        next_cache = (kv_latent_now,k_rope_now,key_padding)
        return output, next_cache


    def create_attention_mask(self,total_len,past_len,key_padding):
        l = total_len - past_len
        q_pos = torch.arange(past_len,total_len,device=key_padding.device).unsqueeze(-1)
        k_pos = torch.arange(total_len,device=key_padding.device).unsqueeze(0)
        casual_bool = k_pos <= q_pos
        attention_bool = casual_bool.view(1,1,l,total_len) & key_padding.view(-1,1,1,total_len)
        return attention_bool

    @staticmethod
    def init(module:nn.Module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            torch.nn.init.trunc_normal_(module.weight,std=0.02)
            if isinstance(module,nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module,nn.RMSNorm):
            nn.init.ones_(module.weight)

            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)




if __name__ == "__main__":
    x = torch.randn((1,2,1024))
    cfg = DSAConfig()
    mla = MLAWithDSA(cfg=cfg).cuda()
    x,cache = mla(x).cuda()
    print(x.shape)
    print(cache[1].shape)
    print(cache[0].shape)