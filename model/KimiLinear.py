import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as f



@dataclass
class Config:
    d_model: int = 512
    n_head: int = 8
    head_dim: int = 128
    conv_size: int = 4
    chunk_size: int = 64
    gate_rank: int | None = None
    n_layer: int = 4
    kda_ratio: int = 3
    expan: int = 4
    kv_latent: int = 64
    q_latent: int = 64
    dropout: float = 0.0
    norm_eps: float = 1e-5


class KimiDeltaAttention(nn.Module):
    """Kimi Linear论文中的Kimi Delta Attention(KDA)。"""

    def __init__(self,d_model,n_head,head_dim=None,conv_size=4,gate_rank=None,norm_eps=1e-5,chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = head_dim if head_dim is not None else d_model//n_head
        self.conv_size = conv_size
        self.chunk_size = chunk_size
        self.proj_dim = n_head*self.head_dim
        self.gate_rank = gate_rank if gate_rank is not None else self.head_dim

        assert conv_size >= 2
        assert chunk_size >= 1
        assert self.head_dim > 0
        assert self.gate_rank > 0

        #论文中q,k,v各自先投影,再经过一个短因果卷积和Swish
        self.q_proj = nn.Linear(d_model,self.proj_dim,bias=False)
        self.k_proj = nn.Linear(d_model,self.proj_dim,bias=False)
        self.v_proj = nn.Linear(d_model,self.proj_dim,bias=False)
        self.conv_q = nn.Conv1d(self.proj_dim,self.proj_dim,conv_size,groups=self.proj_dim,bias=False)
        self.conv_k = nn.Conv1d(self.proj_dim,self.proj_dim,conv_size,groups=self.proj_dim,bias=False)
        self.conv_v = nn.Conv1d(self.proj_dim,self.proj_dim,conv_size,groups=self.proj_dim,bias=False)

        #逐通道遗忘门,先降维再升维,rank在论文中等于head_dim
        self.alpha_down = nn.Linear(d_model,self.gate_rank,bias=False)
        self.alpha_up = nn.Linear(self.gate_rank,self.proj_dim,bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(n_head).uniform_(1,16)))

        #dt_bias按照官方实现初始化,让初始遗忘速度分布在[0.001,0.1]
        dt = torch.exp(
            torch.rand(self.proj_dim)*(math.log(0.1)-math.log(0.001))+math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt+torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        #beta仍然是每个head一个标量,控制delta rule的写入速度
        self.beta = nn.Linear(d_model,n_head,bias=False)


        #论文的输出门同样使用低秩投影,激活函数为Sigmoid
        self.gate_down = nn.Linear(d_model,self.gate_rank,bias=False)
        self.gate_up = nn.Linear(self.gate_rank,self.proj_dim,bias=False)
        self.out_norm = nn.RMSNorm(self.head_dim,eps=norm_eps)
        self.W_o = nn.Linear(self.proj_dim,d_model,bias=False)

    def causal_short_conv(self,x,conv,state=None):
        """
        短因果卷积的cache只保存前conv_size-1个token。
        x: b l d
        state: b d conv_size-1
        """
        x = x.transpose(1,2) # b d l
        history_len = self.conv_size-1

        if state is None:
            state = x.new_zeros(x.size(0),x.size(1),history_len)

        x_with_history = torch.cat([state,x],dim=-1)
        y = conv(x_with_history)
        next_state = x_with_history[...,-history_len:]
        return y.transpose(1,2),next_state

    def get_alpha(self,x):
        """返回alpha和log(alpha),后者可以直接对应论文中的g。"""
        b,l,_ = x.shape
        gate = self.alpha_up(self.alpha_down(x)).view(b,l,self.n_head,self.head_dim)
        gate = gate.float()+self.dt_bias.view(1,1,self.n_head,self.head_dim)
        log_alpha = -self.A_log.float().exp().view(1,1,self.n_head,1)*f.softplus(gate)
        return log_alpha.exp().to(x.dtype),log_alpha

    def recurrent_kda(self,q,k,v,log_alpha,beta,S):
        """论文公式(1),用于一次生成一个token时的递推。"""
        assert q.size(2)==1
        q = q.float()
        k = k.float()
        v = v.float()
        alpha = log_alpha[:,0].float().exp().unsqueeze(-1)
        beta = beta[:,0].float().view(q.size(0),self.n_head,1,1)

        decayed_S = alpha*S
        v_old = k@decayed_S
        S = decayed_S+k.transpose(-1,-2)@(beta*(v-v_old))
        output = (q@S).transpose(1,2)
        return output,S

    def chunk_kda(self,q,k,v,log_alpha,beta,S):
        """
        论文第3.1节的chunkwise WY/UT算法。
        块内token通过下三角矩阵一次计算,只在不同chunk之间传递状态。
        """
        dtype = v.dtype
        b,n,l,d = q.shape
        chunk_size = min(self.chunk_size,l)
        #当seq_len不能被chunk_size整除的时候去补一些padding
        pad_len = (chunk_size-l%chunk_size)%chunk_size



        #论文伪代码在float32中计算累计衰减和三角矩阵
        #f.pad在tensor的边缘填充数据
        #chunkwise核心是把整条seq_lenreshape成规整的三维块网络(b,n_head,n_chunk,chunk_size,d),所以qkv需要补0
        q = f.pad(q.float(),(0,0,0,pad_len))
        k = f.pad(k.float(),(0,0,0,pad_len))
        v = f.pad(v.float(),(0,0,0,pad_len))
        g = f.pad(log_alpha.transpose(1,2).float(),(0,0,0,pad_len))
        beta = f.pad(beta.transpose(1,2).float(),(0,pad_len))



        #padding后的总长度
        padded_len = l+pad_len
        n_chunk = padded_len//chunk_size
        q = q.reshape(b,n,n_chunk,chunk_size,d)
        k = k.reshape(b,n,n_chunk,chunk_size,d)
        v = v.reshape(b,n,n_chunk,chunk_size,d)
        g = g.reshape(b,n,n_chunk,chunk_size,d).cumsum(dim=-2)
        beta = beta.reshape(b,n,n_chunk,chunk_size)


        #A_kk[i,j]=k_i^T Diag(γi/γj) k_j
        #Gi-Gj ==> exp(Gi-Gj) = γi/γj  b n n_chunk chunk_size chunk_size d
        g_i_minus_g_j = g.unsqueeze(-2)-g.unsqueeze(-3)



        #Σ(γi/γj)*ki*kj,在d维度求和
        A_kk = torch.einsum(
            "bhnijd,bhnjd,bhnid->bhnij",
            g_i_minus_g_j.exp(),k,k,
        )


        # b n n_chunk chunk_size chunk_size
        # diagonal=-1代表严格主对角线以下(不含对角线)为true
        lower_mask = torch.ones(
            chunk_size,chunk_size,dtype=torch.bool,device=q.device
        ).tril(diagonal=-1)



        #torch.eye生成一个单位矩阵,主对角线上的元素为1其余为0
        L = torch.eye(chunk_size,dtype=torch.float32,device=q.device)
        L = L.view(1,1,1,chunk_size,chunk_size)
        L = L+(A_kk*beta.unsqueeze(-1)).masked_fill(~lower_mask,0)



        #M=(I+StrictTril(...))^-1 Diag(beta),对应论文的UT transform
        #diag_embed 把最后一维 chunk_size 扩展成 chunk_size × chunk_size 的方阵，并把 beta 的值铺在主对角线上
        beta_diag = torch.diag_embed(beta)
        M = torch.linalg.solve_triangular(L,beta_diag,upper=False)
        W = M@(g.exp()*k)
        U = M@v



        #块内输出矩阵,只保留当前token能够看到的历史token
        A_qk = torch.einsum(
            "bhnijd,bhnid,bhnjd->bhnij",
            g_i_minus_g_j.exp(),q,k,
        )
        causal_mask = torch.ones(
            chunk_size,chunk_size,dtype=torch.bool,device=q.device
        ).tril()
        A_qk = A_qk.masked_fill(~causal_mask,0)

        output = torch.zeros_like(v)
        for i in range(n_chunk):
            q_i = q[:,:,i]
            k_i = k[:,:,i]
            g_i = g[:,:,i]
            W_i = W[:,:,i]
            U_i = U[:,:,i]

            pseudo_value = U_i-W_i@S
            output[:,:,i] = (q_i*g_i.exp())@S+A_qk[:,:,i]@pseudo_value


            last_g = g_i[:,:,-1:]
            S = S*last_g.squeeze(-2).exp().unsqueeze(-1)
            decayed_k = k_i*(last_g-g_i).exp()
            S = S+decayed_k.transpose(-1,-2)@pseudo_value



        output = output.reshape(b,n,padded_len,d)[:,:,:l]
        output = output.transpose(1,2).to(dtype)
        return output,S

    def forward(self,x,cache=None,padding_mask=None):
        b,l,_ = x.shape

        #1代表有效token,0代表padding
        if padding_mask is None:
            padding_mask = torch.ones(b,l,dtype=torch.bool,device=x.device)
        else:
            padding_mask = padding_mask.to(device=x.device,dtype=torch.bool)
        token_mask = padding_mask.unsqueeze(-1)
        x = x*token_mask

        if cache is None:
            S = torch.zeros(
                b,self.n_head,self.head_dim,self.head_dim,
                dtype=torch.float32,device=x.device,
            )
            q_conv_state = None
            k_conv_state = None
            v_conv_state = None
        else:
            S = cache["state"].float()
            q_conv_state = cache["q_conv_state"]
            k_conv_state = cache["k_conv_state"]
            v_conv_state = cache["v_conv_state"]

        q,q_conv_state = self.causal_short_conv(self.q_proj(x),self.conv_q,q_conv_state)
        k,k_conv_state = self.causal_short_conv(self.k_proj(x),self.conv_k,k_conv_state)
        v,v_conv_state = self.causal_short_conv(self.v_proj(x),self.conv_v,v_conv_state)

        q = q.view(b,l,self.n_head,self.head_dim).transpose(1,2)
        k = k.view(b,l,self.n_head,self.head_dim).transpose(1,2)
        v = v.view(b,l,self.n_head,self.head_dim).transpose(1,2)
        q = f.normalize(f.silu(q),dim=-1)
        k = f.normalize(f.silu(k),dim=-1)
        v = f.silu(v)

        _,log_alpha = self.get_alpha(x) # b l n d_head
        beta = f.sigmoid(self.beta(x)) # b l n

        #padding位置必须是恒等转移,否则会改变后续token的状态
        valid = padding_mask.view(b,1,l,1)
        q = q*valid
        k = k*valid
        v = v*valid
        log_alpha = log_alpha*padding_mask.view(b,l,1,1)
        beta = beta*padding_mask.unsqueeze(-1)

        #prefill走论文的chunkwise算法,单token生成走recurrent算法
        if l==1:
            output,S = self.recurrent_kda(q,k,v,log_alpha,beta,S)
        else:
            output,S = self.chunk_kda(q,k,v,log_alpha,beta,S)
        gate = f.sigmoid(
            self.gate_up(self.gate_down(x)).view(b,l,self.n_head,self.head_dim)
        )
        output = self.out_norm(output)*gate
        output = output.reshape(b,l,self.proj_dim)
        output = self.W_o(output)*token_mask

        cache = {
            "state":S,
            "q_conv_state":q_conv_state,
            "k_conv_state":k_conv_state,
            "v_conv_state":v_conv_state,
        }
        return output,cache


class NoPEMLA(nn.Module):
    """Kimi Linear混合架构中不使用位置编码的全局MLA。"""

    def __init__(self,d_model,n_head,head_dim,kv_latent,q_latent,dropout=0.0,norm_eps=1e-5):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.dropout = dropout

        self.kv_down = nn.Linear(d_model,kv_latent,bias=False)
        self.kv_norm = nn.RMSNorm(kv_latent,eps=norm_eps)
        self.k_up = nn.Linear(kv_latent,n_head*head_dim,bias=False)
        self.v_up = nn.Linear(kv_latent,n_head*head_dim,bias=False)

        self.q_down = nn.Linear(d_model,q_latent,bias=False)
        self.q_norm = nn.RMSNorm(q_latent,eps=norm_eps)
        self.q_up = nn.Linear(q_latent,n_head*head_dim,bias=False)
        self.W_o = nn.Linear(n_head*head_dim,d_model,bias=False)

    def forward(self,x,cache=None,padding_mask=None):
        b,l,_ = x.shape
        if padding_mask is None:
            padding_mask = torch.ones(b,l,dtype=torch.bool,device=x.device)
        else:
            padding_mask = padding_mask.to(device=x.device,dtype=torch.bool)
        x = x*padding_mask.unsqueeze(-1)

        kv_now = self.kv_norm(self.kv_down(x))
        if cache is None:
            kv = kv_now
            key_mask = padding_mask
            past_len = 0
        else:
            kv = torch.cat([cache["kv_latent"],kv_now],dim=1)
            key_mask = torch.cat([cache["padding_mask"],padding_mask],dim=1)
            past_len = cache["kv_latent"].size(1)
        total_len = kv.size(1)

        q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.view(b,l,self.n_head,self.head_dim).transpose(1,2)
        k = self.k_up(kv).view(b,total_len,self.n_head,self.head_dim).transpose(1,2)
        v = self.v_up(kv).view(b,total_len,self.n_head,self.head_dim).transpose(1,2)

        q_pos = torch.arange(past_len,total_len,device=x.device).view(1,1,l,1)
        k_pos = torch.arange(total_len,device=x.device).view(1,1,1,total_len)
        causal_mask = k_pos<=q_pos
        attention_mask = causal_mask&key_mask.view(b,1,1,total_len)
        dropout_p = self.dropout if self.training else 0.0
        output = f.scaled_dot_product_attention(
            q,k,v,attn_mask=attention_mask,dropout_p=dropout_p,is_causal=False
        )
        output = output.transpose(1,2).contiguous().view(b,l,-1)
        output = self.W_o(output)*padding_mask.unsqueeze(-1)
        cache = {"kv_latent":kv,"padding_mask":key_mask}
        return output,cache


class SwiGlu(nn.Module):
    def __init__(self,d_model,expan=4):
        super().__init__()
        self.W_up = nn.Linear(d_model,expan*d_model,bias=False)
        self.gate = nn.Linear(d_model,expan*d_model,bias=False)
        self.W_down = nn.Linear(expan*d_model,d_model,bias=False)

    def forward(self,x):
        return self.W_down(f.silu(self.gate(x))*self.W_up(x))


class KimiLinearLayer(nn.Module):
    def __init__(self,cfg,use_mla=False):
        super().__init__()
        self.use_mla = use_mla
        self.in_norm = nn.RMSNorm(cfg.d_model,eps=cfg.norm_eps)
        self.ffn_norm = nn.RMSNorm(cfg.d_model,eps=cfg.norm_eps)

        if use_mla:
            self.mixer = NoPEMLA(
                cfg.d_model,cfg.n_head,cfg.head_dim,cfg.kv_latent,
                cfg.q_latent,cfg.dropout,cfg.norm_eps
            )
        else:
            self.mixer = KimiDeltaAttention(
                cfg.d_model,cfg.n_head,cfg.head_dim,cfg.conv_size,
                cfg.gate_rank,cfg.norm_eps,cfg.chunk_size
            )
        self.swiglu = SwiGlu(cfg.d_model,cfg.expan)

    def forward(self,x,cache=None,padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
        else:
            padding_mask = padding_mask.to(device=x.device,dtype=torch.bool)
        token_mask = padding_mask.unsqueeze(-1)

        residual = x
        x,cache = self.mixer(self.in_norm(x),cache,padding_mask)
        x = (x+residual)*token_mask

        residual = x
        x = self.swiglu(self.ffn_norm(x))
        x = (x+residual)*token_mask
        return x,cache


class KDALayer(KimiLinearLayer):
    """和本地DeltaLayer相同用法的纯KDA层。"""

    def __init__(self,d_model,n_head,expan=4,conv_size=4,head_dim=None,gate_rank=None,norm_eps=1e-5,chunk_size=64):
        head_dim = head_dim if head_dim is not None else d_model//n_head
        cfg = Config(
            d_model=d_model,
            n_head=n_head,
            head_dim=head_dim,
            conv_size=conv_size,
            chunk_size=chunk_size,
            gate_rank=gate_rank,
            n_layer=1,
            expan=expan,
            norm_eps=norm_eps,
        )
        super().__init__(cfg,use_mla=False)
        self.kda = self.mixer


class KimiLinear(nn.Module):
    """按照3个KDA层加1个NoPE MLA层的比例组成Kimi Linear骨干网络。"""

    def __init__(self,cfg=None):
        super().__init__()
        if cfg is None:
            cfg = Config()
        if cfg.kda_ratio < 1:
            raise ValueError("kda_ratio必须大于等于1")
        self.cfg = cfg
        cycle = cfg.kda_ratio+1
        self.layers = nn.ModuleList(
            KimiLinearLayer(cfg,use_mla=(i+1)%cycle==0)
            for i in range(cfg.n_layer)
        )
        self.out_norm = nn.RMSNorm(cfg.d_model,eps=cfg.norm_eps)

    def forward(self,x,caches=None,padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
        else:
            padding_mask = padding_mask.to(device=x.device,dtype=torch.bool)
        if caches is None:
            caches = [None]*len(self.layers)
        if len(caches)!=len(self.layers):
            raise ValueError("cache数量必须和层数相同")

        next_caches = []
        for layer,cache in zip(self.layers,caches):
            x,cache = layer(x,cache,padding_mask)
            next_caches.append(cache)
        x = self.out_norm(x)*padding_mask.unsqueeze(-1)
        return x,next_caches


if __name__ == "__main__":
    cfg = Config(d_model=128,n_head=4,head_dim=32,gate_rank=32,n_layer=4)
    x = torch.randn(1,32,cfg.d_model)
    padding_mask = torch.ones(1,32,dtype=torch.bool)
    model = KimiLinear(cfg)
    y,cache = model(x,padding_mask=padding_mask)
    print(y.shape)
    print(["MLA" if layer.use_mla else "KDA" for layer in model.layers])
    print(cache[0]["state"].shape)
