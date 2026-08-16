import torch
import torch.nn as nn
from MLA import MLA
from DeltaNet import DeltaLayer,SwiGlu
from dataclasses import dataclass
@dataclass
class Config:
    d_model: int = 512
    n_head: int = 8
    head_dim: int = 128

    #deltanet
    conv_size: int = 4
    chunk_size: int = 64
    gate_rank: int | None = None
    n_layer: int = 12
    ratio: int = 3  #deltanet层和mla层的比例
    expan: int = 4
    dropout: float = 0.0
    norm_eps: float = 1e-5


    #mla
    qk_nope = 48
    qk_rope = 16
    v_head_dim = 64
    qk_head_dim = 64
    kv_latent: int = 64
    q_latent: int = 64

    use_attn_res = True


def attn_res(block:list,partial_block,k,norm:nn.RMSNorm,linear:nn.Linear):
    """
    attn_res就是对模型的层和层之间去做attention计算,然后对层进行加权求和
    :param block: 前面的n层输出的特征向量
    :param partial_block: 当前层内的子层输出的特征向量
    :param k:
    :param norm: RMS归一化
    :param linear: 伪查询向量,形状为 d 1
    :return:
    """
    source = block if partial_block is None else [*block,partial_block] # n+1 b l d
    v = torch.stack(source,dim=0)
    k = norm(v) #
    q = linear.weight.squeeze(-1) #d
    #softmax计算权重
    logits = torch.einsum("d,nbld -> nbl", q, k)
    weight = torch.softmax(logits, dim=-1)
    #加权求和
    return torch.einsum("nbl,nbld -> d", weight, v)

class ModelLayer(nn.Module):
    def __init__(self,cfg,mla=False):
        super().__init__()
        self.in_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        if mla:
            self.mixer = MLA(cfg)
        else:
            self.mixer = DeltaLayer(cfg)
        self.swiglu = SwiGlu(cfg.d_model,cfg.expan)
    def forward(self,x,cache=None,padding_mask=None):
        b,l,d = x.shape
        residual = x
        if padding_mask is None:
            padding_mask = torch.ones((b,l),dtype=x.dtype,device=x.device)
        else:
            padding_mask = padding_mask.to(device = x.device,dtype = bool)
        token_mask = padding_mask.unsqueeze(-1)
        x = self.in_norm(x)
        x,cache = self.mixer(x,cache,padding_mask)
        x = (x+residual)*token_mask
        residual = x
        x = self.ffn_norm(x)
        x = self.swiglu(x)
        x = (x+residual)*token_mask
        return x,cache
class YuchenModel(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()


        #attn_res
        self.attn_res = cfg.use_attn_res
        self.q = nn.Linear(cfg.d_model,1)
        self.res_norm = nn.RMSNorm(cfg.d_model)



        #混合周期
        self.cycle = cfg.ratio+1
        self.layers = nn.ModuleList(
            [ModelLayer(cfg,mla=(i+1)%cfg.ratio==0) for i in range(cfg.n_layer)]
        )
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
    def forward(self,x,cache=None,padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
        else:
            padding_mask = padding_mask.to(device=x.device,dtype=torch.bool)
        #为每一层准备一个cache槽位
        if cache is None:
            cache = [None]*len(self.layers)
        next_cache = []
        if not self.attn_res:
            #把第i层和第i个cache配对
            for layer,cache in zip(self.layers,cache):
                x,cache = layer(x,cache,padding_mask)
                next_cache.append(cache)
            x = self.norm(x)*padding_mask.unsqueeze(-1)
            return x,next_cache
        else:
            block = [x]
            for layer,cache in zip(self.layers,cache):
                x,cache = layer(x,cache,padding_mask)
                block.append(x)
                if len(block) == self.cycle:
                    x = attn_res(block,x,self.q,self.res_norm,self.q)
                    block.append(x)
                next_cache.append(cache)
            x = self.norm(x)*padding_mask.unsqueeze(-1)
            return x,next_cache

if __name__ == "__main__":
    cfg = Config()
    model = YuchenModel(cfg)
    print(model)
