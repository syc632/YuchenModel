import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from typing import Optional

from .Moe import MoE
from .MLA import MLA
from .GatedDeltaNet import GatedDeltaNet
from.ffn import SwiGlu,SiTUGLU


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
    dropout: float = 0.0
    norm_eps: float = 1e-5
    d_head: int | None = None

    #mla
    #qk不需要做rope的维度
    qk_nope:int = 48
    #qk需要做rope的维度
    qk_rope:int = 16
    v_head_dim:int = 64
    qk_head_dim:int = 64
    kv_latent: int = 64
    q_latent: int = 64

    #moe
    # 共享专家数量
    n_shared_expert: int = 2
    # 每个token选择的路由专家
    n_expert_per_token: int = 2
    # 路由专家的数量
    n_route_expert: int = 8
    aux_loss_alpha: float = 0.01
    use_moe: bool = True
    d_latent:int = 128
    d_inner:int = 768

    use_attn_res:bool = True
    vocab_size:int = 6400
    embd:int = 512
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    #Hook,用于保证传参正确
    def __post_init__(self):
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model必须能被n_head整除")
        if not 1 <= self.n_expert_per_token <= self.n_route_expert:
            raise ValueError("n_expert_per_token必须在[1,n_route_expert]范围内")
        if self.d_head is None:
            self.d_head = self.d_model // self.n_head
        #embedding维度需要和主干隐藏维度保持一致
        self.embd = self.d_model


@dataclass
class CausalLMOutputWithAuxLoss(CausalLMOutputWithPast):
    aux_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None


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
    q = linear.weight.view(-1) #d
    #softmax计算权重
    logits = torch.einsum("d,nbld -> nbl", q, k)
    weight = torch.softmax(logits, dim=0)
    #加权求和
    return torch.einsum("nbl,nbld -> bld", weight, v)


class ModelLayer(nn.Module):
    def __init__(self,cfg,mla=False):
        super().__init__()
        self.in_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.use_moe = cfg.use_moe
        if mla:
            self.mixer = MLA(cfg)
        else:
            self.mixer = GatedDeltaNet(cfg)
        if cfg.use_moe:
            self.ffn = MoE(cfg)
        else:
            self.ffn = SwiGlu(cfg)

    def forward(self,x,cache=None,padding_mask=None):
        b,l,d = x.shape
        if padding_mask is None:
            padding_mask = torch.ones((b,l),dtype=torch.bool,device=x.device)
        else:
            padding_mask = padding_mask.to(device = x.device,dtype = bool)
        token_mask = padding_mask.unsqueeze(-1)
        x = self.in_norm(x)
        x,cache = self.mixer(
            x,
            cache=cache,
            padding_mask=padding_mask,
        )
        x = x*token_mask
        x = self.ffn_norm(x)
        if self.use_moe:
            x,aux_loss = self.ffn(x,padding_mask)
        else:
            x = self.ffn(x)
            aux_loss = x.new_zeros(())
        return x,cache,aux_loss


class YuchenModel(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()

        #attn_res
        self.attn_res = cfg.use_attn_res
        self.q = nn.Linear(cfg.d_model,1,bias=False)
        self.res_norm = nn.RMSNorm(cfg.d_model)

        #混合周期
        self.cycle = cfg.ratio+1
        self.layers = nn.ModuleList(
            [ModelLayer(cfg,mla=(i+1)%self.cycle==0) for i in range(cfg.n_layer)]
        )
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)

    def forward(self,x,cache=None,padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
        else:
            padding_mask = padding_mask.to(device=x.device,dtype=torch.bool)

        #记录Moe的aux_loss的总损失
        #x.new_zeros创建一个零维张量,继承x的dtype和device
        total_aux_loss = x.new_zeros(())
        moe_layers = 0

        #为每一层准备一个cache槽位
        if cache is None:
            cache = [None]*len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("cache数量必须和模型层数一致")
        next_cache = []
        if not self.attn_res:
            #把第i层和第i个cache配对
            for layer,layer_cache in zip(self.layers,cache):
                x,layer_cache,aux_loss = layer(x,layer_cache,padding_mask)
                next_cache.append(layer_cache)
                if layer.use_moe:
                    moe_layers += 1
                    total_aux_loss = total_aux_loss + aux_loss
            if moe_layers>0:
                total_aux_loss = total_aux_loss / moe_layers
            x = self.norm(x)*padding_mask.unsqueeze(-1)
            return x,next_cache,total_aux_loss

        #使用attn_res,每个周期使用一次attn_res
        else:
            block = [x]
            for layer,layer_cache in zip(self.layers,cache):
                x,layer_cache,aux_loss = layer(x,layer_cache,padding_mask)
                if layer.use_moe:
                    total_aux_loss = total_aux_loss + aux_loss
                    moe_layers += 1
                block.append(x)
                if len(block) % self.cycle == 0:
                    x = attn_res(block,x,self.q,self.res_norm,self.q)
                    block.append(x)
                next_cache.append(layer_cache)
            if moe_layers>0:
                total_aux_loss = total_aux_loss/moe_layers
            x = self.norm(x)*padding_mask.unsqueeze(-1)
            return x,next_cache,total_aux_loss


class YuchenModelCausalLLM(nn.Module,GenerationMixin):
    """
    因果语言模型,面向最终任务(文本生成)的顶层封装
    架构组成:
        Token IDs --> YuchenModel --> Hidden State [LM Head] Logits
    关键特性:
        推理优化:支持只计算最后一个Token的Logits,避免全量计算
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg or Config()
        self.model = YuchenModel(self.config)
        self.embd = nn.Embedding(self.config.vocab_size,self.config.d_model)
        self.aux_loss_alpha = self.config.aux_loss_alpha
        #输出头
        self.lm_head = nn.Linear(self.config.d_model,self.config.vocab_size,bias=False)
        #两个矩阵共享参数即可,本身这两个过程就可以看作是一个正向和一个反向的过程
        self.embd.weight = self.lm_head.weight

    def forward(
        self,
        x=None,
        input_ids=None,
        cache=None,
        padding_mask=None,
        attention_mask=None,
        label=None,
        labels=None,
        use_cache=None,
        logits_to_keep:int=0,
    ):
        """
        没写因果掩码是因为MLA内置了
        :param x:b l
        :param cache: 输入序列
        :param label: 标签(训练时使用) b l
        :param padding_mask: 填充
        :param use_cache:KV-cache
        :param logits_to_keep: 在推理时候只计算最后一个Token的Logits,避免全量计算
                0(默认)计算所有token的logits(训练的时候)
                1:计算最后一个token的logits(推理)
        :return:
        """
        del use_cache
        input_ids = input_ids if input_ids is not None else x
        if input_ids is None:
            raise ValueError("必须传入x或input_ids")
        labels = labels if labels is not None else label
        padding_mask = attention_mask if attention_mask is not None else padding_mask
        if padding_mask is None:
            if self.config.pad_token_id is None:
                padding_mask = torch.ones_like(input_ids,dtype=torch.bool)
            else:
                #ne()对input_ids的每个元素逐个判断是否不等于pad_token_id,返回一个input_ids形状的bool张量
                padding_mask = input_ids.ne(self.config.pad_token_id)

        hidden_state,kv_cache,aux_loss = self.model(
            self.embd(input_ids),cache,padding_mask
        )
        slice_indices = slice(-logits_to_keep,None) if logits_to_keep>0 else slice(None)
        sliced_state = hidden_state[:,slice_indices,:]
        logits = self.lm_head(sliced_state)

        #计算损失(仅在训练模式)
        lm_loss = None
        total_loss = None
        if labels is not None:
            if logits_to_keep>0:
                raise ValueError("计算loss时logits_to_keep必须为0")
            #logits需要去掉最后一位,因为最后一位logits是预测E的概率,但是E不存在
            shifted_logits = logits[:,:-1,:].contiguous()
            #label去掉第一位即可
            shifted_label = labels[:,1:].contiguous()
            lm_loss = f.cross_entropy(
                shifted_logits.view(-1,shifted_logits.size(-1)),
                shifted_label.view(-1),
                ignore_index=-100
            )
            total_loss = lm_loss + self.aux_loss_alpha * aux_loss

        #Huggingface定义的一个标准输出容器类,把本次前向的多个结果打包起来
        output = CausalLMOutputWithAuxLoss(
            loss=total_loss,
            logits=logits,
            past_key_values=kv_cache,
            hidden_states=(hidden_state,),
            aux_loss=aux_loss,
            lm_loss=lm_loss,
        )
        return output


if __name__ == "__main__":
    cfg = Config()
    model = YuchenModel(cfg)
    print(model)
