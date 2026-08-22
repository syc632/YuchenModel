import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from ffn import SwiGlu,SiTUGLU




@dataclass
class MoeConfig:
    #隐藏维度
    d_model:int = 512
    d_latent:int = 128
    d_inner:int = 768
    #共享专家数量
    n_shared_expert:int = 8
    #每个token选择的路由专家
    n_expert_per_token:int = 3
    #路由专家的数量
    n_route_expert:int = 32
    aux_loss_alpha:float = 0.01

    use_moe:bool = True

class MoEGate(nn.Module):
    def __init__(self,cfg:MoeConfig):
        """
        MoE门控网络
        输入:
        HiddenState: b l d
        输出:
        topk_idx: b*l topk
        每个token选中的专家编号
        topk_weight:b*l topk
        对应专家的加权系数
        aux_loss:专家负载均衡辅助损失,标量
        """
        super().__init__()
        self.topk = cfg.n_expert_per_token
        self.n_route = cfg.n_route_expert
        self.alpha = cfg.aux_loss_alpha
        #每个专家都对应一个向量,用来计算与token之间的相似程度
        self.weight = nn.Parameter(
            torch.empty(
                cfg.n_route_expert,
                cfg.d_latent
            )
        )
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameter()
    def reset_parameter(self):
        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
        )
    def forward(self,x:torch.Tensor,padding_mask:torch.Tensor=None):
        b,l,d = x.shape
        #把所有的token放到一个维度中
        tokens = x.reshape(-1,d) # b*l d


        #将token与所有的专家向量进行点积计算
        logits = tokens.float() @ self.weight.float().transpose(0,1) # b*l n_expert
        score = self.softmax(logits)


        #把padding遮盖住
        if padding_mask is None:
            padding_mask = torch.ones((b,l),dtype=torch.bool,device=x.device)

        else:
            padding_mask = padding_mask.to(device=x.device,dtype=torch.bool)
        #因为现在token变为b*l d,所以padding_mask也需要改变形状
        valid_mask = padding_mask.reshape(-1)

        #torch.topk返回前k个最大值和对应的索引
        #topk_ids: b*l k
        #topk_weight: b*l k
        topk_weight,topk_idx = torch.topk(score,dim=-1,k=self.topk)
        #.clamp_min把每个元素的最小值限制在1e-20以上
        norm = topk_weight.sum(dim=-1,keepdim =True).clamp_min(1e-20)



        #归一化,因为softmax归一化的是全部专家,选完topk个专家之后概率总和不为1,需要再次归一化
        topk_weight = topk_weight/norm

        #paddingtoken的路由权重归0
        topk_weight = topk_weight*valid_mask.unsqueeze(-1)



        #计算专家损失负载均衡的辅助损失函数,防止token只分配给少数的几个专家
        #topk本身不可以导,梯度主要通过scores传播
        if self.training and self.alpha>0:
            aux_loss = self.load_balance_loss(
                score = score,
                topk_idx = topk_idx,
                batch_size = b,
                padding_mask = padding_mask,
                seq_len = l
            )


        else:
            aux_loss = score.new_zeros(())
        topk_weight = topk_weight.to(x.dtype)
        return topk_idx,topk_weight,aux_loss
    def load_balance_loss(self,score,topk_idx,batch_size,seq_len,padding_mask):
        """
        计算专家负载均衡的辅助损失函数
        :param score: b*l n_expert
        :param topk_idx: b*l k
        :param batch_size:
        :param seq_len:
        :return:
        """

        #每个批次的专家使用情况,b l*k,内容每行代表一个batch中选择的专家编号
        idx_per_sequence = topk_idx.view(batch_size,-1)

        b,l = padding_mask.shape


        #b*l
        valid_mask = padding_mask.reshape(-1)
        #样本中有效tokn的数量
        valid_count = padding_mask.sum(-1).to(score.dtype)
        #clamp_min(1)把每个小于1的数字替换为1,防止某一个样本全是padding导致valid_count的值为0的情况出现
        valid_count = valid_count.clamp_min(1.0)


        #b*l k,用来标记每个专家的路由槽位是否有效
        selected_mask = valid_mask.unsqueeze(-1).expand_as(topk_idx)


        #创建一个全零张量去统计各专家接收了多少次token
        expert_load = torch.zeros(
            batch_size,
            self.n_route,
            device=score.device
            ,dtype=score.dtype
        )
        #通过scatter来去按专家的编号累加计数,并且是在GPU上并行完成,比双for循环块很多
        #scatter_add等价于
        #for batch_ids in range(batch_size):
        #    for expert_idx in idx_per_squence[batch_ids]:
        #           expert_load[batch_ids,expert_idx] += 1
        #实际负载次数 b n_route
        expert_load.scatter_add_(dim=1,index=topk_idx.reshape(batch_size,-1),src=selected_mask.reshape(batch_size,-1).to(score.dtype))


        #理想情况下每个专家接收的token:
        #valid_token*top_k / n_experts
        excepted_load = (valid_count.unsqueeze(-1)*self.topk/self.n_route)

        #公式中的N*f_i
        expert_load = expert_load/excepted_load.clamp_min(1e-6)
        valid_mask_3d = padding_mask.unsqueeze(-1).to(score.dtype)

        score_3d = score.reshape(b, l, self.n_route)


        #公式中的P_i
        mean_probability = (
                (score_3d * valid_mask_3d).sum(dim=1)
                / valid_count.unsqueeze(-1)
        )

        aux_loss = (expert_load*mean_probability).sum(dim=-1).mean()

        return aux_loss



class MoE(nn.Module):
    def __init__(self,cfg:MoeConfig):
        super().__init__()

        #专家
        self.shared_expert = nn.ModuleList(SiTUGLU(cfg.d_model,cfg.d_inner) for _ in range(cfg.n_shared_expert))
        self.route_expert = nn.ModuleList(SiTUGLU(cfg.d_latent,cfg.d_inner) for _ in range(cfg.n_route_expert))


        #路由
        self.gate = MoEGate(cfg)
        self.top_k = cfg.n_expert_per_token


        #潜空间
        self.W_down = nn.Linear(cfg.d_model,cfg.d_latent,bias=False)
        self.W_up = nn.Linear(cfg.d_latent,cfg.d_model,bias=False)

        #Stable Latent MoE在生维的时候加了一个RMS
        self.up_norm = nn.RMSNorm(cfg.d_latent)


    def forward(self,x,padding_mask=None):
        """
        流程：
        1. 门控网络为每个 token 选择 top-k 个路由专家
        2. 每个 token 被路由到选中的专家处理
        3. 专家输出按权重加权求和
        4. 共享专家处理所有 token 并添加到输出
        """
        #复制一份用于共享专家
        identity = x
        x = self.W_down(x)
        b,l,d = x.shape
        if padding_mask is None:
            padding_mask = torch.ones((b,l),dtype=torch.bool,device=x.device)
        else:
            padding_mask = padding_mask.to(device=x.device,dtype=torch.bool)
        #topK_idx:b*l k
        topk_idx, topk_weight, aux_loss = self.gate(x,padding_mask)
        #b*l d
        x = x.reshape(-1,d)
        #b*l
        valid_padding  = padding_mask.reshape(-1)
        #b*l k
        valid_padding = valid_padding.unsqueeze(-1).expand_as(topk_idx)


        if self.training:
            #因为每个token会经过k个专家,所以需要复制k份(串行处理,不是并行)
            x = x.repeat_interleave(self.top_k,dim=0)
            #有效token由专家输出覆盖,无效的padding由0填充
            y = torch.zeros_like(x,dtype=x.dtype)
            unused_expert_dependency = x.new_zeros(())
            for i,expert in enumerate(self.route_expert):
                mask = (topk_idx == i).reshape(-1)&valid_padding.reshape(-1)#mask是一个bool张量,用于识别每个专家所处理的token
                expert_output = expert(x[mask])
                #某个专家有token处理,则更新输出
                if mask.any():
                    y[mask] = expert_output.to(y.dtype)
                #没有专家处理(mask全是False,空张量,输出也为空张量)
                #不更新输出,但是让专家的所有参数都出现在计算图中,避免反向传播的时候报错
                else:
                    #让未命中的专家参数保留在计算图中，DDP 下得到显式零梯度。
                    unused_expert_dependency = unused_expert_dependency + sum(
                        (parameter.sum() * 0.0 for parameter in expert.parameters()),
                        x.new_zeros(()),
                    )


            #把route_expert的输出结果加权求和(在k维)
            #y: b*l*k d
            #topk_weight: b*l k
            #(b*l k d) * (b*l k 1) .sum ==> (b*l d)
            y = (
                y.reshape(*topk_weight.shape,-1)*topk_weight.unsqueeze(-1)
            ).sum(1)
            y = y + unused_expert_dependency
            y = y.reshape(b,l,d)
            y = self.W_up(self.up_norm(y))
        else:
            #推理时使用优化的推理函数
            y = self.moe_infer(x, topk_idx, topk_weight.view(-1, 1)).view(b,l,d)
            y = self.W_up(self.up_norm(x))
        token_mask = padding_mask.unsqueeze(-1)
        identity = identity * token_mask
        for expert in self.shared_expert:
            y = y+expert(identity)
        return y*token_mask,aux_loss

    @torch.no_grad()
    def moe_infer(self, x, topk_idx, topk_weight):
        """
        x:           [token_num, d_model]
        topk_idx:    [token_num, top_k]
        topk_weight: [token_num, top_k]，或展平后的 [token_num * top_k, 1]
        """
        token_num, d_model = x.shape

        # 每个“token-专家”路由槽位对应的专家编号和权重
        flat_expert_ids = topk_idx.reshape(-1)  # [token_num * top_k]
        flat_weights = topk_weight.reshape(-1)  # [token_num * top_k]



        # 按专家编号排序，让同一专家要处理的 token 连续排列
        slot_ids = flat_expert_ids.argsort()  # 排序后位置对应原始路由槽位
        token_ids = torch.div(
            slot_ids,
            self.top_k,
            rounding_mode="floor",
        )  # 每个路由槽位对应的原 token 编号

        sorted_x = x.index_select(0, token_ids)  # [token_num * top_k, d_model]

        # 统计每个专家要接收多少个 token
        #torch.bincount()统计非负张量中每个整数出现的次数
        tokens_per_expert = torch.bincount(
            flat_expert_ids,
            minlength=len(self.route_expert),
        )

        # 分段调用每个专家；输出顺序与 sorted_x 保持一致
        sorted_outputs = torch.empty_like(sorted_x)
        start = 0

        for expert_id, n_tokens in enumerate(tokens_per_expert.tolist()):
            end = start + n_tokens

            if n_tokens > 0:
                sorted_outputs[start:end] = self.route_expert[expert_id](
                    sorted_x[start:end]
                )

            start = end

        # 将结果按原 token 聚合：同一个 token 的 top-k 专家输出加权求和
        output = x.new_zeros(token_num, d_model)

        sorted_weights = flat_weights.index_select(0, slot_ids).unsqueeze(-1)
        output.index_add_(
            0,
            token_ids,
            sorted_outputs * sorted_weights,
        )

        return output


if __name__ == "__main__":
    x = torch.randn((1,32,512))
    b,l,_ = x.shape
    padding_mask = torch.ones((b,l),dtype=torch.bool)
    moe = MoE(MoeConfig())
    y,aux_loss = moe(x,padding_mask)
    print(y.shape)
