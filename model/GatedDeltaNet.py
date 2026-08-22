import torch
import torch.nn as nn
import torch.nn.functional as f
from dataclasses import dataclass
@dataclass
class Config:
    d_model: int = 4
    n_head: int = 4
    conv_size: int = 4
    chunk_size: int = 64
    d_head: int = 4





class GatedDeltaNet(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.d_head = cfg.d_model // cfg.n_head
        self.conv_size = cfg.conv_size
        self.chunk_size =cfg.chunk_size
        assert cfg.chunk_size >=2
        assert cfg.conv_size >=2
        assert cfg.d_model%cfg.n_head==0


        #qkv短卷积,qkv本身携带的token是独立的qt之来自xt,做一次短卷积让局部的token做一次mixing
        self.conv_q = nn.Conv1d(cfg.d_model,cfg.d_model,cfg.conv_size,padding=0,groups=cfg.d_model,bias=False)
        self.conv_k = nn.Conv1d(cfg.d_model,cfg.d_model,cfg.conv_size,padding=0,groups=cfg.d_model,bias=False)
        self.conv_v = nn.Conv1d(cfg.d_model,cfg.d_model,cfg.conv_size,padding=0,groups=cfg.d_model,bias=False)


        self.qkv = nn.Linear(cfg.d_model,3*cfg.d_model,bias=False)
        self.beta = nn.Linear(cfg.d_model,cfg.n_head,bias=False)
        self.W_o = nn.Linear(cfg.d_model,cfg.d_model,bias=False)
        self.in_norm = nn.RMSNorm(cfg.d_model,eps=1e-5)
        self.out_norm = nn.RMSNorm(cfg.d_model,eps=1e-5)
        #用于控制S的遗忘速度,数值越大,遗忘的越慢,越小越快
        self.gate_logit_normalizer = 16.0
        self.alpha = nn.Linear(cfg.d_model, cfg.n_head, bias=False)
        #输出门控
        self.out_gate = nn.Linear(cfg.d_head,cfg.d_head, bias=False)
        self.out_norm = nn.RMSNorm(cfg.d_head,eps=1e-5)

    def causal_short_conv(self, x, conv, state=None,padding_mask=None):
        """
        此函数用于DeltaNet在流式推理(逐token生成)场景下的实现
        当推理的时候,因果段卷积的计算 :
            qt = W0q_t + W1q_(t-1) + W2q_(t-2) + W3q_(t-3)
            因此需要缓存最近conv_size-1个qkv
            因为卷积核只需要k个token,所以只需要去缓存k-1个
        训练:整条序列一次到位,不需要缓存
        推理:逐token生成,需要缓存
        """
        # b l d
        x = x.transpose(1, 2)  # b d l
        #因为pytorch的卷积所需要的形式就是b d l
        #历史缓存窗口的长度,因为当前token的输出需要看到过去k-1个token
        history_len = self.conv_size - 1 #k-1


        #首次调用的时候state为0
        if state is None:
            state = x.new_zeros(x.size(0), x.size(1), history_len)
        # b d k-1



        #把历史的state和现在的x拼接成新的x,b d k-1+l
        x_with_history = torch.cat([state, x], dim=-1)
        #让新输入的每个token都可以看到过去k-1个token,同时也避免了for循环,提高效率
        y = conv(x_with_history)



        #如果没有padding,那么证明本batch里的最后k-1个token不是0,直接去取历史的token就行
        if padding_mask is None:
            next_state = x_with_history[..., -history_len:]
        #如果传入mask的话就说明右padding
        else:
            #计算有效token的长度,因为规定有效token为1,padding为0,所以相加就会得到有效token的长度
            valid_len = padding_mask.sum(dim=-1)  # [B]

            # 对每个样本取“旧 cache + 有效 token”末尾的 history_len 项
            indices = valid_len[:, None] + torch.arange(
                history_len, device=x.device
            )[None, :]

            indices = indices[:, None, :].expand(-1, x.size(1), -1)


            next_state = x_with_history.gather(-1, indices)

            # padding 位置的卷积输出清零
            y = y * padding_mask[:, None, :]

        return y.transpose(1, 2), next_state  # (B, L, D), (B, D, K-1)


    def recurrent(self,q,k,v,log_alpha,beta,S):
        """
        用于一次生成一个的递推
        """
        assert q.size(2) == 1
        q = q.float()
        k = k.float()
        v = v.float()
        alpha = log_alpha[:,0].float().exp().view(q.size(0),self.n_head,1,1)
        beta = beta[:,0].float().view(q.size(0),self.n_head,1,1)

        decayed_S = alpha*S
        v_old = k@decayed_S
        S = decayed_S + k.transpose(-1, -2)@(beta*(v-v_old))
        output = (q@S).transpose(1,2)
        return output,S



    def chunk(self,chunk_size,log_alpha,beta,q,k,v,S):
        """
        与KDA不同的是门控选择的是粗粒度的,即所有的通道都是用的是一个标量
        WY representation/UT transformation
        块内使用下三角矩阵一次计算,块间使串行计算
        """
        #状态S使用FP32保持精度稳定,输出的时候恢复到刚开始的BF16
        #因为下面q.float会把q转换为FP32,所以这里先保存一下q的dtype
        dtype = q.dtype
        b,n,l,d = q.shape
        chunk_size = min(chunk_size, l)
        #保证seq_len能被chunk_size整除,去补一些padding
        n_padding = (chunk_size-l%chunk_size)%chunk_size
        #在qkv的末尾补n_padding个0
        q = f.pad(q.float(),(0,0,0,n_padding))
        k = f.pad(k.float(),(0,0,0,n_padding))
        v = f.pad(v.float(),(0,0,0,n_padding))
        #qkv的形状已经是b n l d,但是α和β的形状是b l n d
        #b n l+n_padding
        g = f.pad(log_alpha.transpose(1,2),(0,n_padding))
        beta = f.pad(beta.transpose(1,2),(0,n_padding))



        pad_len = l + n_padding
        n_chunk = pad_len//chunk_size
        q = q.reshape(b,n,n_chunk,chunk_size,d)
        k = k.reshape(b,n,n_chunk,chunk_size,d)
        v = v.reshape(b,n,n_chunk,chunk_size,d)
        #粗粒度,每个通道使用相同的门控
        #cumsum对chunk内的token进行累计求和,后续做差得到某两个位置的衰减因子G
        #g是每个chunk内累计的logα
        g = g.reshape(b,n,n_chunk,chunk_size).cumsum(dim=-1)
        #β:第i个token的写入强度
        #b n n_chunk chunk_size
        beta = beta.reshape(b,n,n_chunk,chunk_size)


        #因果掩码
        causal_mask = torch.ones((chunk_size, chunk_size), device=q.device, dtype=torch.bool).tril()



        # b n n_chunk chunk_size,chunk_size d
        # ,衰减因子G,表示第k_j传到k_i的时候经过遗忘系数后的结果
        Gi_Gj = g.unsqueeze(-1) - g.unsqueeze(-2)
        # 当遗忘门的系数较强的时候,从j到i的衰减可能会比较大,导致exp直接报inf,虽然前向传播的时候加掩码会正常传播
        # 但是当反向传播的时候会导致inf梯度,所以需要加掩码(0*inf =NaN)
        log_decay = Gi_Gj.masked_fill(~causal_mask, float("-inf"))
        decay = log_decay.exp()


        #b n n_chunk chunk_size chunk_size
        #表示第chunk内第j个k对第i个k的影响
        #decay*(K^T @ K)
        A_kk = decay*torch.einsum("bhnjd,bhnid->bhnij",  k, k)


        #严格下三角矩阵(bool掩码)
        low_mask = torch.ones((chunk_size,chunk_size),device=q.device,dtype=torch.bool).tril(diagonal=-1)


        #单位矩阵I
        I = torch.eye(chunk_size,dtype=q.dtype,device=q.device)
        I = I.view(1,1,1,chunk_size,chunk_size)
        #L = I + StrictTril(Diag(β)Akk)
        #mask_fill(~low_mask,0)把对角线和上三角部分清零,只保留严格下三角
        #low_mask为true的地方替换为0,但是因为需要的是下三角,所以~取反
        L = I+(A_kk*beta.unsqueeze(-1)).masked_fill(~low_mask,0)



        #UT transform
        #M = L^(-1) Diag(β)
        #b n n_chunk chunk_size chunk_size
        beta_diag = torch.diag_embed(beta)
        M = torch.linalg.solve_triangular(L, beta_diag, upper=False)



        #实际写入量P = Mv - M(T*K)S0 <==>  P = U - WS_0
        #U = Mv 准备写入的新内容 W = M(T*K)需要从旧状态中擦除什么(WS_0)
        W = M@(g.exp().unsqueeze(-1)*k)
        U = M@v

        A_qk = decay*torch.einsum(
            "bhnid,bhnjd->bhnij",
             q, k,
        )

        A_qk = A_qk.masked_fill(~causal_mask, 0)



        output = torch.zeros_like(v)
        #块间还是串行计算
        for i in range(n_chunk):
            #b n chunk_size d
            q_i = q[:, :, i]
            k_i = k[:, :, i]
            g_i = g[:, :, i]
            W_i = W[:, :, i]
            U_i = U[:, :, i]

            pseudo_value = U_i - W_i@S
            #第i个块内token之间的影响
            output[:, :, i] = (
                    (q_i * g_i.exp().unsqueeze(-1)) @ S
                    + A_qk[:, :, i] @ pseudo_value
            )

            #chunk内最后一个token位置处的累计对数衰减值
            #b n 1 d
            last_g = g_i[:,:,-1:]
            #把旧状态S乘上本chunk的完整衰减比
            #b n d_head d_head
            S = S*last_g.exp().unsqueeze(-1)
            #从位置j到位置i所经过的衰减系数
            decayed_k = k_i * (last_g - g_i).exp().unsqueeze(-1)
            #把一个chunk内的增加量加入到S中
            S = S + decayed_k.transpose(-1,-2)@pseudo_value

        output = output.reshape(b, n, pad_len, d)[:, :, :l]
        output = output.transpose(1, 2).to(dtype)
        return output, S



    def forward(self,x,cache=None,padding_mask=None):
        residual = x
        x = self.in_norm(x)
        b,l,d = x.shape
        d_head = self.d_model//self.n_head



        #防止padding污染状态S和卷积计算
        #1为有效token,0为padding
        if padding_mask is None:
            padding_mask = torch.ones(b,l,dtype=torch.bool,device=x.device)
        else:
            padding_mask = padding_mask.to(
                device=x.device,
                dtype=torch.bool,
            )
        token_mask = padding_mask.unsqueeze(-1) # b l 1
        x = x*token_mask



        #read cache
        if cache is None:
            S = x.new_zeros(b, self.n_head, d_head, d_head) # b n d_head d_head
            q_conv_state = None
            k_conv_state = None
            v_conv_state = None
        else:
            S = cache["state"]
            q_conv_state = cache["q_conv_state"]
            k_conv_state = cache["k_conv_state"]
            v_conv_state = cache["v_conv_state"]



        #β
        beta = f.sigmoid(self.beta(x))   # b l n

        #α
        log_alpha = (
                f.logsigmoid(self.alpha(x).float())
                / self.gate_logit_normalizer
        )



        #qkv
        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3,dim=-1)
        q,q_conv_state = self.causal_short_conv(q, self.conv_q,q_conv_state,padding_mask=padding_mask)
        k,k_conv_state = self.causal_short_conv(k, self.conv_k,k_conv_state,padding_mask=padding_mask)
        v,v_conv_state = self.causal_short_conv(v, self.conv_v,v_conv_state,padding_mask=padding_mask)
        #分头行动
        q = q.view(b,l,self.n_head,d_head).transpose(1,2) # b n l d
        k = k.view(b,l,self.n_head,d_head).transpose(1,2)
        v = v.view(b,l,self.n_head,d_head).transpose(1,2)
        # L2归一化,分头后归一化可以保证每个头的长度为1
        q = f.normalize(f.silu(q), dim=-1)
        k = f.normalize(f.silu(k), dim=-1)
        v = f.silu(v)


        #log_alpha = 0 exp(log_alpha) = 1,保证State在遇到padding的时候不会衰减
        log_alpha = log_alpha * padding_mask.unsqueeze(-1)
        #beta = 0 ,在遇到padding的时候不会写入State
        beta = beta * padding_mask.unsqueeze(-1)


        #单token使用recurrent,prefill使用chunkwise
        if l ==1:
            output,S = self.recurrent(q,k,v,log_alpha,beta,S)
        else:
            output,S = self.chunk(self.chunk_size,log_alpha,beta,q,k,v,S)


        x = x.view(b,l,self.n_head,d_head)
        gate = f.silu(self.out_gate(x))
        output = self.out_norm(output)
        output = output*gate
        output = output.reshape(b,l,self.d_model)
        output = self.W_o(output)
        cache = {
            "state":S,
            "q_conv_state":q_conv_state,
            "k_conv_state":k_conv_state,
            "v_conv_state":v_conv_state
        }
        output = (output + residual) * token_mask
        return output,cache



