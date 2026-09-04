import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as f


@dataclass
class Mamba2Config:

    d_model: int = 512
    head_dim: int = 128
    conv_size: int = 4
    chunk_size: int = 64
    norm_eps: float = 1e-5
    mamba_d_state: int = 64
    mamba_expand: int = 2
    mamba_n_groups: int = 1


def _segment_sum(x: torch.Tensor) -> torch.Tensor:
    """
    计算一个chunk内任意两个位置之间的累计和。

    x: [..., Q]
    return: [..., Q, Q]，下三角位置(i,j)表示(j,i]之间的累计值。
    """
    chunk_size = x.size(-1)
    # 每一行复制当前位置的A，随后沿行方向做累加，可以避免两个大cumsum直接相减。
    x = x.unsqueeze(-1).expand(*x.shape, chunk_size)
    strict_lower = torch.ones(
        (chunk_size, chunk_size),
        device=x.device,
        dtype=torch.bool,
    ).tril(diagonal=-1)
    x = x.masked_fill(~strict_lower, 0)
    x = torch.cumsum(x, dim=-2)

    lower = torch.ones(
        (chunk_size, chunk_size),
        device=x.device,
        dtype=torch.bool,
    ).tril()
    return x.masked_fill(~lower, float("-inf"))


class Mamba2(nn.Module):
    """
        x: [B, L, D]
        cache: {"conv_state": ..., "ssm_state": ...}
        return: (output, next_cache)

    结构参考官方Mamba2Simple和SSD Listing 1，但使用纯PyTorch实现，
    便于在CPU环境调试以及复用当前模型的padding/cache接口
    """

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.head_dim = cfg.head_dim
        self.d_state = cfg.mamba_d_state
        self.expand = cfg.mamba_expand
        self.n_groups = cfg.mamba_n_groups
        self.conv_size = cfg.conv_size
        self.chunk_size = cfg.chunk_size
        self.norm_eps = cfg.norm_eps

        if self.expand <= 0:
            raise ValueError("mamba_expand必须大于0")
        if self.head_dim <= 0:
            raise ValueError("head_dim必须大于0")
        if self.d_state <= 0:
            raise ValueError("mamba_d_state必须大于0")
        if self.n_groups <= 0:
            raise ValueError("mamba_n_groups必须大于0")
        if self.conv_size < 2:
            raise ValueError("conv_size必须大于等于2")
        if self.chunk_size < 2:
            raise ValueError("chunk_size必须大于等于2")

        self.d_inner = self.expand * self.d_model
        if self.d_inner % self.head_dim != 0:
            raise ValueError("mamba_expand*d_model必须能被head_dim整除")
        self.n_heads = self.d_inner // self.head_dim
        if self.n_heads % self.n_groups != 0:
            raise ValueError("Mamba2的head数量必须能被mamba_n_groups整除")

        # 投影顺序和官方实现相同：[z, x, B, C, dt]。
        # z和x各占d_inner，B/C按group共享，dt为每个head提供一个时间步长。
        projection_dim = (
            2 * self.d_inner
            + 2 * self.n_groups * self.d_state
            + self.n_heads
        )
        self.in_proj = nn.Linear(self.d_model, projection_dim, bias=False)

        self.conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_size,
            groups=self.conv_dim,
            bias=True,
            padding=0,
        )

        # dt初始化在[0.001, 0.1]上对数均匀采样，再通过softplus的逆函数写入bias。
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        inverse_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inverse_dt)

        # A始终以负数参与状态衰减，存储log值可以保证训练过程中-exp(A_log)<0。
        A = torch.empty(self.n_heads, dtype=torch.float32).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        # D是SSM输出中从卷积特征x直接到输出的skip系数，每个head共享一个标量。
        self.D = nn.Parameter(torch.ones(self.n_heads))

        self.out_norm = nn.RMSNorm(self.d_inner, eps=self.norm_eps)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def causal_short_conv(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对[x,B,C]分支执行depthwise因果短卷积，并保存最近K-1个原始输入。

        x: [B, L, conv_dim]
        state: [B, conv_dim, K-1]
        return: 卷积结果[B,L,conv_dim]和下一步卷积缓存。

        当前项目使用右padding，因此缓存只保留每个样本最后一个有效token之前的历史，
        避免prefill末尾的padding在后续逐token生成时进入卷积窗口。
        """
        x = x.transpose(1, 2)  # [B, conv_dim, L]
        history_len = self.conv_size - 1
        if state is None:
            state = x.new_zeros(x.size(0), x.size(1), history_len)

        expected_shape = (x.size(0), x.size(1), history_len)
        if tuple(state.shape) != expected_shape:
            raise ValueError(
                f"conv_state形状应为{expected_shape}，实际为{tuple(state.shape)}"
            )

        x_with_history = torch.cat([state, x], dim=-1)
        output = self.conv1d(x_with_history).transpose(1, 2)
        output = f.silu(output)

        # valid_len=v时，[state,current]中以v为起点的K-1个位置，
        # 正好是读完当前有效前缀后需要保留的最后K-1个token。
        valid_len = padding_mask.sum(dim=-1)
        indices = valid_len[:, None] + torch.arange(
            history_len,
            device=x.device,
        )[None, :]
        indices = indices[:, None, :].expand(-1, x.size(1), -1)
        next_state = x_with_history.gather(-1, indices)

        output = output * padding_mask.unsqueeze(-1)
        return output, next_state

    def ssd(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mamba2的分块SSD算法。

        x: [B, L, H, P]，已经乘上dt
        A: [B, L, H]，已经乘上dt，因此这里存的是离散化前的log衰减
        B/C: [B, L, H, N]
        initial_state: [B, H, P, N]
        return: 输出[B,L,H,P]和最终状态[B,H,P,N]
        """
        batch, seq_len, n_heads, head_dim = x.shape
        if n_heads != self.n_heads or head_dim != self.head_dim:
            raise ValueError("SSD输入的head形状与Mamba2配置不一致")

        # SSD要求序列长度能被chunk_size整除；补出的token令A=0且x/B/C=0，
        # 等价于状态不衰减、不写入，也不会产生输出。
        pad_len = (-seq_len) % self.chunk_size
        if pad_len:
            x = torch.cat(
                [x, x.new_zeros(batch, pad_len, n_heads, head_dim)],
                dim=1,
            )
            A = torch.cat(
                [A, A.new_zeros(batch, pad_len, n_heads)],
                dim=1,
            )
            B = torch.cat(
                [B, B.new_zeros(batch, pad_len, n_heads, self.d_state)],
                dim=1,
            )
            C = torch.cat(
                [C, C.new_zeros(batch, pad_len, n_heads, self.d_state)],
                dim=1,
            )

        total_len = x.size(1)
        n_chunks = total_len // self.chunk_size
        chunk = self.chunk_size

        x = x.reshape(batch, n_chunks, chunk, n_heads, head_dim)
        B = B.reshape(batch, n_chunks, chunk, n_heads, self.d_state)
        C = C.reshape(batch, n_chunks, chunk, n_heads, self.d_state)
        # [B,H,n_chunk,Q]，每个chunk内部对A做累加。
        A = A.reshape(batch, n_chunks, chunk, n_heads).permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. 对角块：一次算出同一个chunk内部所有token之间的因果传播。
        decay = torch.exp(_segment_sum(A))
        y_diag = torch.einsum(
            "bclhn,bcshn,bhcls,bcshp->bclhp",
            C,
            B,
            decay,
            x,
        )

        # 2. 把每个chunk内新增的信息压缩成chunk末尾的状态增量。
        decay_to_end = torch.exp(A_cumsum[..., -1:] - A_cumsum)
        states = torch.einsum(
            "bclhn,bhcl,bclhp->bchpn",
            B,
            decay_to_end,
            x,
        )

        # 3. chunk之间仍然按SSM递推，但这里只有n_chunk个位置。
        states = torch.cat([initial_state.unsqueeze(1), states], dim=1)
        chunk_decay_input = f.pad(A_cumsum[..., -1], (1, 0))
        chunk_decay = torch.exp(_segment_sum(chunk_decay_input))
        boundary_states = torch.einsum(
            "bhzc,bchpn->bzhpn",
            chunk_decay,
            states,
        )
        states, final_state = boundary_states[:, :-1], boundary_states[:, -1]

        # 4. 每个token读取对应chunk起点的历史状态，得到跨chunk输出。
        state_to_output = torch.exp(A_cumsum)
        y_off = torch.einsum(
            "bclhn,bchpn,bhcl->bclhp",
            C,
            states,
            state_to_output,
        )

        output = (y_diag + y_off).reshape(
            batch,
            total_len,
            n_heads,
            head_dim,
        )
        return output[:, :seq_len], final_state

    def recurrent(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单token生成路径，张量形状与ssd方法一致且L必须为1。"""
        if x.size(1) != 1:
            raise ValueError("recurrent只支持一次输入一个token")

        decay = torch.exp(A[:, 0]).unsqueeze(-1).unsqueeze(-1)
        write = torch.einsum("bhp,bhn->bhpn", x[:, 0], B[:, 0])
        state = state * decay + write
        output = torch.einsum("bhpn,bhn->bhp", state, C[:, 0]).unsqueeze(1)
        return output, state

    def forward(self, x, cache=None, padding_mask=None):
        residual = x
        batch, seq_len, _ = x.shape

        if padding_mask is None:
            padding_mask = torch.ones(
                (batch, seq_len),
                device=x.device,
                dtype=torch.bool,
            )
        else:
            padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)
        token_mask = padding_mask.unsqueeze(-1)
        x = x * token_mask

        projected = self.in_proj(x)
        z, xBC, dt = torch.split(
            projected,
            [
                self.d_inner,
                self.conv_dim,
                self.n_heads,
            ],
            dim=-1,
        )

        if cache is None:
            conv_state = None
            ssm_state = x.new_zeros(
                batch,
                self.n_heads,
                self.head_dim,
                self.d_state,
                dtype=torch.float32,
            )
        else:
            if not isinstance(cache, dict) or not {
                "conv_state",
                "ssm_state",
            }.issubset(cache):
                raise ValueError("Mamba2 cache必须包含conv_state和ssm_state")
            conv_state = cache["conv_state"]
            ssm_state = cache["ssm_state"]
            expected_state_shape = (
                batch,
                self.n_heads,
                self.head_dim,
                self.d_state,
            )
            if tuple(ssm_state.shape) != expected_state_shape:
                raise ValueError(
                    f"ssm_state形状应为{expected_state_shape}，实际为{tuple(ssm_state.shape)}"
                )
            ssm_state = ssm_state.float()

        xBC, conv_state = self.causal_short_conv(
            xBC,
            conv_state,
            padding_mask,
        )
        x_ssm, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_groups * self.d_state,
                self.n_groups * self.d_state,
            ],
            dim=-1,
        )

        x_ssm = x_ssm.reshape(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        )
        B = B.reshape(batch, seq_len, self.n_groups, self.d_state)
        C = C.reshape(batch, seq_len, self.n_groups, self.d_state)

        # 同一个group内的多个head共享B和C，这是Mamba2的multi-value SSM结构。
        heads_per_group = self.n_heads // self.n_groups
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)

        # 递推部分统一升到FP32。padding处令A*dt=0、x*dt=0、B/C=0，
        # 因此该位置既不会衰减历史状态，也不会写入新内容。
        dt = f.softplus(dt.float() + self.dt_bias.float())
        A = -torch.exp(self.A_log.float())
        valid = padding_mask.unsqueeze(-1)
        discrete_A = dt * A.view(1, 1, -1)
        discrete_x = x_ssm.float() * dt.unsqueeze(-1)
        discrete_A = discrete_A.masked_fill(~valid, 0)
        discrete_x = discrete_x * valid.unsqueeze(-1)
        B = B.float() * valid.unsqueeze(-1)
        C = C.float() * valid.unsqueeze(-1)

        if seq_len == 1:
            y, ssm_state = self.recurrent(
                discrete_x,
                discrete_A,
                B,
                C,
                ssm_state,
            )
        else:
            y, ssm_state = self.ssd(
                discrete_x,
                discrete_A,
                B,
                C,
                ssm_state,
            )

        # D*x是SSM内部的直接通路；随后按官方norm_before_gate=False的顺序，
        # 先乘silu(z)，再做RMSNorm。
        y = y + self.D.float().view(1, 1, self.n_heads, 1) * x_ssm.float()
        y = y.reshape(batch, seq_len, self.d_inner)
        y = y * f.silu(z.float())
        rms = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        y = y * rms * self.out_norm.weight.float()
        y = y.to(dtype=x.dtype)

        output = self.out_proj(y)
        output = (output + residual) * token_mask
        next_cache = {
            "conv_state": conv_state,
            "ssm_state": ssm_state,
        }
        return output, next_cache


if __name__ == "__main__":
    config = Mamba2Config()
    module = Mamba2(config)
    inputs = torch.randn(1, 8, config.d_model)
    outputs, next_cache = module(inputs)
    print(outputs.shape)
    print({name: value.shape for name, value in next_cache.items()})
