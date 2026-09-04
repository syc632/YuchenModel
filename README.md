# YuchenModel

一款从0到1训练的超轻量语言模型。

项目主要实现:  
一.架构

1.GDN及其分块并行算法

2.Stable Latent MoE  

3.SiTU_GLU  

4.Block Attention Residual  

5.Embedding_gating_MLA  

6.Nope MLA

7.MLA

8.DeepSeekSpareAttention

9.mamba2

二.训练

1.pretrain

2.midtrain:Lora + 全参微调

3.posttrain: OPD + PPO


## 核心架构

### GDN

主要思路包括：

- 使用因果短卷积对局部 token 信息进行混合；
- 使用门控衰减控制历史状态的遗忘速度；
- 通过 Delta Rule 更新递推状态；
- 使用 chunk 计算处理训练序列，并提供 recurrent 路径支持逐 token 推理；
- 为流式生成保留卷积状态和递推状态，支持缓存复用。

### MLA(Rope)

MLA（Multi-head Latent Attention）通过潜空间压缩减少注意力中的表示和缓存开销：

- 将 KV 表示压缩到 latent space，再恢复到多头表示；
- 对 Q 和 KV 使用独立的 latent projection；
- 仅对部分 query/key 维度应用 RoPE；

### Stable Latent MoE

MoE 模块位于前馈网络位置，在 latent space 中完成专家路由：

- 先将 hidden state 投影到较小的 latent dimension；
- 使用 RMSNorm 稳定 latent 表示；
- 同时保留共享专家和 Top-k 路由专家；
- 使用门控网络为每个 token 选择路由专家；
- 使用负载均衡辅助损失，降低专家负载不均的问题；
- 推理阶段提供按专家聚合 token 的执行路径。

### AttnRes



默认配置中，KDA 与 MLA 按周期(3:1)交替使用；每个周期结束后可以执行一次 AttnRes 融合。这样可以在保留线性递推混合效率的同时，引入更强的全局信息交互能力。


## 新增:
- DeepseekSpare Attention

- Mamba2

- Nope

- Embedding—Gate—MLA



测试目录覆盖以下内容：

- MLA 和 GDN 的完整序列/增量缓存一致性；
- 大数值输入下的有限值和梯度稳定性；
- Stable Latent MoE 的 Top-k 路由、padding、负载均衡损失和推理路径；
- AttnRes 的加权结果和反向传播；
- 模型输出、损失、梯度和参数量检查。




## 待办事项

1.更新笔记

2.更换Aux_loss为QB

3.对比横向对比各个组件

4.消融实验

