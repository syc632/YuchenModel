# YuchenModel

一款从0到1训练的超轻量语言模型。

项目主要实现:
1.KDA及其高效硬件算法  

2.Stable Latent MoE  

3.SiTU_GLU  

4.Block Attention Residual  

5.Embedding_gating_MLA  


## 核心架构

### KDA

主要思路包括：

- 使用因果短卷积对局部 token 信息进行混合；
- 使用门控衰减控制历史状态的遗忘速度；
- 通过 Delta Rule 更新递推状态；
- 使用 chunk 计算处理训练序列，并提供 recurrent 路径支持逐 token 推理；
- 为流式生成保留卷积状态和递推状态，支持缓存复用。

### MLA

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

AttnRes 用于跨层残差融合。模型会保存一个周期内的历史 hidden states，并通过可学习查询向量计算各层表示的权重，再进行加权求和。

默认配置中，KDA 与 MLA 按周期(3:1)交替使用；每个周期结束后可以执行一次 AttnRes 融合。这样可以在保留线性递推混合效率的同时，引入更强的全局信息交互能力。

## 训练与测试

主要训练入口如下：

| 文件 | 用途 |
| --- | --- |
| `tokenizer/train_tokenizer.py` | 从中文文本 JSONL 数据训练 BPE tokenizer |
| `train/pretrain.py` | 因果语言模型预训练 |
| `train/SFT.py` | 监督微调（SFT） |
| `train/Lora.py` | LoRA 参数高效微调 |
| `train/train_util.py` | checkpoint、参数统计和训练辅助逻辑 |

测试目录覆盖以下内容：

- MLA 和 GDN 的完整序列/增量缓存一致性；
- 大数值输入下的有限值和梯度稳定性；
- Stable Latent MoE 的 Top-k 路由、padding、负载均衡损失和推理路径；
- AttnRes 的加权结果和反向传播；
- 模型输出、损失、梯度和参数量检查。

## 项目结构

```text
MiniModel/
├── model/                   # 模型组件
│   ├── GatedDeltaNet.py     # GDN / GatedDeltaNet
│   ├── MLA.py               # Multi-head Latent Attention
│   ├── Stable_Latent_Moe.py # Stable Latent MoE
│   ├── ffn.py               # FFN 与 SwiGLU/SiTUGLU
│   └── model.py             # 主干模型、AttnRes 和因果语言模型封装
├── tokenizer/               # tokenizer 训练脚本
├── train/                   # 预训练、SFT、LoRA 和训练工具
├── test/                    # 模块契约、稳定性和缓存测试
├── data/                    # 训练和微调数据
├── BPEmodel/                # 已生成的 tokenizer 文件
├── compare_mla_gated_deltanet.py  # MLA/GDN 对比实验
├── comparison_results.json  # 对比实验结果
└── loss-landscape/          # loss landscape 实验工具
```


