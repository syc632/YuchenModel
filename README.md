# MiniModel

一个基于 PyTorch 的轻量级中文因果语言模型实验仓库，重点探索高效序列建模、低秩潜空间表示和稀疏专家结构的组合方式。

当前项目围绕以下架构组件展开：

```text
BPE Tokenizer
      ↓
Embedding
      ↓
GDN / MLA 混合主干
      ↓
Stable Latent MoE
      ↓
AttnRes 跨层残差融合
      ↓
LM Head
```

## 核心架构

### GDN

GDN 模块当前由 `model/GatedDeltaNet.py` 中的 `GatedDeltaNet` 实现，负责高效的序列混合。对比实验和部分测试仍使用 `DeltaRule` 兼容命名。

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
- 支持 KV cache，用于增量生成；
- 为长序列和低缓存开销方向的实验提供基础。

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

默认配置中，GDN 与 MLA 按周期交替使用；每个周期结束后可以执行一次 AttnRes 融合。这样可以在保留线性递推混合效率的同时，引入更强的全局信息交互能力。

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

## 当前实验记录

仓库中包含一个参数匹配的 MLA 与 GDN 对比实验，结果记录在 `comparison_results.json` 中。该实验使用小规模 copy 任务进行 100 steps 的快速验证，主要用于检查实现和比较趋势，不代表正式基准结论。

| 指标 | MLA | GDN |
| --- | ---: | ---: |
| Eval loss | 3.969 | 4.168 |
| Eval perplexity | 52.92 | 64.55 |
| 完整序列推理速度（tokens/s） | 421,025 | 226,161 |
| 增量推理速度（tokens/s） | 5,645 | 6,045 |
| 完整/增量最大绝对误差 | 6.82e-4 | 1.31e-6 |

正式结论仍需要在统一数据、统一训练预算和标准语言模型基准上进一步验证。

## 待办事项

- **后训练**：完善 SFT、LoRA、对话格式、数据清洗和生成质量评估。
- **基准测试**：增加困惑度、生成质量、吞吐、显存、KV cache 和长上下文测试。
- **实验复现**：建立统一配置、固定随机种子、记录硬件/软件环境和训练曲线。
- **工程整理**：清理 checkpoint、W&B 日志和临时实验产物，减少本地路径依赖。
- **推理接口**：增加统一的命令行推理入口或 Python API。
- **依赖与 CI**：补充标准化依赖文件，并让 CI 自动运行核心模块测试。
- **项目文档**：补充模型卡、许可证、引用方式、安全说明和已知限制。
- **架构验证**：继续完善 GDN、MLA、Stable Latent MoE 和 AttnRes 的独立基准与消融实验。

## 项目状态

这是一个持续迭代中的研究型实验项目。当前重点是验证混合序列建模结构和训练稳定性，模型规模、数据配方和基准结果仍在完善中。
