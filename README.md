# YuchenModel

一款使用pytorch手写的从0到1训练的超轻量语言模型。

## 当前预训练配置

默认模型为8层、512维、8个头，按3 GDN + 1 MLA组成两个周期；保留潜空间MoE和AttnRes，专家中间维为704，潜空间维为128。新训练tokenizer的目标词表大小为8192。

8192词表下总参数为47,237,392，可训练参数为47,237,376；共享embedding和输出头只计一次。预训练入口根据实际tokenizer大小构建模型，已有6400词表仍可使用，对应总参数46,319,888。新词表需要重新训练tokenizer，修改配置不会改变已保存的tokenizer。

预训练默认长度512、batch size为2、梯度累积16步、1个epoch，不限制输入样本数量。每次更新最多16,384个输入token（含padding，最后不足的batch除外），有效预测目标数更少。累积梯度按实际有效预测token数归一化，包含尾部不足16批的更新；MoE辅助损失也按微批的有效预测token数加权。学习率3e-4、warmup比例3%、weight decay为0.1、梯度裁剪1.0，关闭compile。

训练主权重保持FP32，CUDA默认通过autocast执行BF16前向；`dtype="float32"`关闭AMP，`dtype="float16"`启用FP16和GradScaler。CPU使用FP32，不支持BF16的GPU需要显式选择FP16或FP32。

验证集默认由`val_ratio=0.01`、`val_seed=2026`从输入样本固定划分，至少留出一条，训练样本不会包含验证索引。也可设置`val_file="data/validation.jsonl"`使用独立文件，此时不从训练文件划分。划分按样本索引进行，语料去重仍需在数据准备阶段完成。每`eval_interval=100`次成功更新及epoch结束评估一次，按有效预测目标加权统计CE和PPL，不包含MoE辅助损失。记录追加到`validation.jsonl`，验证CE改善时更新最佳权重。

在 `train/pre_train/pretrain.py` 的 `TrainConfig` 中填写实际 `project_dir`、`tokenizer_dir` 和 `data_file` 后，从仓库根目录启动：

```bash
python test/report_parameter_count.py
python -m train.pre_train.pretrain
```

默认 `resume=False`，从零初始化；输出到相对启动目录的 `weight/pretrain_gibc`。新配置与旧12层权重不兼容，首次训练请使用空输出目录。参数统计脚本报告默认词表配置，训练入口另外打印实际模型参数量。

默认512维MoE配置产生以下文件：

- `pretrain_weight_512_moe_resume.pth`：用于恢复，保留原始精度的模型参数、优化器、Scaler、训练位置、随机状态和最佳验证CE；每`save_interval=100`次成功更新及epoch结束保存。
- `pretrain_weight_512_moe.pth`：单独导出的FP16推理权重。
- `pretrain_best_512_moe.pth`及`pretrain_best_512_moe_resume.pth`：验证CE最低时的推理权重和完整状态。
- `train_config.json`、`validation.jsonl`：本次训练配置与验证记录。

续训时保持数据、tokenizer和训练配置一致，将`resume=True`并指向原`save_path`。恢复入口与保存统一使用`pretrain_weight`前缀，缺少checkpoint时会明确报错。旧checkpoint若已经压缩为FP16，加载后无法还原此前丢失的精度。

针对性回归检查：`python -m unittest discover -s test -p 'test_pretrain_training.py' -v`。CPU测试覆盖累积梯度、验证统计、checkpoint精度、最佳权重和中断恢复；GPU可用时额外运行CUDA精度检查。

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

3.posttrain: On Policy Distillation


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

NVIDIA团队首次发布于1月,核心做法就是在通信路由之前先经过一个**下投影矩阵**把向量压缩为一个低维的向量,经过路由门控把向量路由到不同的专家上,计算完再经过一个上投影变为正常矩阵
而Kimi团队在新作KimiK3的时候又进一步改动,把专家网络换为了SiTiGLU,并且引入RMS解决数值不稳定的问题(详细见文档)


### AttnRes

Kimi团队3月份新作,把Attention作用于层和层之间,在Full Attention Residual的模式下,当前层的输出都会和前面所有层的输出计算一次注意力,等于是把每个层的隐藏向量当作token处理,这一点非常的想早期RNN和Attention结合的模型,当然如果查看原论文会发现他们的起点就是从RNN出发的

默认配置中，KDA 与 MLA 按周期(3:1)交替使用；每个周期结束后可以执行一次 AttnRes 融合。这样可以在保留线性递推混合效率的同时，引入更强的全局信息交互能力。



测试目录覆盖以下内容：

- MLA 和 GDN 的完整序列/增量缓存一致性；
- 大数值输入下的有限值和梯度稳定性；
- Stable Latent MoE 的 Top-k 路由、padding、负载均衡损失和推理路径；
- AttnRes 的加权结果和反向传播；
- 模型输出、损失、梯度和参数量检查。





## 模块对比实验

八组受控消融、有效 token 预算训练、多种子确认、参数量对齐和报告生成的完整用法见 [实验工具链说明](experiments/README.md)。

```bash
python -m experiments.cli --help
```

模型通过 `attention_type`、`ffn_type`、`expert_ffn_type` 独立选择注意力、稠密 FFN 和专家 FFN。正式实验从零训练；`test/smoke_experiments.py` 仅用于在模拟数据上验证完整流程。
