# 训练方法对比说明

本项目在相同数据、相同规模（MiniMind 104M / hidden_size=768 / 8 层）下对比了四种训练方法。

## 1. 方法概述

### 1.1 Pretrain（预训练）
- **目标**：学习顾城语言的分布，下一 token 预测。
- **数据**：7481 条真实语料（诗歌+散文+哲思+小说）。
- **参数**：epochs=5, batch_size=16, lr=5e-4, max_seq_len=512。
- **结果**：loss 3.78。能产出语法通顺的中文，但缺乏诗歌约束与指令跟随能力。

### 1.2 SFT（指令微调）
- **目标**：让模型学会"续写/仿写顾城风格诗歌"的指令行为。
- **数据**：213 条手工构造指令样本。
- **参数**：epochs=10, batch_size=16, lr=5e-4, max_seq_len=512（曾尝试 768 导致卡死，降为 512）。
- **结果**：loss 3.16。AR 模型最终生成质量最佳。

### 1.3 dLM（扩散语言模型）
- **目标**：训练双向 Transformer + MDM（Masked Diffusion Model）目标，推理时迭代去噪（unmask）。
- **迁移**：A2D（AR → dLM）——从已训练 AR 权重迁移初始化，再按 dLM 目标训练。
- **参数**：epochs=5, batch_size=8, lr=1e-4, max_seq_len≤512。
- **结果**：能完成去噪采样流程，但输出过短或句式重复。扩散模型在小数据（7481 条）上难收敛，作为研究对照保留。

### 1.4 Linear（线性注意力）
- **目标**：用 Gated DeltaNet 替换 Softmax Attention，实现线性复杂度推理（理论优势：长序列、省显存）。
- **迁移**：A2L（AR → Linear）——其中 6/8 层为线性注意力（full_attention_interval=4），线性层从随机初始化训练。
- **参数**：epochs=5, batch_size=4（PyTorch fallback 下 batch=16 会 OOM）, lr=5e-4。
- **结果**：loss 6.94 → 3.96，完整 5 epochs 收敛。但生成含乱码/重复，因线性注意力层在小数据上未充分训练。

## 2. 对比总览

| 维度 | Pretrain | SFT | dLM | Linear |
|------|----------|-----|-----|--------|
| 训练方式 | 自回归 | 自回归 | 扩散（mask/unmask） | 自回归 |
| 注意力 | Softmax | Softmax | 双向 Softmax | Gated DeltaNet 线性 |
| 迁移初始化 | 随机 | AR 权重 | AR 权重（A2D） | AR 权重（A2L） |
| 数据量 | 7481 | 213 | 7481 | 213 |
| 显存压力 | 低 | 低 | 中 | 高（PyTorch fallback） |
| 生成质量 | 通顺无约束 | **最佳** | 有限 | 有限 |
| 适用场景 | 基座 | 下游任务 | 研究对照 | 长序列/省显存研究 |

## 3. 结论与建议

1. **同规模同数据下，AR + SFT 效果最好**，是当前可用的顾城诗歌生成器。
2. dLM / Linear 在本小数据场景下未显示优势，更适合大语料/大模型的训练范式，本项目保留为**诚实的研究对照**。
3. 若后续要提升 dLM/Linear 质量：建议增加 10 倍以上语料、延长训练 epoch、或从更大预训练模型迁移。
4. Linear 模型建议在有 `flash-linear-attention` 环境的 GPU 上训练，可避免 PyTorch fallback 的显存压力。
