# MiniMind-GuCheng 生成样本说明（v1.0）

生成日期：2026-08-18，模型：MiniMind 104M（hidden_size=768，8 层）三变体

| 模型 | 样本数 | 平均长度(字) | 生成/过滤次数 | 状态 |
|------|--------|--------------|---------------|------|
| MiniMind-GuCheng-AR | 100 | 87 | 见 report 文件 | 合格 |
| MiniMind-GuCheng-dLM | 100 | 43 | 见 report 文件 | 合格 |
| MiniMind-GuCheng-Linear | 0 | - | 100% 乱码 | 无法生成合格样本 |

## 版本标注
每条样本记录包含字段：model（模型名）、version（版本号 v1.0）、prompt（提示词）、temperature（采样温度）、content（诗正文）、timestamp（生成时间）。

## Linear 说明
Linear（Gated DeltaNet）模型 8 层中 6 层线性注意力从随机初始化训练，在小数据集（213 条 SFT）上未充分收敛，
生成文本 100% 含乱码字符（\\ufffd），无法通过质量过滤（无英文/数字混入、长度>=20、无乱码）。
按质量优先原则不生成 Linear 样本集；详细训练方法对比见 TRAINING_COMPARISON.md。
