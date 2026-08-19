# GuCheng-POEM 顾城风格诗歌生成器（MiniMind 版）

基于 [MiniMind](https://github.com/jingyaogong/minimind) 从零训练的顾城风格诗歌生成器。

## 项目定位

本仓库是 **MiniMind 顾城（GuCheng）诗歌生成器** 的模型权重、数据、脚本与评测报告的**独立发布仓库**。所有顾城相关产物（**不含余华等其他作者**）汇总于此。

上游综合仓库（包含历史与多作者混合内容）：[ChineseHardJudgePoem](https://github.com/shikunpneg/ChineseHardJudgePoem)

## 三变体模型（MiniMind 104M，hidden_size=768，8 层）

> **权重存放**：本仓库**只含数据/样本/脚本/报告**。模型权重（每个 131-145 MB，总约 670 MB）托管在 HuggingFace，详见下表对应链接。GitHub 单文件推送限制 100 MB，权重未直接上传。

| 架构 | HuggingFace | 备注 |
|---|---|---|
| AR（自回归 Softmax Attention） | [MiniMind-GuCheng-AR](https://huggingface.co/shikunpunk/MiniMind-GuCheng-AR) | CoT 版本学会"读题→意象→情感→诗"三段式 |
| dLM（扩散语言模型，A2D + MDM） | [MiniMind-GuCheng-dLM](https://huggingface.co/shikunpunk/MiniMind-GuCheng-dLM) | 全局去噪 + Gumbel top-k 采样 |
| Linear（线性注意力 / Gated DeltaNet） | [MiniMind-GuCheng-Linear](https://huggingface.co/shikunpunk/MiniMind-GuCheng-Linear) | O(1) 常数记忆，原生回退实现 |

## 数据（`data/`）

| 文件 | 内容 |
|---|---|
| `data/pretrain_gucheng.jsonl` | 顾城诗集原始段，预训练用 |
| `data/sft_gucheng.jsonl` | 顾城 SFT 数据 |
| `data/sft_gucheng_cot.jsonl` | 顾城 CoT 数据（607 条），含【读题】/【意象】/【情感基调】/【诗】 |

## 生成样本（`samples/`）

| 文件 | 模型 | 数量 |
|---|---|---|
| `samples/gucheng_samples_ar_cot_full.jsonl` | AR-CoT | 100 条 |
| `samples/gucheng_samples_ar_v1.jsonl` | AR（无 CoT） | 100 条 |
| `samples/gucheng_samples_dllm_v1.jsonl` | dLM | 100 条 |

## 训练脚本（`scripts/`）

- `build_gucheng_cot.py`：CoT 数据构建
- `gen_gucheng_batch.py`：三变体批量生成（同提示 × N 条，含读题错位解码约束）

## 训练流程

```
pretrain（顾城诗集 pretrain_gucheng）
  ├─→ AR SFT（full_sft_gucheng）
  │     └─→ AR CoT SFT（full_sft_gucheng_cot）   ← 最优 AR 版本
  ├─→ dLM SFT（dllm_gucheng）
  └─→ Linear SFT（full_sft_linear_gucheng）
```

## 实验报告

详细实验记录见 [MINIMIND_TRAINING_COMPARISON.md](https://github.com/shikunpneg/ChineseHardJudgePoem/blob/main/doc/MINIMIND_TRAINING_COMPARISON.md)（综合仓库 doc 目录）。

关键发现：
- AR-CoT 通过率 100%，需 `fix_title_drift()` 解码约束修复读题行题目漂移
- dLM 在 104M + 短诗上学会 3 段式结构但生成质量不稳定
- Linear 原生回退实现极慢，但训练后模型可在 SFT 上收敛

## 许可证

MIT