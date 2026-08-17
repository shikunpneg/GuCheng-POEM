# GuCheng-POEM 顾城风格诗歌生成器（MiniMind 版）

基于 [MiniMind](https://github.com/jingyaogong/minimind) 从零训练的顾城风格诗歌生成器。

## 项目定位

本仓库用于 **MiniMind 顾城诗歌生成器** 的模型训练、评测与发布：

- 原始自回归模型（MiniMind AR）
- 扩散语言模型（MiniMind-dLM，见 Discussion 618）
- 线性注意力模型（MiniMind-Linear / Gated DeltaNet，见 Discussion 704）

## 数据

顾城诗集原始数据与多版本训练数据位于主项目仓库：

- **GitHub**: https://github.com/shikunpneg/ChineseHardJudgePoem
- **HuggingFace**: https://huggingface.co/shikunpunk/gucheng-poetry-dataset
- **16 区云盘**: `/matpilot/datasets/gucheng_poem_data_20260817/`

## 模型发布（HuggingFace）

| 模型 | 说明 |
|---|---|
| `shikunpunk/Qwen3.8-27B-GuCheng-KTO` | 顾城 KTO 后训练 |
| `shikunpunk/Qwen3.8-27B-GuCheng` | 顾城 SFT adapter |

MiniMind 三变体模型训练完成后将在此仓库发布。

## 训练方法

- Pretrain → SFT → 三变体（AR / dLM / Linear）分叉训练
- 配置：hidden_size=768, 8 层（~64M）
- 训练代码与脚本见 MiniMind 上游仓库

## 许可证

MIT
