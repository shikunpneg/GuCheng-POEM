# GuCheng-POEM 顾城风格诗歌生成器

基于 Qwen3.8-27B / MiniMind 的顾城风格诗歌生成器项目。

## 数据集（data/）

- `gucheng_train.jsonl`: 顾城诗集 183 首（ShareGPT 格式：system=顾城人设, human=创作请求, gpt=真实诗）
- `gucheng_train_annotated.jsonl`: 顾城诗标注补充
- `haizi_train.jsonl` / `haizi_cn_train.jsonl`: 海子诗集（风格池）
- `libai_train.jsonl`: 李白诗集（风格池）
- `annotations_sft_style.jsonl`: SFT 风格数据（顾城 113 / 海子 64 / 李白 3）
- `annotations_dpo_gen_format.jsonl`: DPO/KTO 偏好数据（chosen=真实诗, rejected=非诗文本）
- `annotations_dpo_gen_libai.jsonl`: 李白 KTO 偏好数据（838 条）
- `annotations_dpo_poem_nonpoem.jsonl` / `annotations_dpo_poem_vs_nonpoem.jsonl`: 诗/非诗判别数据
- `annotations_consistent.jsonl`: 标注一致数据集（396 条：诗 182 / 非诗 214）
- `hard_input_1000.jsonl`: 1000 条生成输入（Haizi/GuCheng/Haizi-CN/LiBai 各 250）
- `style100_titles.jsonl`: 100 个风格标题池

## 训练脚本（scripts/）

- `sft_style.py`: Qwen3.8-27B 三诗人风格 QLoRA SFT
- `kto_style.py`: KTO 单模型后训练（替代 DPO 双模型解决 27B+40GB OOM）
- `dpo_style.py`: DPO 双模型后训练
- `gen_hard_qwen38.py`: 生成 hard 负样本 + 相似度评测
- `gen_kto_eval.py`: KTO adapter 生成评测（gen_sim_pool）
- `gen_style_100.py`: 风格生成

## 模型（HuggingFace）

| 模型 | 说明 |
|---|---|
| `shikunpunk/Qwen3.8-27B-GuCheng` | 顾城 SFT adapter |
| `shikunpunk/Qwen3.8-27B-Haizi` | 海子 SFT adapter |
| `shikunpunk/Qwen3.8-27B-LiBai` | 李白 SFT adapter |
| `shikunpunk/Qwen3.8-27B-GuCheng-KTO` | 顾城 KTO 后训练 |
| `shikunpunk/Qwen3.8-27B-Haizi-KTO` | 海子 KTO 后训练 |
| `shikunpunk/Qwen3.8-27B-LiBai-KTO` | 李白 KTO 后训练 |

## 许可证

MIT
