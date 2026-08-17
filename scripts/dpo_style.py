# -*- coding: utf-8 -*-
"""Qwen3.8-27B 三诗人风格 DPO 后训练（基于 SFT adapter）。

用法:
  python dpo_style.py --style GuCheng --train data/annotations_dpo_gen_format.jsonl \
      --base /root/models/Qwen3.8-27B --adapter /root/poetry-hard/saves/sft-GuCheng-ann \
      --out /root/poetry-hard/saves/dpo-GuCheng --epochs 3

DPO 偏好：chosen=人类标注一致的真实诗歌，rejected=人类标注一致的非诗文本。
数据格式（annotations_dpo_gen_format.jsonl）：
  conversations: [system(风格人设), human(创作请求)]
  chosen: 真实诗歌
  rejected: 非诗文本
"""
import argparse
import json
import os
import torch

def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, help="GuCheng / Haizi / LiBai")
    ap.add_argument("--train", required=True, help="DPO 偏好数据 jsonl")
    ap.add_argument("--base", default="/root/models/Qwen3.8-27B")
    ap.add_argument("--adapter", required=True, help="SFT LoRA adapter 目录")
    ap.add_argument("--out", default="/root/poetry-hard/saves/dpo")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, TrainingArguments, TrainerCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    rows = load_rows(args.train)
    # 只保留该风格的样本（meta.style 或按 conversations[0] 匹配风格人设）
    filtered = []
    for r in rows:
        meta = r.get("meta", {}) or {}
        style = meta.get("style", "")
        if not style:
            # fallback：按 conversations[0] 中的诗人名判断
            sysv = (r.get("conversations") or [{}])[0].get("value", "")
            if "顾城" in sysv:
                style = "GuCheng"
            elif "海子" in sysv:
                style = "Haizi"
            elif "李白" in sysv:
                style = "LiBai"
        if style == args.style:
            filtered.append(r)
    rows = filtered
    print(f"[{args.style}] DPO rows: {len(rows)}", flush=True)
    assert rows, f"style {args.style} 无 DPO 数据"

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # Policy: base + SFT LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_cfg,
    )
    if os.path.isdir(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Reference: base + SFT LoRA（冻结）。双 4bit 模型共约 35GB，需配合 max_length=256
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
    )
    if os.path.isdir(args.adapter):
        ref_model = PeftModel.from_pretrained(ref_model, args.adapter, is_trainable=False)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # 构造 DPO dataset（trl 原生 prompt/chosen/rejected 文本格式）
    from datasets import Dataset

    def build_prompt(conv):
        # conversations: [system, human]
        msgs = []
        for m in conv:
            if m.get("from") == "system":
                msgs.append({"role": "system", "content": m["value"]})
            elif m.get("from") == "human":
                msgs.append({"role": "user", "content": m["value"]})
        return processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    data = {
        "prompt": [build_prompt(r["conversations"]) for r in rows],
        "chosen": [r["chosen"]["value"] if isinstance(r.get("chosen"), dict) else r["chosen"] for r in rows],
        "rejected": [r["rejected"]["value"] if isinstance(r.get("rejected"), dict) else r["rejected"] for r in rows],
    }
    ds = Dataset.from_dict(data)
    print(f"[{args.style}] dataset ready: {len(ds)}", flush=True)
    print("example prompt:", data["prompt"][0][:120], flush=True)

    out_dir = f"{args.out}-{args.style}"
    os.makedirs(out_dir, exist_ok=True)

    class EpochSummaryCallback(TrainerCallback):
        def __init__(self, total_epochs):
            self.total_epochs = total_epochs
            self.last_epoch = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            epoch = logs.get("epoch")
            if epoch is not None and int(epoch) > self.last_epoch:
                self.last_epoch = int(epoch)
                loss = logs.get("loss")
                if loss is not None:
                    print(f"epoch={self.last_epoch}/{self.total_epochs} train_loss={loss:.6f}", flush=True)

    # 使用 TRL DPOTrainer（trl 1.10 API：DPOConfig 已包含全部训练参数，processing_class 替代 tokenizer）
    from trl import DPOTrainer, DPOConfig
    dpo_config = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        beta=args.beta,
        max_length=args.max_len,
        disable_dropout=True,
    )
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        callbacks=[EpochSummaryCallback(args.epochs)],
    )
    dpo_trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[{args.style}] DPO DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
