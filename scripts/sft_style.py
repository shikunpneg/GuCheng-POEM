# -*- coding: utf-8 -*-
"""Qwen3.8-27B 三诗人风格 QLoRA SFT 微调。

用法:
  python sft_style.py --style GuCheng --train data/gucheng_train.jsonl \
      --model /root/models/Qwen3.8-27B --out /root/poetry-hard/saves/sft-GuCheng \
      --epochs 8

数据格式：ShareGPT 风格 jsonl（system + human + gpt），meta.model 标注诗人。
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


def build_messages(conv):
    msgs = []
    for m in conv:
        if m.get("from") == "system":
            msgs.append({"role": "system", "content": m["value"]})
        elif m.get("from") == "human":
            msgs.append({"role": "user", "content": m["value"]})
        elif m.get("from") == "gpt":
            msgs.append({"role": "assistant", "content": m["value"]})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, help="GuCheng / Haizi / LiBai")
    ap.add_argument("--train", required=True, help="训练数据 jsonl")
    ap.add_argument("--model", default="/root/models/Qwen3.8-27B")
    ap.add_argument("--out", default="/root/poetry-hard/saves/sft")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--resume-adapter", default=None, help="已有 LoRA adapter 目录（继续增强微调）")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=768)
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    rows = load_rows(args.train)
    print(f"[{args.style}] train rows: {len(rows)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_cfg,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # Qwen3_5: linear_attn + mlp + lm_head
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if args.resume_adapter and os.path.isdir(args.resume_adapter):
        from peft import PeftModel as _PeftModel
        # 已有 adapter 继续训练：加载原 adapter，保持可训练
        model = _PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"[{args.style}] resumed from adapter: {args.resume_adapter}", flush=True)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[{args.style}] trainable {trainable/1e6:.2f}M / {total/1e9:.2f}B", flush=True)

    # 构造 dataset
    from datasets import Dataset

    def tokenize_fn(batch):
        texts = []
        for conv in batch["conversations"]:
            msgs = build_messages(conv)
            texts.append(processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False))
        enc = processor(text=texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_len)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        # mask pad
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    data = {"conversations": [r["conversations"] for r in rows]}
    ds = Dataset.from_dict(data)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["conversations"])
    ds.set_format("torch")

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

    args_out = TrainingArguments(
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
        save_steps=200,
        save_total_limit=2,
        save_only_model=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=args_out, train_dataset=ds,
                      callbacks=[EpochSummaryCallback(args.epochs)])
    trainer.train()
    # 保存 LoRA adapter（不合并）
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[{args.style}] SFT DONE -> {out_dir}", flush=True)
    print("train_loss", end=" ", flush=True)
    logs = trainer.state.log_history
    for lg in logs:
        if "loss" in lg:
            print(f"{lg['loss']:.4f}", end=" ", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
