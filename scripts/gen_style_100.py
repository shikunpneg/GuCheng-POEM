# -*- coding: utf-8 -*-
"""生成三风格同题模仿诗：从统一标题池取前 N 个标题，三个模型共用相同题目各生成一首。

用法:
  python gen_style_100.py --model /root/poetry-hard/saves/sft-GuCheng --adapter /root/poetry-hard/saves/sft-GuCheng \
      --style GuCheng --input data/style100_titles.jsonl --out out/style100_GuCheng.jsonl
"""
import argparse
import json
import os
import time
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base 模型路径")
    ap.add_argument("--adapter", required=True, help="LoRA adapter 路径")
    ap.add_argument("--style", required=True, help="GuCheng / Haizi / LiBai")
    ap.add_argument("--input", required=True, help="标题输入 jsonl（每条含 system/human/meta）")
    ap.add_argument("--out", required=True, help="输出 jsonl")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("meta", {}).get("model") == args.style]
    print(f"[{args.style}] rows: {len(rows)}", flush=True)
    assert rows

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    if os.path.isdir(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    gen_cfg = dict(
        max_new_tokens=args.max_new,
        temperature=1.0,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

    def encode(recs):
        msgs_list = []
        for rec in recs:
            msgs = []
            for m in rec["conversations"]:
                if m["from"] == "system":
                    msgs.append({"role": "system", "content": m["value"]})
                elif m["from"] == "human":
                    msgs.append({"role": "user", "content": m["value"]})
            msgs_list.append(msgs)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs_list]
        enc = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(model.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(model.device)
        return input_ids, attn

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    n = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i:i + args.batch_size]
            input_ids, attn = encode(chunk)
            with torch.no_grad():
                out = model.generate(inputs=input_ids, attention_mask=attn, **gen_cfg)
            gen_ids = out[:, input_ids.shape[1]:]
            texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for rec, text in zip(chunk, texts):
                gen = text.strip()
                while gen and gen[-1] in "，、：,":
                    gen = gen[:-1].strip()
                rec_out = {
                    "model": args.style,
                    "title": rec["meta"]["title"],
                    "genre": rec["meta"].get("genre", ""),
                    "prompt": next(m["value"] for m in rec["conversations"] if m["from"] == "human"),
                    "generated": gen,
                }
                fo.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                n += 1
            fo.flush()
            if (i // args.batch_size + 1) % 5 == 0:
                print(f"progress={n}/{len(rows)} elapsed={int(time.time()-t0)}s", flush=True)
    print(f"[{args.style}] DONE {n} rows -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
