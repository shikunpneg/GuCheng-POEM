# -*- coding: utf-8 -*-
"""用 KTO 后训练 adapter（base + PeftModel）生成中文诗歌并评估风格保真（sim_pool）。

与 gen_hard_qwen38.py 的差异：
- 加载 base + KTO LoRA adapter（PeftModel），而非单一模型路径；
- 只生成 --style 指定风格（GuCheng / Haizi）的诗歌；
- 计算与同标题真实诗的字符 2-gram Jaccard/余弦相似度 + 风格池平均相似度。

用法:
  python gen_kto_eval.py --base /root/models/Qwen3.8-27B \
      --adapter /root/poetry-hard/saves/kto-GuCheng-GuCheng \
      --style GuCheng --input /root/poetry-hard/data/hard_input_1000.jsonl \
      --pool-dir /root/poetry-hard/data --out /root/poetry-hard/saves/eval_kto_GuCheng.jsonl

输出:
  <out>                      每条生成 + 相似度标注
  <out 同名目录>/stats.json  整体统计
"""
import argparse
import json
import math
import os
import random
import re
import time
import torch

BATCH = 4
MAX_NEW = 200
SEED = 20260812
POOL_SAMPLE = 20

STYLE_POOLS = {
    "GuCheng": "gucheng_train.jsonl",
    "Haizi": "haizi_train.jsonl",
    "LiBai": "libai_train.jsonl",
}


def bigrams(text):
    text = re.sub(r"\s+", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1)]


def jaccard(a, b):
    A, B = set(bigrams(a)), set(bigrams(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def cosine(a, b):
    from collections import Counter
    va, vb = Counter(bigrams(a)), Counter(bigrams(b))
    if not va or not vb:
        return 0.0
    dot = sum(c * vb.get(g, 0) for g, c in va.items())
    na = math.sqrt(sum(c * c for c in va.values()))
    nb = math.sqrt(sum(c * c for c in vb.values()))
    if na * nb == 0:
        return 0.0
    return dot / (na * nb)


def trim_incomplete(text):
    t = text.strip()
    while t and t[-1] in "，、：,":
        t = t[:-1].strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基座模型路径")
    ap.add_argument("--adapter", required=True, help="KTO LoRA adapter 目录")
    ap.add_argument("--style", required=True, choices=["GuCheng", "Haizi", "LiBai"])
    ap.add_argument("--input", required=True, help="生成输入 jsonl（含 meta.model 风格字段）")
    ap.add_argument("--pool-dir", default=".", help="风格池数据目录")
    ap.add_argument("--out", required=True, help="输出 jsonl")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new", type=int, default=MAX_NEW)
    ap.add_argument("--n", type=int, default=0, help="只跑前 n 条（调试用）")
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    # 只保留目标风格
    rows = [r for r in rows if (r.get("meta", {}) or {}).get("model") == args.style]
    if args.n:
        rows = rows[: args.n]
    print(f"[KTO-{args.style}] {len(rows)} prompts -> {out_path}", flush=True)
    assert rows, f"style {args.style} 无输入"

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
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
        args.base, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_cfg,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    gen_cfg = dict(
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 加载风格真实诗池
    pool_recs = []
    fname = STYLE_POOLS[args.style]
    p = os.path.join(args.pool_dir, fname)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gpt = next((m["value"] for m in rec.get("conversations", []) if m.get("from") == "gpt"), "")
                if gpt.strip():
                    pool_recs.append(gpt)
    print(f"[KTO-{args.style}] style pool: {len(pool_recs)}", flush=True)
    rng = random.Random(SEED)

    def encode(recs):
        msgs_list = []
        for rec in recs:
            conv = rec["conversations"]
            msgs = []
            for m in conv:
                if m["from"] == "system":
                    msgs.append({"role": "system", "content": m["value"]})
                elif m["from"] == "human":
                    msgs.append({"role": "user", "content": m["value"]})
            msgs_list.append(msgs)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs_list]
        enc = processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        )
        input_ids = enc["input_ids"].to(model.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(model.device)
        return input_ids, attn

    t0 = time.time()
    results = []
    sims = []
    with open(out_path, "w", encoding="utf-8") as fo:
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            input_ids, attn = encode(chunk)
            with torch.no_grad():
                out = model.generate(inputs=input_ids, attention_mask=attn, **gen_cfg)
            gen_ids = out[:, input_ids.shape[1]:]
            texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for rec, text in zip(chunk, texts):
                gen = trim_incomplete(text.strip())
                meta = rec["meta"]
                ref = next(m["value"] for m in rec["conversations"] if m["from"] == "gpt")
                sj = jaccard(gen, ref)
                sc = cosine(gen, ref)
                if pool_recs:
                    sp = sum(jaccard(gen, p) for p in rng.sample(pool_recs, min(POOL_SAMPLE, len(pool_recs)))) / min(POOL_SAMPLE, len(pool_recs))
                else:
                    sp = 0.0
                sims.append((sj, sc, sp))
                rec_out = {
                    "model": meta["model"],
                    "title": meta["title"],
                    "genre": meta["genre"],
                    "prompt": next(m["value"] for m in rec["conversations"] if m["from"] == "human"),
                    "generated": gen,
                    "real_text": ref,
                    "sim_jaccard": round(sj, 4),
                    "sim_cosine": round(sc, 4),
                    "sim_pool": round(sp, 4),
                }
                fo.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                results.append(rec_out)
            fo.flush()
            if (i // BATCH + 1) % 5 == 0 or (i + BATCH) >= len(rows):
                n = len(results)
                avg_j = sum(s[0] for s in sims) / max(len(sims), 1)
                avg_c = sum(s[1] for s in sims) / max(len(sims), 1)
                avg_p = sum(s[2] for s in sims) / max(len(sims), 1)
                print(f"progress={n}/{len(rows)} gen_sim_jaccard={avg_j:.4f} gen_sim_cosine={avg_c:.4f} gen_sim_pool={avg_p:.4f} elapsed={int(time.time()-t0)}s", flush=True)

    n = len(sims)
    jacs = [s[0] for s in sims]
    coss = [s[1] for s in sims]
    pools = [s[2] for s in sims]
    stats = {
        "model": f"Qwen3.8-27B+KTO-{args.style}",
        "adapter": args.adapter,
        "count": n,
        "jaccard_mean": round(sum(jacs) / n, 4),
        "cosine_mean": round(sum(coss) / n, 4),
        "pool_mean": round(sum(pools) / n, 4),
        "elapsed_sec": int(time.time() - t0),
    }
    stat_path = os.path.join(os.path.dirname(out_path), f"stats_kto_{args.style}.json")
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[KTO-{args.style}] DONE {n} rows, avg_jaccard={stats['jaccard_mean']}, avg_cosine={stats['cosine_mean']}, avg_pool={stats['pool_mean']}, {stats['elapsed_sec']}s")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
