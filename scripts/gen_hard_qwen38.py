# -*- coding: utf-8 -*-
"""用 Qwen3.8-27B 生成中文诗歌困难负样本（hard negative samples）。

与旧实验 (gen_hard.py) 的差异：
- 旧实验：4 个 QLoRA 微调模型（LiBai/Haizi/Haizi-CN/GuCheng）各 1250 条；
- 本次：单一最新 Qwen3.8-27B 模型，通过 system 风格人设提示控制诗人风格，生成 1000 条
  （每风格 250 条），计算与同标题真实诗的字符 2-gram Jaccard/余弦相似度 + 风格池平均相似度。

用法:
  python gen_hard_qwen38.py --model models/Qwen3.8-27B --input hard_input_1000.jsonl \
      --data-dir llama_data --out hard_gen_Qwen38.jsonl

输出:
  hard_gen_Qwen38.jsonl       每条生成 + 相似度标注
  hard_stats_Qwen38.json      整体统计
"""
import argparse
import json
import math
import os
import random
import re
import time
import torch
BATCH = 6
MAX_NEW = 200
SEED = 20260812

# 各风格对应的真实诗池（计算风格保真度 sim_pool）
STYLE_POOLS = {
    "GuCheng": "gucheng_train.jsonl",
    "Haizi": "haizi_train.jsonl",
    "Haizi-CN": "haizi_cn_train.jsonl",
    "LiBai": "libai_train.jsonl",
}
POOL_SAMPLE = 20


# ---------- 相似度 ----------
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
    ap.add_argument("--model", required=True, help="模型路径或 HF 仓库")
    ap.add_argument("--input", required=True, help="生成输入 jsonl")
    ap.add_argument("--data-dir", default="llama_data", help="风格池数据目录")
    ap.add_argument("--out", default="hard_gen_Qwen38.jsonl", help="输出文件名")
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
    if args.n:
        rows = rows[: args.n]
    print(f"[Qwen38] {len(rows)} prompts -> {out_path}", flush=True)
    assert rows, "输入无数据"

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    # Qwen3.8-27B 是多模态 Qwen3_5 架构；纯文本生成走 AutoProcessor + AutoModelForCausalLM
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
        torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_cfg,
    )
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
    style_recs = {}
    for group, fname in STYLE_POOLS.items():
        recs = []
        p = os.path.join(args.data_dir, fname)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    gpt = next((m["value"] for m in rec.get("conversations", []) if m.get("from") == "gpt"), "")
                    if gpt.strip():
                        recs.append(gpt)
        style_recs[group] = recs
        print(f"[Qwen38] style pool {group}: {len(recs)}", flush=True)
    rng = random.Random(SEED)

    # 预编码（按批左填充）
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
                pool = style_recs.get(meta["model"], [])
                if pool:
                    sp = sum(jaccard(gen, p) for p in rng.sample(pool, min(POOL_SAMPLE, len(pool)))) / min(POOL_SAMPLE, len(pool))
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

    # 统计
    n = len(sims)
    jacs = [s[0] for s in sims]
    coss = [s[1] for s in sims]
    pools = [s[2] for s in sims]
    stats = {
        "model": "Qwen3.8-27B",
        "count": n,
        "jaccard_mean": round(sum(jacs) / n, 4),
        "jaccard_median": round(sorted(jacs)[n // 2], 4),
        "jaccard_max": round(max(jacs), 4),
        "jaccard_over_0_4": round(sum(1 for x in jacs if x > 0.4) / n, 4),
        "jaccard_over_0_6": round(sum(1 for x in jacs if x > 0.6) / n, 4),
        "cosine_mean": round(sum(coss) / n, 4),
        "pool_mean": round(sum(pools) / n, 4),
        "pool_median": round(sorted(pools)[n // 2], 4),
        "elapsed_sec": int(time.time() - t0),
    }
    stat_path = os.path.join(os.path.dirname(out_path), "hard_stats_Qwen38.json")
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[Qwen38] DONE {n} rows, avg_jaccard={stats['jaccard_mean']}, avg_cosine={stats['cosine_mean']}, avg_pool={stats['pool_mean']}, {stats['elapsed_sec']}s")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
