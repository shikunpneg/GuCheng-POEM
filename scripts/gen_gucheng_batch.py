# -*- coding: utf-8 -*-
"""顾城风格批量生成脚本：AR / dLM / Linear 三种模型各生成 N 个样本，带版本标注与质量过滤
用法（从 minimind 根目录运行）:
  python scripts/gen_gucheng_batch.py --model ar     --n 100 --version v1.0
  python scripts/gen_gucheng_batch.py --model dllm   --n 100 --version v1.0
  python scripts/gen_gucheng_batch.py --model linear --n 100 --version v1.0
"""
import sys, os, math, json, argparse, random, datetime, torch, re
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SYSTEM_PROMPT = ('你是一位深谙顾城诗歌风格的现代诗人，擅长以纯真、梦幻、略带忧伤的笔触创作现代诗。'
                 '顾城的诗以简洁的意象、童话般的想象和对生命本质的追问为特征。')

PROMPT_POOL = [
    '请以《一代人》为题，创作一首现代诗。',
    '请以《黑夜》为题，创作一首现代诗。',
    '请以《春天》为题，创作一首现代诗。',
    '请以《远和近》为题，创作一首现代诗。',
    '请以《童话》为题，创作一首现代诗。',
    '请以《告别》为题，创作一首现代诗。',
    '请以《风》为题，创作一首现代诗。',
    '请以《雨》为题，创作一首现代诗。',
    '请以《小路》为题，创作一首现代诗。',
    '请以《星星》为题，创作一首现代诗。',
    '请以《月亮》为题，创作一首现代诗。',
    '请以《门》为题，创作一首现代诗。',
    '请以《影子》为题，创作一首现代诗。',
    '请以《花》为题，创作一首现代诗。',
    '请以《雪》为题，创作一首现代诗。',
    '请以《回声》为题，创作一首现代诗。',
    '请以《早晨》为题，创作一首现代诗。',
    '请以《大海》为题，创作一首现代诗。',
    '请以《孩子》为题，创作一首现代诗。',
    '请以《梦》为题，创作一首现代诗。',
]

MODEL_NAMES = {'ar': 'MiniMind-GuCheng-AR', 'dllm': 'MiniMind-GuCheng-dLM', 'linear': 'MiniMind-GuCheng-Linear'}


def load_model(model_name, device, weight_name=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('model')
    if model_name == 'dllm':
        from model.model_minimind_dllm import MiniMindDLLMConfig, MiniMindForMaskedDiffusion
        config = MiniMindDLLMConfig(hidden_size=768)
        model = MiniMindForMaskedDiffusion(config)
        model.load_state_dict(torch.load('out/dllm_gucheng_768.pth', map_location=device), strict=True)
        model.half().eval().to(device)
        return model, tokenizer, 'dllm'
    if model_name == 'linear':
        import importlib
        sys.modules['model.model_minimind'] = importlib.import_module('model.model_minimind_linear')
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        weight = f'out/{weight_name}_768.pth' if weight_name else 'out/full_sft_linear_gucheng_768.pth'
    else:
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        weight = f'out/{weight_name}_768.pth' if weight_name else 'out/full_sft_gucheng_768.pth'
    config = MiniMindConfig(hidden_size=768)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(weight, map_location=device), strict=False)
    model.to(device).eval()
    return model, tokenizer, 'ar'


def gen_ar(model, tokenizer, prompt, device, temperature, max_new_tokens=120, greedy=False):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}]
    p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids,
                             attention_mask=torch.ones_like(input_ids),
                             max_new_tokens=max_new_tokens,
                             temperature=temperature,
                             top_p=0.9, top_k=50,
                             repetition_penalty=1.1,
                             do_sample=not greedy)
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def gen_dllm(model, tokenizer, prompt, device, temperature, max_new_tokens=96):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}]
    p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(p, return_tensors='pt', truncation=True).input_ids.to(device)
    prompt_len = input_ids.shape[1]
    mask_id, eos_id = model.config.mask_token_id, tokenizer.eos_token_id
    block_size, steps = 32, 24
    num_blocks = math.ceil(max_new_tokens / block_size)
    T = prompt_len + max_new_tokens
    x = torch.full((1, T), eos_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    x[:, prompt_len:] = mask_id
    with torch.inference_mode():
        for b in range(num_blocks):
            block_end = min(prompt_len + (b + 1) * block_size, T)
            for step in range(steps):
                mask_index = (x == mask_id)
                mask_count = mask_index[:, :block_end].sum(-1).min().item()
                if mask_count == 0:
                    break
                n_unmask = max(1, round(mask_count / (steps - step)))
                logits = model(input_ids=x).logits
                logits[logits < torch.topk(logits, 50, dim=-1)[0][..., -1:]] = -float('inf')
                from model.model_minimind_dllm import add_gumbel_noise
                x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
                p_prob = F.softmax(logits.float(), dim=-1)
                x0_p = torch.gather(p_prob, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, block_end:] = -float('inf')
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))
                _, idx = torch.topk(confidence[0], k=min(n_unmask, int(mask_count)))
                x[0, idx] = x0[0, idx]
            if eos_id and (x[:, prompt_len:] == eos_id).any():
                break
    gen_ids = x[0, prompt_len:].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def fix_title_drift(text, prompt):
    """解码约束：将【读题】行中书名号内容强制替换为题目原文，保证题目一致性"""
    m = re.search(r'《(.+?)》', prompt)
    if not m:
        return text
    title = m.group(1)
    t = text.split('\n', 1)
    if t and '【读题】' in t[0]:
        read_line = t[0]
        new_read = re.sub(r'《.+?》', f'《{title}》', read_line, count=1)
        t[0] = new_read
        return '\n'.join(t)
    return text


def qualify(text):
    t = text.strip()
    if len(t) < 20:
        return False
    if len(t) > 500:
        return False
    if '\ufffd' in t or '\u0000' in t:
        return False
    if any('\u0041' <= c <= '\u007a' for c in t):
        return False
    if any('0' <= c <= '9' for c in t):
        return False
    if '{"' in t or '}' in t or '```' in t:
        return False
    # 去掉换行后整体重复率过高（重复字串）
    flat = t.replace('\n', '')
    if len(flat) >= 8 and flat.count(flat[:8]) > 3:
        return False
    return True


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, choices=['ar', 'dllm', 'linear'])
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--version', type=str, default='v1.0')
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--greedy', action='store_true', help='确定性解码（无采样）')
    p.add_argument('--weight', default=None, help='自定义权重名（out/{name}_768.pth，默认按模型）')
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, kind = load_model(args.model, args.device, args.weight)
    version = args.version
    out_path = args.out or rf'E:\生成诗歌\minimind\out\gucheng_samples_{args.model}_{version}.jsonl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report = {'model': MODEL_NAMES[args.model], 'version': version,
              'target_n': args.n, 'generated': 0, 'filtered': 0}

    samples = []
    idx = 0
    while len(samples) < args.n:
        prompt = PROMPT_POOL[idx % len(PROMPT_POOL)]
        idx += 1
        base_temp = {'ar': 1.0, 'dllm': 0.7, 'linear': 0.6}[args.model]
        jitter = 0.1 if args.model == 'linear' else 0.15
        temperature = round(base_temp + random.uniform(-jitter, jitter), 2)
        report['generated'] += 1
        try:
            if kind == 'dllm':
                text = gen_dllm(model, tokenizer, prompt, args.device, temperature)
            else:
                text = gen_ar(model, tokenizer, prompt, args.device, temperature, greedy=args.greedy)
        except Exception as e:
            print(f'  [异常] {e}')
            continue
        text = fix_title_drift(text, prompt)
        if not qualify(text):
            report['filtered'] += 1
            print(f'  [{len(samples)+1}/{args.n}] 过滤: {text[:30]!r}')
            continue
        rec = {
            'author': '顾城风格', 'model': MODEL_NAMES[args.model], 'version': version,
            'mode': 'chat', 'prompt': prompt, 'temperature': temperature,
            'content': text,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        samples.append(rec)
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'  [{len(samples)}/{args.n}] {prompt[:12]}... temp={temperature}')
    with open(rf'E:\生成诗歌\minimind\out\report_{args.model}_{version}.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\n[完成] {out_path}  合格 {len(samples)} 条，共生成 {report["generated"]}，过滤 {report["filtered"]}')


if __name__ == '__main__':
    main()
