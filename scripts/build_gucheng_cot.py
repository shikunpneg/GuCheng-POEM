# -*- coding: utf-8 -*-
"""构建顾城 CoT SFT 数据：sft_gucheng_cot.jsonl
模板：<题目> → 【读题】→【意象】→【情感基调】→【诗】
- 基础：对现有 213 条 SFT 样本的 assistant 诗体前加 CoT 前缀
- 意象从原诗文本真实提取（意象词表匹配）
- 情感基调从情感词表匹配（无匹配则用中性词）
- 全部为真实诗歌内容，CoT 前缀基于诗文本特征规则生成，不虚构
"""
import json, re, os, random
from collections import Counter

random.seed(42)
SFT = r"E:\生成诗歌\minimind\dataset\sft_gucheng.jsonl"
PRETRAIN = r"E:\生成诗歌\minimind\dataset\pretrain_gucheng.jsonl"
OUT = r"E:\生成诗歌\minimind\dataset\sft_gucheng_cot.jsonl"

# 顾城常用意象词表（从真实诗作归纳）
IMAGE_WORDS = ["黑夜", "眼睛", "光明", "天空", "月亮", "星星", "太阳", "云", "风", "雨", "雪",
               "树", "花", "鸟", "鱼", "水", "海", "山", "土地", "田野", "房子", "门", "窗",
               "灯", "火", "烟", "梦", "心", "路", "秋天", "春天", "夏天", "冬天", "声音",
               "钟", "船", "桥", "影子", "落叶", "早晨", "黄昏", "夜晚", "光", "虹", "孩子",
               "母亲", "手指", "眼睛", "蚂蚁", "乌鸦", "蝴蝶", "雪花", "石子", "玻璃", "钟表"]
EMO_WORDS = {
    "忧伤": ["忧伤", "悲哀", "难过", "流泪", "哭", "哭泣", "伤", "痛", "泪"],
    "孤独": ["孤独", "寂寞", "独自", "一个人", "无", "空", "静", "冷清"],
    "希望": ["希望", "光明", "明天", "生长", "春天", "发芽", "寻找", "等待"],
    "童真": ["孩子", "小孩", "童年", "玩具", "童话", "游戏", "天真", "纯"],
    "宁静": ["宁静", "安详", "安静", "默默", "轻轻", "悄悄", "静"],
    "荒凉": ["荒凉", "荒芜", "空旷", "废墟", "枯", "干涸"],
    "温暖": ["温暖", "温", "暖", "阳光", "火"],
    "迷惘": ["迷惘", "迷茫", "不知", "仿佛", "好像", "模糊", "遥远"],
}

def extract_images(text):
    found = []
    for w in IMAGE_WORDS:
        if w in text and w not in found:
            found.append(w)
    return found[:4]

def extract_emotion(text):
    scores = Counter()
    for emo, words in EMO_WORDS.items():
        c = sum(text.count(w) for w in words)
        if c:
            scores[emo] = c
    if scores:
        return scores.most_common(1)[0][0]
    return "沉静内敛"

def title_of(poem_text):
    # pretrain 格式: "标题\n\n正文"
    m = re.match(r"^(.+?)\n\n(.*)$", poem_text, re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "无题", poem_text

def build_cot(poem, title=None):
    """给一首诗生成 CoT 前缀（读题/意象/情感基调来自诗文本特征）"""
    if title is None:
        title, poem = title_of(poem)
    imgs = extract_images(poem)
    emo = extract_emotion(poem)
    if imgs:
        read = (f"《{title}》让我想到{imgs[0]}" +
                (f"与{imgs[1]}" if len(imgs) > 1 else "") +
                f"构成的画面，{emo}的气息藏在字句之间，我试着用顾城式的简洁把它写出来。")
        img_line = "、".join(imgs)
    else:
        read = f"《{title}》这个题目带着{emo}的气息，我试着用顾城式的简洁把它写出来。"
        img_line = "（意象内隐于行间）"
    return (f"【读题】{read}\n"
            f"【意象】{img_line}\n"
            f"【情感基调】{emo}\n"
            f"【诗】\n{poem}")

# ---------- 1. 从 213 条 SFT 样本生成 ----------
items = []
sft_items = [json.loads(l) for l in open(SFT, encoding="utf-8")]
print("sft base:", len(sft_items))
for it in sft_items:
    convs = it["conversations"]
    user_txt = convs[1]["content"] if convs[1]["role"] == "user" else convs[-2]["content"]
    assist_txt = convs[-1]["content"]
    m = re.search(r"《(.+?)》", user_txt)
    title = m.group(1) if m else "无题"
    new_assist = build_cot(assist_txt, title=title)
    new_convs = list(convs)
    new_convs[-1] = {"role": "assistant", "content": new_assist}
    items.append({"conversations": new_convs})
print("after sft base:", len(items))

# ---------- 2. 从 pretrain 真诗扩充（带标题的诗，严格过滤散文/对话） ----------
seen = set()
for it in items:
    seen.add(it["conversations"][-1]["content"])
SYSTEM_V = ("你是一位深谙顾城诗歌风格的现代诗人，擅长以纯真、梦幻、略带忧伤的笔触创作现代诗。"
            "顾城的诗以简洁的意象、童话般的想象和对生命本质的追问为特征。"
            "创作时先读题，得到自己的理解，再把理解转化为诗歌。")

def looks_like_poem(poem):
    if re.search(r"[A-Za-z]", poem):
        return False
    if re.search(r"\ufffd", poem):
        return False
    if re.search(r"[：:]{2,}", poem) or poem.count("：") > 3:
        return False
    lines = [l for l in poem.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    avg_len = sum(len(l) for l in lines) / len(lines)
    if avg_len > 45:  # 散文长句特征
        return False
    return True

added = 0
for line in open(PRETRAIN, encoding="utf-8"):
    obj = json.loads(line)
    t = obj.get("text", "")
    if "\n\n" not in t:
        continue
    title, poem = title_of(t)
    if not looks_like_poem(poem):
        continue
    if len(poem) < 30 or len(poem) > 400:
        continue
    new_assist = build_cot(poem, title=title)
    if new_assist in seen:
        continue
    seen.add(new_assist)
    items.append({"conversations": [
        {"role": "system", "content": SYSTEM_V},
        {"role": "user", "content": f"请以《{title}》为题，创作一首现代诗。"},
        {"role": "assistant", "content": new_assist},
    ]})
    added += 1
    if added >= 400:
        break
print("after pretrain expand:", len(items), "(added", added, ")")

# ---------- 3. 清理噪声样本 ----------
def clean_noise(assist):
    if re.search(r"[A-Za-z]", assist):
        return None
    if "pdg_" in assist or "策划监制" in assist or "特约编辑" in assist:
        return None
    return assist

kept = []
for it in items:
    a = it["conversations"][-1]["content"]
    a2 = clean_noise(a)
    if a2 is None:
        continue
    it["conversations"][-1] = {"role": "assistant", "content": a2}
    kept.append(it)
items = kept
print("after noise clean:", len(items))

with open(OUT, "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("written:", OUT, os.path.getsize(OUT))
