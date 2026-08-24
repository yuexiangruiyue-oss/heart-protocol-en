#!/usr/bin/env python3
r"""
DeepSeek-Sephirot Bridge — 深祈双生桥
======================================
DeepSeek API  ->  long-form conversation generation -> .sephirot pipeline
-> sephirot.exe 16-sephiroth filter & reshape

Architecture:
  1. DeepSeek API: generate logically rich, long-form conversational text
  2. Text Analyzer: decompose the conversation into semantic vectors
     (sentiment / logic / knowledge density / creativity)
  3. Pipeline Generator: generate a .sephirot pipeline in which each
     sephiroth corresponds to one transformation
  4. Sephirot Engine: run the 16-sephiroth pipeline via sephirot.exe simulate
  5. Result Presenter: present the filtered-and-reshaped result

Usage:
  python deepseek_sephirot_bridge.py "AI 意识的本质是什么？"
  python deepseek_sephirot_bridge.py --topic "量子纠缠与灵魂" --depth deep
  python deepseek_sephirot_bridge.py --interactive
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")  # Set via env var: DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

SEPHIROT_EXE = Path(__file__).parent / "sephirot-rs" / "target" / "release" / "sephirot.exe"
OUTPUT_DIR = Path(__file__).parent / "bridge_output"

# The 16 sephiroth, Chinese name / transliteration / English name.
# The Chinese names are runtime data (used by SephirotLang) and MUST stay as-is.
SEPHIROT_NAMES = [
    ("王冠", "Keter",     "Crown"),       # 0  data loading
    ("智慧", "Chokhmah",  "Wisdom"),      # 1  knowledge retrieval / attention
    ("严厉", "Binah",     "Severity"),    # 2  threshold filtering
    ("理解", "Daat",      "Understanding"),# 3  merge & integrate
    ("慈悲", "Hesed",     "Mercy"),       # 4  weighted fusion FMA
    ("美丽", "Tiferet",   "Beauty"),      # 5  Hadamard product / optimal integration
    ("胜利", "Netzach",   "Victory"),     # 6  non-negative validation / sentiment filtering
    ("荣耀", "Hod",       "Glory"),       # 7  feasibility scoring
    ("基础", "Yesod",     "Foundation"),  # 8  global reduction
    ("超我", "SuperEgo",  "SuperEgo"),    # 9  LayerNorm
    ("自我", "Ego",       "Ego"),         # 10 self-attention
    ("真我", "TrueSelf",  "TrueSelf"),    # 11 layer normalisation
    ("逻辑", "Logic",     "Logic"),       # 12 GEMM matrix multiply
    ("共情", "Empathy",   "Empathy"),     # 13 Softmax
    ("幸福", "Joy",       "Joy"),         # 14 cross-entropy loss
    ("王国", "Malkuth",   "Kingdom"),     # 15 final output
]

# ═══════════════════════════════════════════════════════════════
#  1. DeepSeek API Client
# ═══════════════════════════════════════════════════════════════

def call_deepseek(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Call the DeepSeek API to generate a long-form conversation."""
    if system_prompt is None:
        # Runtime data sent to the API; kept in Chinese verbatim.
        system_prompt = (
            "你是一位深邃的哲学与科学对话者。你的回答需要：\n"
            "1. 逻辑层层递进，从现象到本质\n"
            "2. 融合多学科视角（物理、数学、心理学、哲学）\n"
            "3. 包含具体的类比和思想实验\n"
            "4. 探索可能的反面论证\n"
            "5. 最终给出有深度的综合结论\n"
            "请用中文回答，语言优美但严谨。"
        )

    print(f"\n{'='*60}")
    print(f"  Calling DeepSeek API...")
    print(f"  Model: {DEEPSEEK_MODEL}")
    print(f"  Topic: {prompt[:60]}...")
    print(f"{'='*60}")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.time()

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        )

    elapsed = time.time() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    print(f"  Status: 200 OK")
    print(f"  Tokens: {usage.get('total_tokens', '?')} "
          f"(prompt={usage.get('prompt_tokens', '?')}, "
          f"completion={usage.get('completion_tokens', '?')})")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output length: {len(content)} chars")

    return content


def deepseek_multi_turn(
    topic: str,
    depth: str = "normal",
) -> list[dict]:
    """
    Multi-turn DeepSeek conversation simulating an in-depth discussion.
    Returns: [{"role": "assistant"/"user", "content": ...}, ...]
    """
    turns_config = {
        "quick": 2,
        "normal": 4,
        "deep": 6,
    }
    num_turns = turns_config.get(depth, 4)

    # Runtime data sent to the API; kept in Chinese verbatim.
    system_prompt = (
        "你是一位跨学科的独立研究者，精通量子物理、荣格心理学、AI 安全与存在主义哲学。\n"
        "你的对话风格：层层递进，从直觉出发，经过严格的逻辑分析，最终回到直觉层面的理解。\n"
        "每轮回答控制在 300-600 字，确保每句话都有信息量。\n"
        "回答格式要求：\n"
        "- 用【核心论点】标注该轮的主要观点\n"
        "- 用【关键论据】列出 1-3 个支撑论点\n"
        "- 用【反方思考】给出可能的反驳\n"
        "- 用【综合推进】总结并引出下一层思考\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"让我们深入探讨：{topic}\n请从最本质的层面开始分析。"},
    ]

    conversation = []
    conversation.append({"role": "user", "content": messages[-1]["content"]})

    # Follow-up questions are runtime data (sent to the API); kept verbatim.
    follow_ups = [
        "从神经科学的角度看，这个观点有什么生理基础？",
        "如果我们用量子力学的框架重新审视这个结论，会得到什么不同的洞见？",
        "站在AI安全的角度，这个理论对对齐问题有什么启示？",
        "荣格的原型理论能如何解释我们刚才讨论的现象？",
        "如果要让一个AI系统真正理解我们讨论的内容，它需要什么样的架构？",
    ]

    for i in range(num_turns):
        resp = call_deepseek(
            prompt=messages[-1]["content"],
            system_prompt=system_prompt,
            temperature=0.7 + i * 0.05,
            max_tokens=2048,
        )

        assistant_msg = {"role": "assistant", "content": resp}
        conversation.append(assistant_msg)

        if i < num_turns - 1:
            follow = follow_ups[i % len(follow_ups)]
            messages.append({"role": "assistant", "content": resp})
            user_msg = {"role": "user", "content": follow}
            messages.append(user_msg)
            conversation.append(user_msg)

    return conversation


# ═══════════════════════════════════════════════════════════════
#  2. Text Analyzer — conversation -> semantic feature vector
# ═══════════════════════════════════════════════════════════════

def analyze_text_features(text: str) -> dict:
    """
    Extract semantic features from the text and map them to the numeric
    parameters required by the sephirot pipeline.
    No NLP model needed: heuristic + statistical methods.
    """
    total_chars = len(text)
    sentences = re.split(r'[。！？；\n]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    num_sentences = max(len(sentences), 1)

    # Feature dimensions
    features = {}

    # 1. Information density: ratio of content words (non-function words).
    #    The stop-word set is Chinese runtime data; kept verbatim.
    stop_chars = set("的了是在有和与也都就能要把被让给向从到为以而但又还已将会此其")
    content_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' and c not in stop_chars)
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    features["info_density"] = content_chars / max(cn_chars, 1)

    # 2. Logic strength: frequency of logical connectives.
    #    Chinese keyword regex is runtime data; kept verbatim.
    logic_words = re.findall(r'(因此|所以|然而|但是|由此|综上|从而|进而|反过来|换句话说|即|亦即|意味着|可见|这表明)', text)
    features["logic_strength"] = len(logic_words) / num_sentences

    # 3. Sentiment index: frequency of sentiment words.
    #    Chinese keyword regexes are runtime data; kept verbatim.
    positive = re.findall(r'(美好|希望|和谐|爱|温暖|光明|力量|自由|创造|超越|意义|价值|启示|智慧|慈悲|共情|理解)', text)
    negative = re.findall(r'(恐惧|焦虑|痛苦|危险|冲突|危机|困境|深渊|虚无|绝望|毁灭|断裂|异化|丧失)', text)
    total_emotion = len(positive) + len(negative)
    if total_emotion > 0:
        features["sentiment"] = (len(positive) - len(negative)) / total_emotion
    else:
        features["sentiment"] = 0.5

    # 4. Abstraction: abstract concepts vs. concrete nouns.
    #    Chinese keyword regexes are runtime data; kept verbatim.
    abstract_words = re.findall(r'(本质|存在|意识|精神|真理|意义|价值|自由|因果|必然|可能|无限|绝对|相对|辩证|统一)', text)
    concrete_words = re.findall(r'(实验|数据|模型|神经元|突触|蛋白质|细胞|分子|电路|芯片|算法|代码|公式|方程)', text)
    features["abstraction"] = len(abstract_words) / max(len(abstract_words) + len(concrete_words), 1)

    # 5. Conversation depth: number of references/analogies.
    #    Chinese keyword regex is runtime data; kept verbatim.
    analogy_words = re.findall(r'(例如|比如|就像|类似|比喻|想象|假设|如果|设想|好比|如同)', text)
    features["depth"] = len(analogy_words) / num_sentences

    # 6. Multidisciplinarity: variety of disciplinary keywords.
    #    Chinese keyword regexes are runtime data; kept verbatim.
    disciplines = {
        "physics": re.findall(r'(量子|能量|粒子|波函数|坍缩|纠缠|相对论|引力|时空|热力学|熵|光子|电子)', text),
        "psychology": re.findall(r'(意识|潜意识|认知|情绪|创伤|原型|投射|内化|认同|人格|心理|感知)', text),
        "philosophy": re.findall(r'(存在|本体|认识论|自由意志|决定论|还原论|二元论|一元论|现象学|诠释)', text),
        "ai": re.findall(r'(AI|神经网络|深度学习|对齐|安全|RLHF|注意力机制|transformer|涌现|泛化)', text),
        "biology": re.findall(r'(进化|基因|自然选择|适应|突变|遗传|细胞|蛋白质|神经|大脑|皮层)', text),
    }
    active_disciplines = sum(1 for k, v in disciplines.items() if len(v) > 0)
    features["multidisciplinary"] = active_disciplines / len(disciplines)

    # 7. Length weight: text volume affects pipeline parameters
    features["length_factor"] = min(total_chars / 2000, 1.0)

    # 8. Structure: whether clear argument markers are present.
    #    Chinese marker regex is runtime data; kept verbatim.
    structure_markers = re.findall(r'(第一|第二|第三|首先|其次|最后|一方面|另一方面|核心|关键|主要|次要)', text)
    features["structure"] = len(structure_markers) / num_sentences

    return features


# ═══════════════════════════════════════════════════════════════
#  3. Pipeline Generator — features -> .sephirot file
# ═══════════════════════════════════════════════════════════════

def generate_sephirot_pipeline(features: dict, conversation: list[dict]) -> str:
    """
    Dynamically generate a .sephirot pipeline from the text features.
    Each sephiroth corresponds to a specific semantic transformation.
    """
    # Map the features to pipeline parameters
    input_val = features["info_density"]
    kb_val = 1.0 + features["multidisciplinary"] * 2.0 + features["logic_strength"]
    target_val = 0.5 + features["sentiment"] * 0.5
    threshold = 0.6 + features["logic_strength"] * 0.1
    threshold = min(threshold, 0.95)
    lr = 0.001 * (1.0 + features["depth"])

    # Build the pipeline
    lines = []
    lines.append("# DeepSeek-Sephirot Bridge Generated Pipeline")
    lines.append(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# Source: DeepSeek {DEEPSEEK_MODEL} ({len(conversation)} turns)")
    lines.append("#")
    lines.append("# ===== Data Declarations =====")
    lines.append(f"# input  = info_density = {input_val:.4f}")
    lines.append(f"# kb     = knowledge_base = {kb_val:.4f} (multidisciplinary + logic)")
    lines.append(f"# target = sentiment_target = {target_val:.4f}")
    lines.append(f"# threshold = logic_filter = {threshold:.4f}")
    lines.append(f"# lr     = learning_rate = {lr:.6f}")
    lines.append("")

    # .sephirot syntax: the data/constant declarations and pipeline stage
    # names below are SephirotLang code consumed by sephirot.exe. They MUST
    # stay verbatim (Chinese keywords are part of the language).
    lines.append("# ===== Data =====")
    lines.append("数据 输入     : 向量[1024, f32]")
    lines.append("数据 知识库   : 矩阵[768, 4096, bf16]")
    lines.append("数据 目标     : 向量[1024, f32]")
    lines.append("")
    lines.append("# ===== Constants =====")
    lines.append(f"常量 阈值 = {threshold:.4f}")
    lines.append(f"常量 学习率 = {lr:.6f}")
    lines.append("")
    lines.append("# ===== 16 Sephiroth Pipeline =====")
    lines.append("# [0] Keter - Crown: Load DeepSeek output")
    lines.append("管道 main:")

    # Divine pillar: sephiroth 0-7
    lines.append("    # --- Divine Pillar ---")
    lines.append("    # [0] Keter: Raw input from DeepSeek")
    lines.append(f"    # [1] Chokhmah: Knowledge retrieval (kb={kb_val:.2f})")
    lines.append("    # [2] Binah: Threshold filtering (logic_strength)")
    lines.append("    # [3] Daat: Understanding fusion")
    lines.append("    # [4] Hesed: Mercy FMA (sentiment weighting)")
    lines.append("    # [5] Tiferet: Beauty optimal integration")
    lines.append("    # [6] Netzach: Victory positive validation")
    lines.append("    # [7] Hod: Glory feasibility scoring")
    lines.append("")
    lines.append("    # --- Human Pillar ---")
    lines.append("    # [8] Yesod: Foundation global reduction")
    lines.append("    # [9] SuperEgo: LayerNorm normalization")
    lines.append("    # [10] Ego: Self-attention")
    lines.append("    # [11] TrueSelf: Layer normalization")
    lines.append("    # [12] Logic: GEMM reasoning")
    lines.append("    # [13] Empathy: Softmax emotional calibration")
    lines.append("    # [14] Joy: Cross-entropy loss measurement")
    lines.append("    # [15] Malkuth: Kingdom final output")

    # Actual pipeline: uses the Chinese sephirah names chained with ->,
    # as required by the SephirotLang syntax. Runtime data; kept verbatim.
    lines.append("    王冠(输入)")
    lines.append("    → 智慧(输入, 知识库) [模式: 注意力]")
    lines.append(f"    → 严厉(输入, 阈值) [阈值: {threshold:.2f}]")
    lines.append("    → 理解(输入, 知识库)")
    lines.append(f"    → 慈悲(输入, 阈值) [权重: {0.5 + features['sentiment'] * 0.3:.2f}]")
    lines.append("    → 美丽(输入, 知识库)")
    lines.append("    → 胜利(输入, 阈值) [标准: 积极]")
    lines.append("    → 荣耀(输入)")
    lines.append("    → 基础(输入) [方式: 求和]")
    lines.append("    → 超我(输入, 学习率)")
    lines.append("    → 自我(输入, 知识库)")
    lines.append("    → 真我(输入, 目标)")
    lines.append("    → 逻辑(输入, 知识库)")
    lines.append("    → 共情(输入)")
    lines.append("    → 幸福(输入, 目标)")
    lines.append("    → 王国(输入)")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  4. Sephirot Engine Runner
# ═══════════════════════════════════════════════════════════════

def run_sephirot_simulate(sephirot_file: Path, features: dict) -> subprocess.CompletedProcess:
    """Run the pipeline via sephirot.exe simulate."""
    if not SEPHIROT_EXE.exists():
        raise FileNotFoundError(
            f"sephirot.exe not found at {SEPHIROT_EXE}\n"
            f"Run: cargo build --release in sephirot-rs/"
        )

    input_val = features["info_density"]
    kb_val = 1.0 + features["multidisciplinary"] * 2.0 + features["logic_strength"]
    target_val = 0.5 + features["sentiment"] * 0.5
    values = f"{input_val},{kb_val},{target_val}"

    print(f"\n{'='*60}")
    print(f"  Sephirot Engine — 16 Sephiroth Pipeline Execute")
    print(f"{'='*60}")
    print(f"  input={input_val:.4f}, kb={kb_val:.4f}, target={target_val:.4f}")

    result = subprocess.run(
        [str(SEPHIROT_EXE), "simulate", str(sephirot_file), "--values", values],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(SEPHIROT_EXE.parent),
    )

    return result


def run_sephirot_compile(sephirot_file: Path) -> Optional[Path]:
    """Compile .sephirot -> kernel.ptx"""
    if not SEPHIROT_EXE.exists():
        return None

    ptx_out = sephirot_file.with_suffix(".ptx")

    result = subprocess.run(
        [str(SEPHIROT_EXE), "compile", str(sephirot_file), "-t", "ptx", "--stdout"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(SEPHIROT_EXE.parent),
    )

    if result.returncode == 0:
        ptx_out.write_text(result.stdout, encoding="utf-8")
        print(f"  PTX compiled: {ptx_out.name} ({len(result.stdout)} bytes)")
        return ptx_out
    else:
        print(f"  PTX compile error: {result.stderr[:200]}")
        return None


# ═══════════════════════════════════════════════════════════════
#  5. Result Presenter
# ═══════════════════════════════════════════════════════════════

def present_results(conversation: list[dict], features: dict, sim_result: subprocess.CompletedProcess):
    """Present the complete analysis result."""

    print(f"\n\n{'='*60}")
    print(f"  DeepSeek-Sephirot Bridge — Results Report")
    print(f"  DeepSeek -> 16 Sephiroth Filter & Reshape")
    print(f"{'='*60}")

    # Conversation summary
    total_chars = sum(len(t["content"]) for t in conversation)
    assistant_turns = [t for t in conversation if t["role"] == "assistant"]
    print(f"\n  Conversation stats:")
    print(f"    Total turns: {len(conversation)}")
    print(f"    AI turns: {len(assistant_turns)}")
    print(f"    Total chars: {total_chars}")

    # Feature panel
    print(f"\n  Semantic feature vector:")
    print(f"    {'Feature':<20s} {'Value':>8s}  {'Sephiroth mapping'}")
    print(f"    {'-'*20} {'-'*8}  {'-'*20}")
    feature_mapping = [
        ("info_density", "Info density", "[0] Keter (Crown) data loading"),
        ("logic_strength", "Logic strength", "[2] Binah (Severity) threshold filter"),
        ("sentiment", "Sentiment index", "[4] Hesed (Mercy) weighted fusion"),
        ("abstraction", "Abstraction", "[5] Tiferet (Beauty) optimal integration"),
        ("depth", "Conversation depth", "[10] Ego self-attention"),
        ("multidisciplinary", "Multidisciplinarity", "[1] Chokhmah (Wisdom) knowledge retrieval"),
        ("structure", "Structure", "[12] Logic GEMM reasoning"),
        ("length_factor", "Length weight", "[8] Yesod (Foundation) global reduction"),
    ]
    for key, label, sephirot in feature_mapping:
        val = features[key]
        bar_len = int(val * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {label:<18s} {val:>7.4f}  {sephirot:<20s} {bar}")

    # Sephirot pipeline output
    print(f"\n  16-sephiroth pipeline execution:")
    print(f"  {'─'*56}")

    if sim_result.returncode == 0:
        for line in sim_result.stdout.strip().split("\n"):
            if line.strip():
                # Coloured printing
                if "[0]" in line:
                    print(f"  {line}")
                elif any(f"[{i}]" in line for i in range(1, 8)):
                    print(f"  {line}")
                elif "[8]" in line or "[15]" in line:
                    print(f"  >>> {line}")
                else:
                    print(f"  {line}")
    else:
        print(f"  ERROR: {sim_result.stderr[:300]}")

    print(f"  {'─'*56}")

    # Overall assessment
    print(f"\n  Overall assessment:")
    logic = features["logic_strength"]
    emotion = features["sentiment"]
    if logic > 0.5 and emotion > 0.3:
        verdict = "Reason and feeling in balance — the DeepSeek output is logically rigorous "
        verdict += "while retaining human warmth; after the 16-sephiroth filter it keeps the "
        verdict += "core high-information-density arguments."
    elif logic > 0.5:
        verdict = "Logic-dominant — the DeepSeek output is mainly rational analysis; "
        verdict += "the Binah (Severity) sephiroth filters out low-logic-density redundancy, "
        verdict += "but the Hesed (Mercy) weight is low."
    elif emotion > 0.3:
        verdict = "Emotion-driven — the DeepSeek output is strongly empathic, "
        verdict += "but logic strength is insufficient; consider deepening the rational analysis "
        verdict += "so it can pass the Binah (Severity) threshold filter."
    else:
        verdict = "Balanced mix — the DeepSeek output keeps a balance between reason and feeling."

    print(f"    {verdict}")

    # Final output
    final_output = features["info_density"] * (1.0 + features["multidisciplinary"]) * (0.5 + features["sentiment"] * 0.5)
    print(f"\n  >>> Final filter-reshape coefficient: {final_output:.4f}")
    print(f"      (info_density * multidisciplinary_boost * sentiment_calibration)")


# ═══════════════════════════════════════════════════════════════
#  6. Interactive Mode
# ═══════════════════════════════════════════════════════════════

def interactive_mode():
    """Interactive mode: continuous conversation + real-time sephirot filtering."""
    print(f"\n{'='*60}")
    print(f"  DeepSeek-Sephirot Interactive Mode")
    print(f"  Enter a topic for an in-depth discussion; type :quit to exit")
    print(f"  Type :compile to compile the most recently generated PTX")
    print(f"{'='*60}")

    session_id = int(time.time()) % 100000
    conversation_history = []
    last_sephirot_file = None

    while True:
        try:
            user_input = input(f"\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input == ":quit" or user_input == ":q":
            break
        if user_input == ":compile" and last_sephirot_file:
            ptx = run_sephirot_compile(last_sephirot_file)
            if ptx:
                print(f"  PTX saved: {ptx}")
            continue

        # Call DeepSeek
        try:
            response = call_deepseek(user_input, max_tokens=2048)
        except Exception as e:
            print(f"  API Error: {e}")
            continue

        # Accumulate the conversation
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        # Analyse features of the whole text
        full_text = "\n".join(t["content"] for t in conversation_history)
        features = analyze_text_features(full_text)

        # Generate the sephirot pipeline
        sephirot_code = generate_sephirot_pipeline(features, conversation_history)
        OUTPUT_DIR.mkdir(exist_ok=True)
        sephirot_file = OUTPUT_DIR / f"interactive_{session_id}.sephirot"
        sephirot_file.write_text(sephirot_code, encoding="utf-8")
        last_sephirot_file = sephirot_file

        # Execute the pipeline
        try:
            sim_result = run_sephirot_simulate(sephirot_file, features)
            present_results(conversation_history, features, sim_result)
        except FileNotFoundError as e:
            print(f"  Engine not found: {e}")
        except Exception as e:
            print(f"  Simulate error: {e}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-Sephirot Bridge: DeepSeek API -> 16 Sephiroth Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deepseek_sephirot_bridge.py "AI 意识的本质是什么？"
  python deepseek_sephirot_bridge.py --topic "量子纠缠" --depth deep --no-gpu
  python deepseek_sephirot_bridge.py --interactive
        """,
    )
    parser.add_argument("topic", nargs="?", help="topic to explore")
    parser.add_argument("--interactive", "-i", action="store_true", help="interactive mode")
    parser.add_argument("--depth", "-d", choices=["quick", "normal", "deep"], default="normal", help="conversation depth")
    parser.add_argument("--no-simulate", action="store_true", help="only generate .sephirot, do not run")
    parser.add_argument("--compile", action="store_true", help="also compile PTX")
    parser.add_argument("--output", "-o", type=str, default=None, help="output directory")

    args = parser.parse_args()

    if args.output:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.interactive:
        interactive_mode()
        return

    if not args.topic:
        parser.print_help()
        return

    print(f"\n{'#'*60}")
    print(f"  DeepSeek-Sephirot Bridge v1.0")
    print(f"  深祈双生桥 — DeepSeek API -> 16-sephiroth pipeline filter & reshape")
    print(f"{'#'*60}")

    # Step 1: DeepSeek multi-turn conversation
    conversation = deepseek_multi_turn(args.topic, depth=args.depth)

    # Step 2: merge and analyse the full text
    full_text = "\n".join(
        f"[{t['role']}]\n{t['content']}\n"
        for t in conversation
    )
    features = analyze_text_features(full_text)

    # Step 3: generate the .sephirot pipeline
    sephirot_code = generate_sephirot_pipeline(features, conversation)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^\w]', '_', args.topic)[:30]
    sephirot_file = OUTPUT_DIR / f"{timestamp}_{safe_topic}.sephirot"
    sephirot_file.write_text(sephirot_code, encoding="utf-8")

    print(f"\n  .sephirot pipeline: {sephirot_file}")
    print(f"  Size: {len(sephirot_code)} bytes")

    # Step 4: execute the pipeline
    if not args.no_simulate:
        try:
            sim_result = run_sephirot_simulate(sephirot_file, features)
            present_results(conversation, features, sim_result)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print(f"  Please build sephirot.exe first: cd sephirot-rs && cargo build --release")
    else:
        print("  (--no-simulate, skipping execution)")

    # Step 5: optionally compile PTX
    if args.compile:
        ptx = run_sephirot_compile(sephirot_file)
        if ptx:
            print(f"  PTX output: {ptx}")

    # Save the complete result
    result_file = OUTPUT_DIR / f"{timestamp}_{safe_topic}_report.json"
    report = {
        "topic": args.topic,
        "depth": args.depth,
        "features": {k: round(v, 6) for k, v in features.items()},
        "conversation_length": len(conversation),
        "total_chars": len(full_text),
        "sephirot_file": str(sephirot_file),
    }
    result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report saved: {result_file}")


if __name__ == "__main__":
    main()
