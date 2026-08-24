# -*- coding: utf-8 -*-
r"""
DeepSeek-Sephirot Bridge — local data source version
====================================================
Local files -> .sephirot pipeline -> sephirot.exe 16-sephiroth filter & reshape

Architecture:
  1. Local File Reader: read text data from local files
  2. Text Analyzer: decompose the text into semantic vectors
     (sentiment / logic / knowledge density / creativity)
  3. Pipeline Generator: generate a .sephirot pipeline in which each
     sephiroth corresponds to one transformation
  4. Sephirot Engine: run the 16-sephiroth pipeline via sephirot.exe simulate
  5. Result Presenter: present the filtered-and-reshaped result

Usage:
  # genericized from a machine-specific absolute path
  python deepseek_sephirot_bridge_local.py "path/to/your/file.docx"
  python deepseek_sephirot_bridge_local.py --dir "path/to/your/corpus" --batch
  python deepseek_sephirot_bridge_local.py --interactive
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
from typing import Optional, List
import docx
import glob

# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

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
#  1. Local File Reader
# ═══════════════════════════════════════════════════════════════

def read_docx_file(file_path: str) -> str:
    """Read the contents of a .docx file"""
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return ""

def read_text_file(file_path: str) -> str:
    """Read the contents of a .txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return ""

def read_file(file_path: str) -> str:
    """Read a file according to its extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.docx':
        return read_docx_file(file_path)
    elif ext in ['.txt', '.md', '.json', '.yaml', '.yml', '.py']:
        return read_text_file(file_path)
    else:
        print(f"  Unsupported file type: {ext}")
        return ""

def load_local_corpus(file_path: str, max_chars: int = 100000) -> str:
    """
    Load a local corpus file, capped at a maximum character count
    """
    print(f"\n{'='*60}")
    print(f"  Local File Reader")
    print(f"{'='*60}")
    print(f"  Loading: {file_path}")
    
    content = read_file(file_path)
    if not content:
        raise ValueError(f"Failed to read file: {file_path}")
    
    # Limit the length
    if len(content) > max_chars:
        content = content[:max_chars]
        print(f"  Truncated to {max_chars} chars")
    
    print(f"  Loaded: {len(content)} chars, {len(content.splitlines())} lines")
    return content

def load_local_corpus_batch(dir_path: str, file_pattern: str = "*.docx", max_files: int = 10) -> List[dict]:
    """
    Batch-load files from a directory
    Returns: [{"file_path": "...", "content": "...", "file_name": "..."}, ...]
    """
    print(f"\n{'='*60}")
    print(f"  Local File Batch Reader")
    print(f"{'='*60}")
    print(f"  Scanning: {dir_path}")
    
    files = []
    for ext in ['*.docx', '*.txt', '*.md']:
        files.extend(glob.glob(os.path.join(dir_path, ext)))
    
    if not files:
        raise ValueError(f"No files found in {dir_path}")
    
    results = []
    for i, file_path in enumerate(files[:max_files]):
        print(f"  [{i+1}/{len(files[:max_files])}] Reading: {os.path.basename(file_path)}")
        content = read_file(file_path)
        if content:
            results.append({
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "content": content[:50000]  # cap each file at 50k chars
            })
    
    print(f"  Total loaded: {len(results)} files")
    return results

# ═══════════════════════════════════════════════════════════════
#  2. Text Analyzer — text -> semantic feature vector
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

def generate_sephirot_pipeline(features: dict, source_info: dict) -> str:
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
    lines.append("# Local-Sephirot Bridge Generated Pipeline")
    lines.append(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# Source: {source_info.get('source', 'Local File')}")
    lines.append(f"# File: {source_info.get('file_name', 'Unknown')}")
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
    lines.append("# [0] Keter - Crown: Load local file content")
    lines.append("管道 main:")

    # Divine pillar: sephiroth 0-7
    lines.append("    # --- Divine Pillar ---")
    lines.append("    # [0] Keter: Raw input from local file")
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

def present_results(features: dict, sim_result: subprocess.CompletedProcess, source_info: dict):
    """Present the complete analysis result."""

    print(f"\n\n{'='*60}")
    print(f"  Local-Sephirot Bridge — Results Report")
    print(f"  Local File -> 16 Sephiroth Filter & Reshape")
    print(f"{'='*60}")

    # Source file info
    print(f"\n  Source file info:")
    print(f"    File: {source_info.get('file_name', 'Unknown')}")
    print(f"    Path: {source_info.get('file_path', 'Unknown')}")
    print(f"    Char count: {source_info.get('char_count', 0)}")

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
        verdict = "Reason and feeling in balance — the text is logically rigorous "
        verdict += "while retaining human warmth; after the 16-sephiroth filter it keeps "
        verdict += "the core high-information-density arguments."
    elif logic > 0.5:
        verdict = "Logic-dominant — the text is mainly rational analysis; "
        verdict += "the Binah (Severity) sephiroth filters out low-logic-density redundancy, "
        verdict += "but the Hesed (Mercy) weight is low."
    elif emotion > 0.3:
        verdict = "Emotion-driven — the text is strongly empathic, "
        verdict += "but logic strength is insufficient; consider deepening the rational analysis "
        verdict += "so it can pass the Binah (Severity) threshold filter."
    else:
        verdict = "Balanced mix — the text keeps a balance between reason and feeling."

    print(f"    {verdict}")

    # Final output
    final_output = features["info_density"] * (1.0 + features["multidisciplinary"]) * (0.5 + features["sentiment"] * 0.5)
    print(f"\n  >>> Final filter-reshape coefficient: {final_output:.4f}")
    print(f"      (info_density * multidisciplinary_boost * sentiment_calibration)")
    
    # Data quality suggestions
    print(f"\n  >>> Data quality suggestions:")
    if final_output > 0.7:
        print(f"      ✓ High-quality data, suitable for SFT/RLHF training")
    elif final_output > 0.4:
        print(f"      ⚠ Medium-quality data, needs further cleaning or mixed usage")
    else:
        print(f"      ✗ Low-quality data, consider filtering out or discarding")


# ═══════════════════════════════════════════════════════════════
#  6. Batch Processing Mode
# ═══════════════════════════════════════════════════════════════

def batch_process_mode(dir_path: str, output_dir: str = None):
    """Batch processing mode: process every file in the directory"""
    print(f"\n{'='*60}")
    print(f"  Local-Sephirot Batch Processing Mode")
    print(f"  Processing directory: {dir_path}")
    print(f"{'='*60}")
    
    if output_dir:
        OUTPUT_DIR = Path(output_dir)
    else:
        OUTPUT_DIR = Path(__file__).parent / "bridge_output_batch"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load all files
    files = load_local_corpus_batch(dir_path, max_files=50)
    
    results = []
    for i, file_info in enumerate(files):
        print(f"\n{'─'*40}")
        print(f"  Processing file {i+1}/{len(files)}: {file_info['file_name']}")
        print(f"{'─'*40}")
        
        try:
            # Analyse features
            features = analyze_text_features(file_info['content'])
            
            # Generate the pipeline
            source_info = {
                "source": "Local File",
                "file_name": file_info['file_name'],
                "file_path": file_info['file_path'],
                "char_count": len(file_info['content'])
            }
            sephirot_code = generate_sephirot_pipeline(features, source_info)
            
            # Save the pipeline file
            safe_name = re.sub(r'[^\w]', '_', file_info['file_name'])[:50]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            sephirot_file = OUTPUT_DIR / f"{timestamp}_{safe_name}.sephirot"
            sephirot_file.write_text(sephirot_code, encoding="utf-8")
            
            # Execute the pipeline
            try:
                sim_result = run_sephirot_simulate(sephirot_file, features)
                present_results(features, sim_result, source_info)
                
                # Compute the final coefficient
                final_output = features["info_density"] * (1.0 + features["multidisciplinary"]) * (0.5 + features["sentiment"] * 0.5)
                
                results.append({
                    "file_name": file_info['file_name'],
                    "file_path": file_info['file_path'],
                    "char_count": len(file_info['content']),
                    "features": features,
                    "sephirot_file": str(sephirot_file),
                    "final_output": final_output,
                    "quality": "high" if final_output > 0.7 else "medium" if final_output > 0.4 else "low"
                })
                
                # Save the result report
                report_file = OUTPUT_DIR / f"{timestamp}_{safe_name}_report.json"
                report = {
                    "file_info": source_info,
                    "features": {k: round(v, 6) for k, v in features.items()},
                    "final_output": round(final_output, 6),
                    "sephirot_file": str(sephirot_file),
                    "timestamp": timestamp
                }
                report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                
            except FileNotFoundError as e:
                print(f"  Engine not found: {e}")
                results.append({
                    "file_name": file_info['file_name'],
                    "error": "sephirot.exe not found"
                })
            except Exception as e:
                print(f"  Simulate error: {e}")
                results.append({
                    "file_name": file_info['file_name'],
                    "error": str(e)
                })
                
        except Exception as e:
            print(f"  Processing error: {e}")
            results.append({
                "file_name": file_info['file_name'],
                "error": str(e)
            })
    
    # Generate the batch report
    print(f"\n\n{'='*60}")
    print(f"  Batch Processing Summary")
    print(f"{'='*60}")
    
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    print(f"  Total files: {len(files)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if successful:
        print(f"\n  Quality Distribution:")
        high = [r for r in successful if r.get("quality") == "high"]
        medium = [r for r in successful if r.get("quality") == "medium"]
        low = [r for r in successful if r.get("quality") == "low"]
        
        print(f"    High quality (>0.7): {len(high)} files")
        print(f"    Medium quality (0.4-0.7): {len(medium)} files")
        print(f"    Low quality (<0.4): {len(low)} files")
        
        # Recommend high-quality files
        if high:
            print(f"\n  Recommended High-Quality Files:")
            for r in high[:5]:  # show the first 5
                print(f"    ✓ {r['file_name']} (score: {r['final_output']:.3f})")
    
    # Save the batch report
    summary_file = OUTPUT_DIR / f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": dir_path,
        "output_dir": str(OUTPUT_DIR),
        "total_files": len(files),
        "successful": len(successful),
        "failed": len(failed),
        "results": results
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Summary saved: {summary_file}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Local-Sephirot Bridge: Local Files -> 16 Sephiroth Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # genericized from a machine-specific absolute path
  python deepseek_sephirot_bridge_local.py "path/to/your/file.docx"
  python deepseek_sephirot_bridge_local.py --dir "path/to/your/corpus" --batch
  python deepseek_sephirot_bridge_local.py --file "path/to/file.docx" --no-simulate
        """,
    )
    parser.add_argument("file_path", nargs="?", help="path to a single file")
    parser.add_argument("--dir", "-d", type=str, help="directory path (batch processing)")
    parser.add_argument("--batch", "-b", action="store_true", help="batch processing mode")
    parser.add_argument("--no-simulate", action="store_true", help="only generate .sephirot, do not run")
    parser.add_argument("--compile", action="store_true", help="also compile PTX")
    parser.add_argument("--output", "-o", type=str, default=None, help="output directory")
    parser.add_argument("--max-files", type=int, default=50, help="max number of files for batch processing")

    args = parser.parse_args()

    if args.output:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Local-Sephirot Bridge v1.0")
    print(f"  Local data source -> 16-sephiroth pipeline filter & reshape")
    print(f"{'#'*60}")

    # Batch processing mode
    if args.batch and args.dir:
        batch_process_mode(args.dir, args.output)
        return
    
    # Directory batch processing (without the --batch flag)
    if args.dir and not args.file_path:
        batch_process_mode(args.dir, args.output)
        return

    # Single file processing
    if not args.file_path:
        parser.print_help()
        return

    # Step 1: load the local file
    content = load_local_corpus(args.file_path)
    if not content:
        print(f"  ERROR: Failed to load file: {args.file_path}")
        return

    # Step 2: analyse features
    features = analyze_text_features(content)

    # Step 3: generate the .sephirot pipeline
    source_info = {
        "source": "Local File",
        "file_name": os.path.basename(args.file_path),
        "file_path": args.file_path,
        "char_count": len(content)
    }
    sephirot_code = generate_sephirot_pipeline(features, source_info)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r'[^\w]', '_', os.path.basename(args.file_path))[:50]
    sephirot_file = OUTPUT_DIR / f"{timestamp}_{safe_name}.sephirot"
    sephirot_file.write_text(sephirot_code, encoding="utf-8")

    print(f"\n  .sephirot pipeline: {sephirot_file}")
    print(f"  Size: {len(sephirot_code)} bytes")

    # Step 4: execute the pipeline
    if not args.no_simulate:
        try:
            sim_result = run_sephirot_simulate(sephirot_file, features)
            present_results(features, sim_result, source_info)
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
    result_file = OUTPUT_DIR / f"{timestamp}_{safe_name}_report.json"
    report = {
        "file_info": source_info,
        "features": {k: round(v, 6) for k, v in features.items()},
        "sephirot_file": str(sephirot_file),
        "timestamp": timestamp
    }
    result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report saved: {result_file}")


if __name__ == "__main__":
    main()
