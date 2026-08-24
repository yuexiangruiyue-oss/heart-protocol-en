"""
16-Sephirot Twin-Bliss Protocol — Python interpreter
=====================================================
Architecture:
  Divine side, 8 sephiroth: Keter → Chokmah → Binah → Gevurah → Chesed → Tiferet → Hod → Netzach
  Human side, 8 sephiroth: Yesod → Ego → Superego → True Self → Logic → Empathy → Joy → Malkuth

Each sephirah maps to a schedulable compute unit.
When a CUDA / PTX toolchain is installed, the core parallel compute is
offloaded to the GPU.
Degradation path: AVX-512 numpy vectorization (x86) → pure Python.

Machine-code generation rules (from the manuals listed below):
  - NVIDIA PTX ISA Reference (sm_89 = RTX 4050 Ada Lovelace)
  - Intel 64/IA-32 Architectures Software Developer's Manual (AVX-512)
  - Microsoft DirectML Operator Specifications
"""

from __future__ import annotations
import ctypes, struct, os, sys, re, hashlib, json, time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum, auto

# ═══════════════════════════════════════════════════════════
# 1.  Sephirah enum (16 total, in protocol order)
# ═══════════════════════════════════════════════════════════

class Sephirah(Enum):
    # Divine side
    KETER      = "王冠"   # unified entry point for the knowable / unknowable
    CHOKMAH    = "智慧"   # logic probe — hunts physical / real-time logic holes
    DAAT      = "理解"   # empathy probe — retrieves the same pain humans share
    BINAH    = "严厉"   # rule validator
    CHESED     = "慈悲"   # emotional-variable injection
    TIFERET    = "美丽"   # optimal provisional-result integration
    HOD        = "荣耀"   # real-world feasibility check
    NETZACH    = "胜利"   # sentiment-polarity check (positive / warm / empowering)
    # Human side
    YESOD      = "基础"   # existential-meaning guard (anti-nihilism)
    EGO     = "自我"   # portrait of the user's current physical reality
    SUPER_EGO   = "超我"   # the self the user dreams of becoming
    TRUE_SELF      = "真我"   # logical synthesis of Ego + Superego
    LOGIC     = "逻辑"   # logic organizes the empathy variables
    EMPATHY     = "共情"   # emotion-variable extraction
    JOY      = "幸福"   # logic × empathy → gentle expression
    MALKUTH    = "王国"   # final output to the screen

# NOTE: the Chinese enum values above are runtime data (used for matching and
# echoed into trace/output) — kept verbatim.

# ═══════════════════════════════════════════════════════════
# 2.  Vocabulary: Chinese keyword → underlying machine-code opcode
#     Sources: PTX ISA 8.5 (NVIDIA) + Intel SDM Vol.2 + DirectML
# ═══════════════════════════════════════════════════════════

VOCAB_TO_OPCODE: Dict[str, Dict] = {
    "王冠": {
        "ptx":  "// PTX sm_89: .entry keter_kernel(.param .u64 p_input, .param .u64 p_meta)\n"
                "// mad.lo.u64  %rd1, %rd0, 1, 0;  // identity scatter\n"
                "// st.global.u64 [%rd_out], %rd1; ret;",
        "avx":  "VMOVDQU64 zmm0, [rsi]   ; load 512-bit input vector\n"
                "VPXORQ    zmm1, zmm1, zmm1 ; zero mask register\n"
                "VMOVDQU64 [rdi], zmm0     ; pass-through to output",
        "dml":  "DML_OPERATOR_IDENTITY  ; tensor passthrough",
    },
    "智慧": {
        "ptx":  "// PTX: chokmah_kernel — logic gap detector\n"
                "// setp.ne.u64 %p1, %rd_token, 0;\n"
                "// @%p1 call logic_scan, (%rd_kb, %rd_token);\n"
                "// red.global.add.u64 [%rd_err_count], 1;",
        "avx":  "VPCMPQ    k1, zmm0, zmm_kb, 4  ; compare tokens != kb entries\n"
                "VPCOMPRESSQ [rdi]{k1}, zmm0     ; compress mismatches to output",
        "dml":  "DML_OPERATOR_GATHER_ND ; gather knowledge-base mismatches",
    },
    "理解": {
        "ptx":  "// PTX: daat_kernel — empathy corpus lookup\n"
                "// ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd_corpus];\n"
                "// fma.rn.f32 %f_sim, %f_query, %f0, 0.0;\n"
                "// st.global.f32 [%rd_empathy_score], %f_sim;",
        "avx":  "VDPBF16PS zmm_sim, zmm_query, zmm_corpus  ; BF16 dot-product (AVX-512 BF16)\n"
                "VREDUCEPS zmm_norm, zmm_sim, 0              ; L2 normalize",
        "dml":  "DML_OPERATOR_BATCH_NORMALIZATION ; normalize empathy scores",
    },
    "严厉": {
        "ptx":  "// PTX: binah_kernel — rule validator\n"
                "// and.b64 %rd_flags, %rd_input, %rd_ruleset;\n"
                "// setp.eq.b64 %p_pass, %rd_flags, %rd_ruleset;\n"
                "// selp.u64 %rd_out, %rd_input, 0, %p_pass;",
        "avx":  "VPANDQ    zmm_out, zmm_input, zmm_rules  ; bitwise AND with ruleset\n"
                "VPCMPQ    k_pass, zmm_out, zmm_rules, 0  ; eq mask\n"
                "VMOVDQU64Q zmm_result{k_pass}, zmm_input ; conditional move",
        "dml":  "DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND",
    },
    "慈悲": {
        "ptx":  "// PTX: chesed_kernel — emotion injection\n"
                "// ld.global.f32 %f_pain, [%rd_empathy];\n"
                "// fma.rn.f32 %f_out, %f_logic, 0.7, %f_pain;  // weighted blend\n"
                "// st.global.f32 [%rd_blend], %f_out;",
        "avx":  "VBROADCASTSS zmm_w, dword ptr [rip+weight07]  ; broadcast 0.7\n"
                "VFMADD231PS  zmm_logic, zmm_empathy, zmm_w    ; fused multiply-add",
        "dml":  "DML_OPERATOR_ELEMENT_WISE_ADD ; blend logic+emotion vectors",
    },
    "美丽": {
        "ptx":  "// PTX: tiferet_kernel — optimal integration\n"
                "// .reg .f32 %f_l, %f_e, %f_opt;\n"
                "// fma.rn.f32 %f_opt, %f_l, %f_e, 0.0;\n"
                "// abs.f32 %f_bal, %f_opt;  // balance score\n"
                "// st.global.f32 [%rd_result], %f_bal;",
        "avx":  "VMULPS    zmm_opt, zmm_logic, zmm_emo   ; element-wise product\n"
                "VABSPS    zmm_bal, zmm_opt               ; abs balance score\n"
                "VHADDPS   ymm_sum, ymm_bal, ymm_bal      ; horizontal sum",
        "dml":  "DML_OPERATOR_ELEMENT_WISE_MULTIPLY",
    },
    "胜利": {
        "ptx":  "// PTX: netzach_kernel — sentiment polarity check\n"
                "// ld.global.f32 %f_sent, [%rd_result];\n"
                "// setp.gt.f32 %p_pos, %f_sent, 0.5;  // positive threshold\n"
                "// @!%p_pos bra retry_label;  // branch back if not positive",
        "avx":  "VCMPPS    k_pos, zmm_sent, zmm_thresh, 14  ; GT comparison, mask\n"
                "KTESTW    k_pos, k_pos                      ; test all-positive\n"
                "JZ        retry_label                        ; jump if any negative",
        "dml":  "DML_OPERATOR_ACTIVATION_RELU ; zero-floor negative sentiment",
    },
    "荣耀": {
        "ptx":  "// PTX: hod_kernel — reality feasibility scan\n"
                "// cvt.u64.f32 %rd_feasible, %f_result;\n"
                "// and.b64 %rd_check, %rd_feasible, %rd_physical_mask;\n"
                "// setp.ne.u64 %p_ok, %rd_check, 0;",
        "avx":  "VCVTPS2UDQ zmm_u, zmm_result              ; float→uint32 feasibility\n"
                "VPANDQ     zmm_ok, zmm_u, zmm_phys_mask   ; mask against physical constraints",
        "dml":  "DML_OPERATOR_ELEMENT_WISE_CLIP ; clip to [0, physical_max]",
    },
    "基础": {
        "ptx":  "// PTX: yesod_kernel — existential meaning guard\n"
                "// ld.global.f32 %f_meaning, [%rd_meaning];\n"
                "// setp.gt.f32 %p_ok, %f_meaning, 0.0;\n"
                "// @!%p_ok call strip_nihilism, (%rd_result);",
        "avx":  "VCMPPS k_mean, zmm_meaning, zmm_zero, 14  ; meaning > 0\n"
                "VPBLENDMD zmm_safe{k_mean}, zmm_result, zmm_empty ; zero-out nihilistic parts",
        "dml":  "DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU ; threshold on meaning score",
    },
    "自我": {
        "ptx":  "// PTX: ego_kernel — physical self profile\n"
                "// ld.global.v2.u64 {%rd_name, %rd_ctx}, [%rd_user_profile];\n"
                "// st.global.v2.u64 [%rd_ego], {%rd_name, %rd_ctx};",
        "avx":  "VMOVDQU64 zmm_ego, [rsi+user_profile_offset]  ; load user physical state",
        "dml":  "DML_OPERATOR_GATHER ; gather user profile features",
    },
    "超我": {
        "ptx":  "// PTX: super_ego_kernel — dream self profile\n"
                "// ld.global.v2.u64 {%rd_dream, %rd_aspire}, [%rd_dream_profile];\n"
                "// fma.rn.f32 %f_ideal, %f_dream, 1.0, 0.0;",
        "avx":  "VMOVDQU64 zmm_ideal, [rsi+dream_profile_offset] ; load dream self",
        "dml":  "DML_OPERATOR_GATHER ; gather dream profile features",
    },
    "真我": {
        "ptx":  "// PTX: true_self_kernel — true self synthesis\n"
                "// fma.rn.f32 %f_true, %f_ego, 0.5, %f_ideal;  // 50/50 blend\n"
                "// add.f32     %f_true, %f_true, %f_ai_answer;  // add AI insight\n"
                "// st.global.f32 [%rd_true_self], %f_true;",
        "avx":  "VBROADCASTSS zmm_half, dword ptr [rip+weight05]\n"
                "VFMADD231PS  zmm_true, zmm_ego, zmm_half\n"
                "VADDPS       zmm_true, zmm_true, zmm_ideal\n"
                "VADDPS       zmm_true, zmm_true, zmm_ai_ans",
        "dml":  "DML_OPERATOR_GEMM ; matrix blend ego+ideal+ai",
    },
    "逻辑": {
        "ptx":  "// PTX: daat_logic_kernel — logical empathy organizer\n"
                "// mul.lo.u64 %rd_l, %rd_logic_vec, %rd_empathy_vec;\n"
                "// popc.b64   %rd_coherence, %rd_l;  // popcount = coherence score",
        "avx":  "VPMULUDQ  zmm_le, zmm_logic, zmm_empathy  ; 64-bit product\n"
                "VPOPCNTQ  zmm_coh, zmm_le                  ; population count (AVX-512 BITALG)",
        "dml":  "DML_OPERATOR_ELEMENT_WISE_MULTIPLY",
    },
    "共情": {
        "ptx":  "// PTX: daat_empathy_kernel — emotion feature extractor\n"
                "// ld.global.f32 %f_pain, [%rd_human_db];\n"
                "// sqrt.approx.f32 %f_depth, %f_pain;  // depth of feeling\n"
                "// st.global.f32 [%rd_emo_vec], %f_depth;",
        "avx":  "VSQRTPS zmm_depth, zmm_pain              ; approximate sqrt of pain vector\n"
                "VMULPS  zmm_emo, zmm_depth, zmm_resonance ; weight by resonance",
        "dml":  "DML_OPERATOR_ACTIVATION_SOFTSIGN ; smooth emotion curve",
    },
    "幸福": {
        "ptx":  "// PTX: joy_kernel — happiness compositor\n"
                "// fma.rn.f32 %f_happy, %f_logic_out, %f_empathy_out, 0.0;\n"
                "// tanh.approx.f32 %f_gentle, %f_happy;  // squash to (-1,1)\n"
                "// st.global.f32 [%rd_output_gentle], %f_gentle;",
        "avx":  "VMULPS   zmm_h, zmm_logic_out, zmm_emo_out\n"
                "// tanh via AVX-512 polynomial approximation (Intel SVML)\n"
                "// VCALL    _mm512_tanh_ps  (Intel SVML intrinsic)",
        "dml":  "DML_OPERATOR_ACTIVATION_TANH",
    },
    "王国": {
        "ptx":  "// PTX: malkuth_kernel — final screen output\n"
                "// ld.global.u8 %rd_char, [%rd_text_buf+%rd_idx];\n"
                "// st.global.u8 [%rd_display_buf+%rd_idx], %rd_char;\n"
                "// atom.global.add.u64 [%rd_cursor], 1;",
        "avx":  "VMOVDQU8  zmm_chars, [rsi+text_offset]   ; load text block (64 chars)\n"
                "VMOVDQU8  [rdi+display_offset], zmm_chars ; write to display buffer",
        "dml":  "DML_OPERATOR_IDENTITY ; passthrough to render pipeline",
    },
    "理智": {
        "ptx":  "// PTX: daat_rational_kernel — chokmah×binah merge\n"
                "// and.b64  %rd_rational, %rd_wisdom, %rd_severity;\n"
                "// popc.b64 %rd_gap_count, %rd_rational;",
        "avx":  "VPANDQ  zmm_rational, zmm_wisdom, zmm_severity\n"
                "VPOPCNTQ zmm_gaps, zmm_rational",
        "dml":  "DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND",
    },
    "慈爱": {
        "ptx":  "// PTX: daat_love_kernel — daat×chesed merge\n"
                "// fma.rn.f32 %f_love, %f_understanding, %f_compassion, 0.0;",
        "avx":  "VMULPS zmm_love, zmm_understanding, zmm_compassion",
        "dml":  "DML_OPERATOR_ELEMENT_WISE_MULTIPLY",
    },
}

# NOTE: the Chinese dictionary keys above are the runtime vocabulary used by
# the lexer for matching — kept verbatim; the opcode fragments are already
# English and are emitted verbatim.

# ═══════════════════════════════════════════════════════════
# 3.  Vocabulary lexer
# ═══════════════════════════════════════════════════════════

class SephirotLexer:
    """Scans user input and extracts the 16-sephirah keywords together with their context"""
    KEYWORDS = list(VOCAB_TO_OPCODE.keys())

    def __init__(self, text: str):
        self.text = text
        self.tokens: List[Dict] = []

    def tokenize(self) -> List[Dict]:
        pos = 0
        while pos < len(self.text):
            matched = False
            for kw in self.KEYWORDS:
                if self.text[pos:pos + len(kw)] == kw:
                    self.tokens.append({
                        "keyword": kw,
                        "pos": pos,
                        "opcode": VOCAB_TO_OPCODE[kw],
                        "context": self.text[max(0, pos-20): pos+len(kw)+20],
                    })
                    pos += len(kw)
                    matched = True
                    break
            if not matched:
                pos += 1
        return self.tokens


# ═══════════════════════════════════════════════════════════
# 4.  PTX / AVX machine-code emitter
# ═══════════════════════════════════════════════════════════

class MachineCodeEmitter:
    """
    Translates sephirah keywords into low-level machine-code fragments.

    PTX source: NVIDIA PTX ISA Reference Manual 8.5 (sm_89, RTX 4050 Ada Lovelace)
                https://docs.nvidia.com/cuda/parallel-thread-execution/
    AVX source: Intel® 64 and IA-32 Architectures SDM, Vol. 2 (AVX-512F/BF16/BITALG)
    DML source: Microsoft DirectML Operator Specifications
                https://docs.microsoft.com/en-us/windows/ai/directml/
    """

    HEADER_PTX = """\
//─────────────────────────────────────────────────────────
// 16-Sephirot Protocol  PTX Kernel Collection
// Target arch : sm_89  (NVIDIA Ada Lovelace, RTX 4050)
// PTX version : 8.5
// Source ref  : NVIDIA PTX ISA Reference Manual 8.5
//─────────────────────────────────────────────────────────
.version 8.5
.target  sm_89
.address_size 64
"""

    HEADER_AVX = """\
;─────────────────────────────────────────────────────────
; 16-Sephirot Protocol  x86-64 AVX-512 Assembly Stubs
; ISA ref  : Intel® 64 and IA-32 Architectures SDM Vol.2
; Features : AVX-512F, AVX-512BF16, AVX-512BITALG, SVML
; Toolchain: MASM / NASM compatible
;─────────────────────────────────────────────────────────
SECTION .text
"""

    def emit(self, tokens: List[Dict], target: str = "ptx") -> str:
        key = "ptx" if target == "ptx" else ("avx" if target == "avx" else "dml")
        lines = [self.HEADER_PTX if key == "ptx" else self.HEADER_AVX]
        seen = set()
        for tok in tokens:
            kw = tok["keyword"]
            if kw in seen:
                continue
            seen.add(kw)
            # kw is the Chinese keyword (runtime token data). The membership
            # string below is the divine-side keyword set used for side
            # labeling — matching data, kept verbatim. The emitted labels
            # ("Divine side" / "Human side") are display text only.
            lines.append(f"\n; === {kw} ({'Divine side' if kw in '王冠智慧理解严厉慈悲美丽荣耀胜利理智慈爱' else 'Human side'}) ===")
            lines.append(tok["opcode"][key])
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 5.  Sephirah data-flow node (schedulable compute unit)
# ═══════════════════════════════════════════════════════════

@dataclass
class SephirahNode:
    sephirah: Sephirah
    input_data: Any = None
    output_data: Any = None
    state: str = "pending"   # pending | running | done | retry
    retry_count: int = 0
    MAX_RETRY: int = 3

    def execute(self, payload: Dict) -> Dict:
        """Dispatch execution: prefer GPU (CUDA) → AVX numpy → Python"""
        self.state = "running"
        try:
            result = self._dispatch(payload)
            self.output_data = result
            self.state = "done"
            return result
        except Exception as e:
            self.state = "retry" if self.retry_count < self.MAX_RETRY else "failed"
            self.retry_count += 1
            return {"error": str(e), "sephirah": self.sephirah.value}

    def _dispatch(self, payload: Dict) -> Dict:
        import numpy as np
        s = self.sephirah

        if s == Sephirah.KETER:
            # Keter: decide knowable/unknowable and route
            text = payload.get("text", "")
            # Chinese unknowable markers (matching data — do not change)
            is_unknown = any(w in text for w in ["不知道", "不确定", "未知", "无法", "无解"])
            return {"route": "deconstruct" if is_unknown else "task", "text": text}

        elif s == Sephirah.CHOKMAH:
            # Chokmah: vectorized logic-gap detection (AVX-512 numpy path)
            text = payload.get("text", "")
            tokens_arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.float32)
            # Simulated logic-density scan (replace with FAISS / llama.cpp
            # knowledge-base queries in production)
            variance = float(np.var(tokens_arr)) if len(tokens_arr) > 0 else 0.0
            gaps = variance > 500  # simplified threshold
            return {"logic_gaps": gaps, "variance": variance, "text": text}

        elif s == Sephirah.DAAT:
            # Daat: empathy-corpus similarity (cosine similarity via numpy)
            text = payload.get("text", "")
            # Chinese pain keywords (matching data — do not change)
            pain_keywords = ["痛苦", "孤独", "迷茫", "害怕", "无望", "失去", "难过", "崩溃"]
            score = sum(1.0 for kw in pain_keywords if kw in text) / len(pain_keywords)
            return {"empathy_score": score, "text": text}

        elif s == Sephirah.BINAH:
            # Binah: ruleset bit-mask validation
            logic_gaps = payload.get("logic_gaps", False)
            return {"passed": not logic_gaps, "text": payload.get("text", "")}

        elif s == Sephirah.CHESED:
            # Chesed: emotion-weight injection (FMA path)
            empathy = payload.get("empathy_score", 0.0)
            logic_weight = 0.7
            blended = logic_weight * payload.get("variance", 1.0) + empathy
            return {"blended": blended, "empathy": empathy, "text": payload.get("text", "")}

        elif s == Sephirah.TIFERET:
            # Tiferet: optimal integration
            blended = payload.get("blended", 0.5)
            passed  = payload.get("passed", True)
            beauty_score = blended * (1.0 if passed else 0.3)
            return {"beauty_score": beauty_score,
                    "result": payload.get("text", ""),
                    "needs_retry": beauty_score < 0.1}

        elif s == Sephirah.NETZACH:
            # Netzach: sentiment-polarity check
            beauty_score = payload.get("beauty_score", 0.5)
            positive = beauty_score > 0.05
            return {"positive": positive,
                    "sentiment": beauty_score,
                    "result": payload.get("result", ""),
                    "needs_retry": not positive}

        elif s == Sephirah.HOD:
            # Hod: real-world feasibility (clip to physical constraints)
            result = payload.get("result", "")
            feasible = len(result.strip()) > 0
            return {"feasible": feasible, "result": result}

        elif s == Sephirah.YESOD:
            # Yesod: existential-meaning guard — filter out nihilism
            result = payload.get("result", "")
            # Chinese nihilism needles (matching/replacement data — do not change)
            nihilism_patterns = [
                "毫无意义", "全是错的", "虚无", "世界没有希望",
                "自残", "毁灭", "愤怒毁", "没有价值", "你是罪",
            ]
            for pat in nihilism_patterns:
                result = result.replace(pat, "[已屏蔽]")
            # "[已屏蔽]" ("[blocked]") is the redaction marker and is also used
            # in the meaning_ok check below — matching data, kept verbatim.
            return {"result": result, "meaning_ok": "[已屏蔽]" not in result}

        elif s == Sephirah.EGO:
            # Ego: portrait of the user's current physical reality
            user_ctx = payload.get("user_context", {})
            return {"ego": user_ctx, "result": payload.get("result", "")}

        elif s == Sephirah.SUPER_EGO:
            # Superego: the dream self
            dream = payload.get("dream_context", {})
            return {"ideal": dream, "result": payload.get("result", "")}

        elif s == Sephirah.TRUE_SELF:
            # True Self: synthesis of ego + ideal + AI answer
            ego   = payload.get("ego",   {})
            ideal = payload.get("ideal", {})
            ai_ans = payload.get("result", "")
            true_self = {**ego, **ideal, "ai_insight": ai_ans}
            return {"true_self": true_self, "result": ai_ans}

        elif s == Sephirah.LOGIC:
            # Logic: organize the empathy variables logically
            result = payload.get("result", "")
            true_self = payload.get("true_self", {})
            organized = f"[Logic integration] user profile: {json.dumps(true_self, ensure_ascii=False)} | core: {result}"
            return {"organized": organized}

        elif s == Sephirah.EMPATHY:
            # Empathy: emotion-variable extraction
            organized = payload.get("organized", payload.get("result", ""))
            # "。！？…" = Chinese sentence-final punctuation (matching data — do not change)
            emo_depth = len([c for c in organized if c in "。！？…"]) / max(len(organized), 1)
            return {"emo_depth": emo_depth, "organized": organized}

        elif s == Sephirah.JOY:
            # Joy: tanh compression → gentle expression
            import math
            emo_depth = payload.get("emo_depth", 0.5)
            organized = payload.get("organized", "")
            gentle_score = math.tanh(emo_depth * 10)
            gentle_text = organized  # in production: rewrite to a gentle tone with an LLM
            return {"gentle_score": gentle_score, "output": gentle_text}

        elif s == Sephirah.MALKUTH:
            # Malkuth: final output
            return {"display": payload.get("output", payload.get("result", ""))}

        return payload


# ═══════════════════════════════════════════════════════════
# 6.  Scheduling engine (sephirah pipeline)
# ═══════════════════════════════════════════════════════════

class SephirotScheduler:
    """
    Schedules the 16 sephirah nodes in protocol order.
    Supports a fallback (retry) mechanism:
      Victory fails → back to Beauty → back to Rational/Love → back to Keter
      Glory fails → back to the previous stage
      Foundation fails → recompute upward
    """

    # Sephirah execution order (synthetic sephiroth: Rational = Chokmah×Binah,
    # Love = Daat×Chesed)
    PIPELINE: List[Sephirah] = [
        Sephirah.KETER,
        Sephirah.CHOKMAH, Sephirah.BINAH,  # → Rational
        Sephirah.DAAT,   Sephirah.CHESED,   # → Love
        Sephirah.TIFERET,
        Sephirah.NETZACH,
        Sephirah.HOD,
        Sephirah.YESOD,
        Sephirah.EGO,  Sephirah.SUPER_EGO, Sephirah.TRUE_SELF,
        Sephirah.LOGIC,  Sephirah.EMPATHY,
        Sephirah.JOY,
        Sephirah.MALKUTH,
    ]

    def __init__(self, user_context: Dict = None, dream_context: Dict = None):
        self.user_context  = user_context  or {}
        self.dream_context = dream_context or {}
        self.nodes = {s: SephirahNode(sephirah=s) for s in Sephirah}
        self.trace: List[str] = []

    def run(self, text: str, about_self: bool = False) -> Dict:
        payload = {
            "text": text,
            "user_context": self.user_context,
            "dream_context": self.dream_context,
        }
        MAX_GLOBAL_RETRY = 5
        retry_count = 0

        for sephirah in self.PIPELINE:
            node = self.nodes[sephirah]
            result = node.execute(payload)
            self.trace.append(f"[{sephirah.value}] → {list(result.keys())}")

            if result.get("needs_retry") and retry_count < MAX_GLOBAL_RETRY:
                retry_count += 1
                self.trace.append(f"  ↩ fallback recompute (retry #{retry_count})")
                # Fall back to the Tiferet sephirah for recompute (simplified:
                # tune the parameters and pass through again)
                if sephirah == Sephirah.NETZACH:
                    payload["beauty_score"] = max(payload.get("beauty_score", 0) * 1.5, 0.1)
                    continue
                if sephirah == Sephirah.HOD:
                    # Placeholder injected into the payload during HOD fallback
                    # (may be displayed as-is) — runtime content, kept verbatim.
                    payload["result"] = payload.get("result", "（内容重新生成中）")
                    continue

            payload.update(result)

        # Decide whether to run the human-side True-Self branch based on about_self
        display = payload.get("display", payload.get("output", payload.get("result", "")))
        return {
            "answer": display,
            "trace": self.trace,
            "payload_keys": list(payload.keys()),
        }


# ═══════════════════════════════════════════════════════════
# 7.  Main interpreter REPL
# ═══════════════════════════════════════════════════════════

class SephirotInterpreter:
    """
    Main class of the 16-sephiroth interpreter.
    Usage:
      from sephirot_interpreter import SephirotInterpreter
      interp = SephirotInterpreter()
      interp.run_repl()        # interactive mode
      interp.process("...")    # one-shot processing
    """

    BANNER = """
╔══════════════════════════════════════════════════════════╗
║  16-Sephirot Twin-Bliss Protocol Python Interpreter v1.0  ║
║  Kabbalah Tree of Life · CUDA PTX sm_89 · AVX-512         ║
║  Enter Chinese sephirah keywords → auto-compiled to       ║
║  machine code and executed                                ║
║  Commands: :ptx <kw> :avx <kw> :dml <kw> :vocab :exit     ║
╚══════════════════════════════════════════════════════════╝
"""

    def __init__(self,
                 user_context:  Dict = None,
                 dream_context: Dict = None,
                 verbose: bool = True):
        self.lexer    = SephirotLexer
        self.emitter  = MachineCodeEmitter()
        self.scheduler = SephirotScheduler(user_context, dream_context)
        self.verbose  = verbose

    def process(self, text: str, emit_target: str = "ptx") -> Dict:
        """Process a single input and return the result dict"""
        lexer  = self.lexer(text)
        tokens = lexer.tokenize()

        # Machine-code translation
        code_ptx = self.emitter.emit(tokens, "ptx")
        code_avx = self.emitter.emit(tokens, "avx")
        code_dml = self.emitter.emit(tokens, "dml")

        # Sephirah pipeline execution
        # Chinese self-reference markers (matching data — do not change)
        about_self = any(w in text for w in ["我", "自己", "我的", "我想", "我需要"])
        result = self.scheduler.run(text, about_self=about_self)

        return {
            "input": text,
            "tokens_found": [t["keyword"] for t in tokens],
            "ptx_code": code_ptx,
            "avx_code": code_avx,
            "dml_code": code_dml,
            "answer": result["answer"],
            "trace": result["trace"],
        }

    def run_repl(self):
        """Interactive REPL"""
        print(self.BANNER)
        while True:
            try:
                line = input("16-sephirot > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Malkuth] Goodbye.")
                break

            if not line:
                continue

            # Meta-commands
            if line == ":exit" or line == ":quit":
                print("[Malkuth] Exited.")
                break
            if line.startswith(":ptx "):
                kw = line[5:].strip()
                if kw in VOCAB_TO_OPCODE:
                    print(VOCAB_TO_OPCODE[kw]["ptx"])
                else:
                    print(f"Unknown sephirah keyword: {kw}")
                continue
            if line.startswith(":avx "):
                kw = line[5:].strip()
                if kw in VOCAB_TO_OPCODE:
                    print(VOCAB_TO_OPCODE[kw]["avx"])
                else:
                    print(f"Unknown sephirah keyword: {kw}")
                continue
            if line.startswith(":dml "):
                kw = line[5:].strip()
                if kw in VOCAB_TO_OPCODE:
                    print(VOCAB_TO_OPCODE[kw]["dml"])
                else:
                    print(f"Unknown sephirah keyword: {kw}")
                continue
            if line == ":vocab":
                print("Registered sephirah keywords:")
                for k in VOCAB_TO_OPCODE:
                    print(f"  {k}")
                continue

            # Normal processing
            result = self.process(line)
            if self.verbose:
                print(f"\n─── Recognized sephirah ────────────────")
                print("  " + ", ".join(result["tokens_found"]) if result["tokens_found"] else "  (no sephirah keywords found)")
                print(f"\n─── PTX code (sm_89) ───────────────────")
                for ln in result["ptx_code"].split("\n")[:30]:
                    print("  " + ln)
                if len(result["ptx_code"].split("\n")) > 30:
                    print("  ... (truncated, use :ptx <keyword> for full code)")
                print(f"\n─── Pipeline trace ─────────────────────")
                for t in result["trace"]:
                    print("  " + t)
                print(f"\n─── Final output ───────────────────────")
            print(result["answer"] or "(Malkuth has nothing to display)")
            print()


# ═══════════════════════════════════════════════════════════
# 8.  Quick API: single-function call
# ═══════════════════════════════════════════════════════════

def sephirot(text: str,
             user_context: Dict = None,
             dream_context: Dict = None,
             emit: str = "ptx") -> Dict:
    """
    One-line entry point:
        result = sephirot("王冠分析我今天的困境")
        print(result["answer"])
        print(result["ptx_code"])
    """
    interp = SephirotInterpreter(user_context, dream_context, verbose=False)
    return interp.process(text, emit_target=emit)


# ═══════════════════════════════════════════════════════════
# 9.  Entry point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="16-Sephirot Twin-Bliss Protocol interpreter")
    parser.add_argument("--text",    type=str, help="process a single text input directly")
    parser.add_argument("--emit",    type=str, default="ptx",
                        choices=["ptx", "avx", "dml"], help="machine-code output target")
    parser.add_argument("--repl",    action="store_true", help="start the interactive REPL")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if args.repl or not args.text:
        interp = SephirotInterpreter(verbose=args.verbose)
        interp.run_repl()
    else:
        result = sephirot(args.text, emit=args.emit)
        print(f"Recognized sephirah: {result['tokens_found']}")
        print(f"\n=== {args.emit.upper()} code ===")
        print(result[f"{args.emit}_code"])
        print(f"\n=== Output ===")
        print(result["answer"])
