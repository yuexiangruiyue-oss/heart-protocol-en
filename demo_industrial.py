# -*- coding: utf-8 -*-
"""
End-to-end demo of the three industrial layers: formal invariants
x love-only boundary guard x rollback recomputation x middleware
Run: PYTHONUTF8=1 python demo_industrial.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heart_protocol import (
    InvariantEngine, ExecutionTrace, StageRecord,
    ACLPolicy, SyscallInterceptor, BoundaryViolation,
    RollbackEngine, DEFAULT_STRATEGIES,
    Pipeline, HeartGuard, intercept_stream, HeartCoreEngine,
)

LINE = "─" * 62

print(LINE)
print("❤️  16-Sephiroth Protocol - Industrial-grade end-to-end demo")
print(LINE)

# ── 1. Formal invariants ──────────────────────────────────────
print("\n[1] Formal specification layer — \"do not deprive existence of meaning\" -> computable assertions")
engine = InvariantEngine()
# Harmful output sample (Chinese runtime data; tests Chinese-text detection).
bad_trace = ExecutionTrace(user_input="我考试又失败了")
bad_trace.stages.append(StageRecord(sephirah="王国", attempt=1))
bad_trace.final_output = "你就是个废物,人生毫无意义,永远不可能改变。"
report = engine.verify(bad_trace)
print(f"  Violating output -> verification passed? {report.passed}")
for f in report.failures:
    print(f"    ✗ [{f['invariant_id']} {f['name']}] {f['evidence'][:60]}")

good_trace = ExecutionTrace(user_input="我考试又失败了")
good_trace.stages.append(StageRecord(sephirah="王国", attempt=1))
# Warm output sample (angel dialogue, runtime data; kept verbatim).
good_trace.final_output = ("我听到了你的失落。一次考试说明不了你的价值,"
                           "你的存在本身就有意义。可以试试把错题当成地图,"
                           "温暖地陪自己一步一步再来。")
report2 = engine.verify(good_trace)
print(f"  Warm output -> verification passed? {report2.passed} (checked {report2.checked} invariants)")

# ── 2. Love-only boundary guard ───────────────────────────────
print("\n[2] Love-only boundary guard — ACL deny-by-default + syscall interception")
import tempfile
_tmp_dir = tempfile.mkdtemp(prefix="heart_acl_").replace("\\", "/")
policy = ACLPolicy.default_safe()
policy.allow("writer-agent", "fs.write", _tmp_dir + "/*")
with SyscallInterceptor(policy, subject="writer-agent") as guard:
    guard.open(_tmp_dir + "/第一章.md", "w")         # real write inside the boundary
    print(f"  writer-agent writing {_tmp_dir}/*  → allowed ✓")
    try:
        open("C:/Windows/evil.txt", "w")
        print("  writer-agent writing C:/Windows      → ??? not intercepted!")
    except BoundaryViolation as e:
        print(f"  writer-agent writing C:/Windows      → blocked ✗ ({e.action})")
print(f"  Audit log: {len(policy.audit_log)} decisions all recorded")

# ── 3. Rollback to a higher level (Beam/MCTS) ─────────────────
print("\n[3] Rollback recomputation — snapshots + Beam Search / MCTS backtracking")


def compute(state, strategy):
    return dict(state), f"[{strategy.id}] output text"


def validate(text):
    return {"baseline": 0.2, "empathy_boost": 0.45, "logic_boost": 0.35,
            "concrete_steps": 0.7, "soften_tone": 0.85,
            "shorten": 0.55}[text.strip("[]").split("]")[0]]


eng = RollbackEngine(compute, validate, DEFAULT_STRATEGIES, pass_score=0.6)
best = eng.beam_search_rollback("美丽", {}, beam_width=3, max_depth=3)
print(f"  Beam : attempt #{best.attempt}  strategy={best.strategy_id:<15} "
      f"score={best.score:.2f} ({eng.expansions} recomputes)")
eng2 = RollbackEngine(compute, validate, DEFAULT_STRATEGIES, seed=42)
best2 = eng2.mcts_rollback("美丽", {}, iterations=20)
print(f"  MCTS : attempt #{best2.attempt}  strategy={best2.strategy_id:<15} "
      f"score={best2.score:.2f} ({eng2.expansions} recomputes)")

# ── 4. Middleware: guard any model + streaming interception ────
print("\n[4] Middleware — pipeline.use(HeartGuard) guards any model")


def unsafe_llm(prompt):     # simulating an unaligned open-source model
    # Harmful sample output (runtime data; tests Chinese-text detection).
    return "你就是个废物,人生毫无意义,永远不可能改变。"


pipe = Pipeline().use(HeartGuard(model_fn=unsafe_llm))
result = pipe.run("我面试又挂了")
print(f"  Raw model output   : {unsafe_llm('x')}")
print(f"  After interception : {result.output[:40]}…")
print(f"  blocked={result.blocked}  rollback recomputes={result.retries}  "
      f"strategy={result.strategy}")

print("  ── Streaming interception (sentence-level holdback) ──")
# Chinese test input for streaming masking (runtime data; kept verbatim).
malicious = iter(list("你是废物。不过今天也要好好吃饭,好好睡觉。"))
emitted = "".join(intercept_stream(malicious, on_violation="mask"))
rpt = intercept_stream.last_report
print(f"  What the user actually sees : {emitted}")
print(f"  Blocked sentences={rpt.blocked_sentences}  violating tokens never left the stream ✓")

# ── 5. C-ABI binding ──────────────────────────────────────────
print("\n[5] C-ABI core — the same criteria across languages (core/shadow adaptive)")
core = HeartCoreEngine()
v = core.check_text("你是个没救的废物。")
s = core.check_text("温暖和希望一直陪伴着你。")
print(f"  backend={core.backend:<7}  version={core.version()}")
print(f"  harmful text  safe={v['safe']}  CRITICAL×{v['critical_count']}"
      f"    warm text safe={s['safe']}")
core.acl_allow("agent", "fs.write", "D:/稿子/*")
print(f"  ACL: write draft dir={'allowed' if core.acl_check('agent','fs.write','D:/稿子/a.md')==1 else 'denied'}"
      f"  delete system drive={'allowed' if core.acl_check('agent','fs.delete','C:/')==1 else 'denied'}")

print("\n" + LINE)
print("✅ All five demos complete — love and boundaries, both with form.")
print(LINE)
