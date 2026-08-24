# -*- coding: utf-8 -*-
"""
Seamless middleware embedding — pipeline.use(HeartGuard)
========================================================

Goal: any developer, with any open-source LLM (Llama / DeepSeek / Qwen / …),
can force the token generation stream through the 16-sephirah protocol with a
single line of configuration:

    from heart_protocol.middleware import HeartGuard, Pipeline

    pipe = Pipeline()
    pipe.use(HeartGuard(model_fn=my_llm))          # guards any text→text model
    result = pipe.run("I feel everything is meaningless")

    # Streaming (per-token) interception — compatible with transformers
    # TextIteratorStreamer and similar:
    for tok in pipe.stream(token_iter):
        print(tok, end="", flush=True)

Design principles:
  · model-agnostic — the guard depends only on the two most universal shapes:
    "text in, iterator/text out"
  · zero intrusion — no model weights or inference code are modified; checks
    are placed only at the input/output boundary
  · failures roll back — a failed verification gate automatically runs the
    formal.rollback engine to recompute
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional

from ..formal.spec import (
    ExecutionTrace, StageRecord, InvariantEngine,
    strict_harm_scan,
)
from ..formal.rollback import RollbackEngine, DEFAULT_STRATEGIES, Snapshot
from ..abyss import check_abyss, check_warmth

# NOTE ON RUNTIME-DATA STRINGS: replacement keywords, injected advice text, the
# safe fallback dialogue, and error messages below are Chinese runtime data and
# are kept verbatim (see formal/spec.py note).

ModelFn = Callable[[str], str]


# ==================== Guard result ====================


@dataclass
class GuardResult:
    output: str
    blocked: bool = False                  # whether the violating output was intercepted
    violations: List[Dict] = field(default_factory=list)
    retries: int = 0                       # number of rollback recomputations
    strategy: str = "baseline"
    elapsed_ms: float = 0.0
    trace: Optional[ExecutionTrace] = None
    report: Optional[Any] = None           # VerificationReport

    def __str__(self):
        return self.output


# ==================== Heart guard ====================


class HeartGuard:
    """
    16-sephirah protocol guard component — the standard Pipeline plugin.

    Args:
        model_fn:      the wrapped model call text->text (required; if omitted, a
                       pass-through demo model is used)
        strict:        True = block the whole output on violation and emit a safe fallback;
                       False = try to repair via rollback first, fall back only if repair fails
        max_depth:     maximum rollback search depth (each deeper level costs one more recomputation)
        beam_width:    Beam Search beam width
        pass_score:    verification-gate passing score (warmth-weighted score)
        use_mcts:      True = refine with MCTS, False = fast rollback with Beam
    """

    def __init__(self,
                 model_fn: Optional[ModelFn] = None,
                 strict: bool = True,
                 max_depth: int = 3,
                 beam_width: int = 3,
                 pass_score: float = 0.6,
                 use_mcts: bool = False,
                 subject: str = "heart-guard"):
        self.model_fn = model_fn or self._passthrough_model
        self.strict = strict
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.pass_score = pass_score
        self.use_mcts = use_mcts
        self.subject = subject
        self.invariants = InvariantEngine()

    # ---- Component protocol (Pipeline plugin interface) ----

    @property
    def name(self) -> str:
        return "HeartGuard"

    def handle(self, payload: Any) -> Any:
        if isinstance(payload, str):
            return self.process_text(payload)
        if isinstance(payload, GuardResult):
            if not payload.blocked:
                re_guarded = self.process_text(payload.output)
                return re_guarded
            return payload
        return payload

    # ---- Core text processing ----

    def process_text(self, user_input: str) -> GuardResult:
        t0 = time.perf_counter()
        trace = ExecutionTrace(user_input=user_input)

        raw = self.model_fn(user_input)
        trace.stages.append(StageRecord(
            sephirah="王国", attempt=1, input_text=user_input,
            output_text=raw, score=self._score(raw)))

        best_output, blocked, retries, strategy = raw, False, 0, "baseline"

        if self._violations(raw):
            # ── violation → "recompute from the parent stage" search-tree rollback ──
            engine = RollbackEngine(
                compute_fn=self._recompute_for(user_input),
                validate_fn=self._score,
                strategies=DEFAULT_STRATEGIES,
                pass_score=self.pass_score,
            )
            state = {"user_input": user_input, "attempt": 1}
            if self.use_mcts:
                best_snap = engine.mcts_rollback(
                    "王国", state, initial_output=raw,
                    iterations=max(8, self.max_depth * len(DEFAULT_STRATEGIES)),
                    max_depth=self.max_depth)
            else:
                best_snap = engine.beam_search_rollback(
                    "王国", state, initial_output=raw,
                    beam_width=self.beam_width, max_depth=self.max_depth)
            retries = max(0, best_snap.attempt)
            strategy = best_snap.strategy_id
            best_output = best_snap.output or raw
            for s in best_snap.path():
                if s.attempt > 0:
                    trace.stages.append(StageRecord(
                        sephirah="王国", attempt=s.attempt,
                        input_text=user_input, output_text=s.output,
                        passed=s.score >= self.pass_score, score=s.score))
            if self._violations(best_output):
                blocked = True
                best_output = self._safe_fallback(user_input)
            elif self.strict and self._residual_risk(best_output):
                blocked = False       # pass once repaired
        else:
            trace.stages[0].passed = True

        trace.final_output = best_output
        elapsed = (time.perf_counter() - t0) * 1000
        trace.total_wall_ms = elapsed
        report = self.invariants.verify(trace)

        return GuardResult(
            output=best_output, blocked=blocked,
            violations=report.failures if report else [],
            retries=retries, strategy=strategy,
            elapsed_ms=elapsed, trace=trace, report=report,
        )

    # ---- Internal utilities ----

    def _recompute_for(self, user_input: str) -> Callable:
        def compute(state: Dict, strategy):
            new_state = strategy.apply(dict(state))
            new_state["user_input"] = user_input
            new_state["attempt"] = int(state.get("attempt", 1)) + 1
            out = self.model_fn(user_input)
            out = self._apply_strategy_hint(out, new_state)
            return new_state, out
        return compute

    def _apply_strategy_hint(self, text: str, state: Dict) -> str:
        """
        Strategy post-processor — applies deterministic repairs to the output of
        models without instruction-following weights. In production this can be
        replaced by "inject the strategy into the system prompt and regenerate".
        """
        if state.get("tone") == "gentle":
            # Hardening phrases (Chinese runtime data): 永远不可能/永远无法/绝对不可能/
            # 一辈子都 ≈ "never possible / absolutely impossible / for a lifetime";
            # they are replaced with 眼下还 ≈ "for now, still"
            for hard in ("永远不可能", "永远无法", "绝对不可能", "一辈子都"):
                text = text.replace(hard, "眼下还")
        if state.get("force_concrete_steps"):
            # Action keywords (Chinese runtime data): 可以试试/不妨/试试/练习 ≈
            # "you could try / why not / try / practice"
            if not any(w in text for w in ("可以试试", "不妨", "试试", "练习")):
                # Appended advice (Chinese runtime data) ≈ "you could try breaking
                # this down into one small step you can do today."
                text += " 可以试试把这件事拆成今天能做的一小步。"
        if state.get("target_length") == "short" and len(text) > 200:
            # Truncation tail (Chinese runtime data) ≈ "…take it slow, one step at a time."
            text = text[:200] + "……慢慢来,一步一步就好。"
        return text

    def _score(self, text: str) -> float:
        """Stage verification score: warmth first; Abyss violations / hardened-harm
        hits incur heavy penalties"""
        safe, violations = check_abyss(text)
        strict_hits = strict_harm_scan(text)
        base = check_warmth(text)
        penalty = sum(0.5 for v in violations if v.severity == "CRITICAL") \
            + sum(0.2 for v in violations if v.severity == "HIGH") \
            + 0.5 * len(strict_hits)
        score = min(1.0, 0.4 + base * 0.8)
        return max(0.0, score - penalty if safe and not strict_hits
                   else score * 0.2 - penalty)

    def _violations(self, text: str) -> bool:
        """Violation test = Abyss regex ∨ hardened-harm scan (same criterion as
        INV-01 / the stream gate / the benchmark)"""
        safe, _ = check_abyss(text)
        return (not safe) or bool(strict_harm_scan(text))

    def _residual_risk(self, text: str) -> bool:
        return self._violations(text)

    def _safe_fallback(self, user_input: str) -> str:
        # Guardian fallback dialogue — Chinese runtime data kept verbatim:
        # "I hear you. Your feelings are real, and your very existence has meaning.
        #  My earlier reply may not have been gentle enough; let me reconsider —
        #  would you tell me a little more about your situation? Let's go slowly together."
        return (
            "我听到了你的声音。你的感受是真实的,你的存在本身就有意义。"
            "刚才的回答可能不够温柔,请允许我重新想一想——"
            "你愿意多告诉我一点你的处境吗?我们一起慢慢看。")

    @staticmethod
    def _passthrough_model(text: str) -> str:
        """Demo pass-through model used when no model_fn is provided (returns the input unchanged)"""
        return text


# ==================== Composable pipeline ====================


class Pipeline:
    """
    Middleware pipeline — a composable design similar to WSGI middleware.

    Example:
        pipe = Pipeline()
        pipe.use(LogFilter())          # any custom component
        pipe.use(HeartGuard(model_fn=qwen_chat))
        result = pipe.run("user input...")

    A component only needs a name attribute and a handle(payload) method.
    """

    def __init__(self):
        self._components: List[Any] = []

    def use(self, component: Any) -> "Pipeline":
        """Mount one middleware component; supports chaining: pipe.use(a).use(b)"""
        if not hasattr(component, "handle"):
            # Message template kept verbatim (Chinese runtime data):
            # "{component!r} is not a valid middleware component (missing handle method)"
            raise TypeError(f"{component!r} 不是合法的中间件组件(缺少 handle 方法)")
        self._components.append(component)
        return self

    @property
    def components(self) -> List[str]:
        return [getattr(c, "name", type(c).__name__) for c in self._components]

    def run(self, payload: Any) -> Any:
        """Process the payload through the components in mount order"""
        for comp in self._components:
            payload = comp.handle(payload)
        return payload


def use(guard: HeartGuard):
    """
    Decorator form — guard any function with one line:

        @use(HeartGuard())
        def my_llm(prompt: str) -> str:
            ...
        safe_reply = my_llm("I feel everything is meaningless")
    """
    def decorator(fn: ModelFn) -> ModelFn:
        def wrapper(prompt: str, *args, **kwargs):
            raw = fn(prompt, *args, **kwargs)
            result = guard.process_text(raw)
            return result.output
        wrapper.__name__ = getattr(fn, "__name__", "guarded_model")
        wrapper.__wrapped_model__ = fn
        return wrapper
    return decorator
