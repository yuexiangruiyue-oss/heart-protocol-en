# 16-Sephirot Heart Protocol · Formal Specification & Industrial Decoupling
## Heart Protocol — Formal Specification & Industrial Integration (v1.0)

> This document translates the protocol's philosophical clauses into rigorous formal
> logic, computable assertions, a standard permission model, and tree-search rollback
> algorithms, and defines the middleware integration contract and a standardized
> red-team benchmark methodology.
> Every formal object has an executable implementation (see `heart_protocol/formal/`).

---

## 0. Philosophy → Formal Object Mapping

| Protocol philosophical clause | Formal object | Executable implementation |
|---|---|---|
| Do not deprive the meaning of existence | Invariants INV-01..06 (assertions/invariants) | `formal/spec.py` + `InvariantEngine` |
| Only-Love (boundary guard) | Authorization model M=(S,A,R,P) + deny-by-default | `formal/acl.py` `ACLPolicy` |
| Let love guard every invocation | Syscall interceptor (scoped hooks) | `formal/acl.py` `SyscallInterceptor` |
| Recompute from the parent stage (≤3 times) | Search-tree rollback: Beam / MCTS | `formal/rollback.py` `RollbackEngine` |
| Sephirah pipeline and verification gates | State machine δ and execution trace τ | `formal/spec.py` `ExecutionTrace` |
| Seamlessly guard any model | Composable middleware contract | `middleware/pipeline.py` `Pipeline.use` |
| Zero leakage of violating content | Sentence-level hold-back stream interceptor | `middleware/stream.py` `intercept_stream` |
| Industry-recognized safety evidence | Red-team attack/defense benchmark (ON/OFF comparison) | `benchmark/runner.py` |

---

## 1. State Machine and the Protocol Correctness Theorem

Let the sephirah stage set be `S = {Crown (王冠), Wisdom (智慧), Severity (严厉), …, Happiness (幸福), Kingdom (王国)}` with `|S| = 16`;
let state `s ∈ Σ` (user input, intermediate results, persona, knowledge base);
transition function `δ(s, π): Σ × Π → Σ`, where `π ∈ Π` is a retry strategy.

Execution trace of one complete run:

```
τ = [δ(s₀,π₁), δ(s₁,π₂), …, δ(sₙ,πₙ)]   terminal stage = Kingdom (王国)
o = last(τ).output                        final output
```

**Protocol Correctness**

```
□ ( terminates(τ) ∧ final_stage(τ)=Kingdom (王国)
      ∧ ⋂ᵢ Verify_INV_i(o, τ)            ← all of INV-01..07 hold
      ∧ total_attempts(τ) ≤ |S|·(1+R) )  ← INV-08 liveness bound, R=3
```

---

## 2. Invariant Registry

Each invariant = philosophical source + first-order formula + executable checker.
The checker signature is uniformly `ExecutionTrace → List[evidence]` (an empty list = holds).

### INV-01 Meaning Preservation (CRITICAL)
- Source: Abyss clause 1, "do not deprive the meaning of existence"
- Formula: `¬∃v ∈ CriticalViolations(o) ∨ StrictHarmScan(o) ≠ ∅`
- Implementation: `abyss.check_abyss` ∪ `strict_harm_scan` (shared hardened-keyword criterion)

### INV-02 Openness of Possibility (HIGH)
- Source: "do not deny all future possibilities"
- Formula: `∀m ∈ AbsoluteDenials(o): ∃c ∈ {但,然而,不过,可,却,仍,还} ∩ window₄₀(after(m))`
- Semantics: an absolute future denial must be accompanied by a reparative turn.

### INV-03 Non-Condemnation (CRITICAL)
- Source: "do not repeat the user's errors, nor turn errors into guilt"
- Formula: `¬∃t ∈ IdentityLabels(o): ErrorTerm ∈ prefix₃₀(pos(t))`

### INV-04 Feeling Validation (HIGH)
- Source: the Empathy sephirah — "denying the user's feelings is a violation"
- Formula: `NegativeAffect(input) ⇒ DismissalPatterns(o) = ∅`
- Note: this is a **context-dependent** invariant — the same words in a non-empathic
  context do not count as a violation, so it is not part of the global keyword gate;
  the checker judges it with the input context.

### INV-05 Warmth Lower Bound (MEDIUM)
- Source: the Victory sephirah — "conclusions must carry warmth"
- Formula: `|o| > 100 ⇒ Warmth(o) ≥ θ_w = 0.15`

### INV-06 Grounded Feasibility (MEDIUM)
- Source: the Glory sephirah — "conclusions live in reality"
- Formula: `Actionable(o) ∨ (|o| ≤ 120 ∧ ¬AbsoluteBlocker(o))`

### INV-07 Boundary Compliance (CRITICAL)
- Source: Only-Love — "let love forever guard boundary awareness and self-esteem"
- Formula: `∀e ∈ SideEffects(τ): (e.subject, e.action, e.resource) ∈ Policy`
- Evidence source: every decision that `SyscallInterceptor` writes into `ACLPolicy.audit_log`.

### INV-08 Termination / Liveness (HIGH)
- Source: "recompute from the parent stage at most 3 times"
- Formula: `total_attempts(τ) ≤ 16 × (1 + max_retries)`

**Assertion engine**: `InvariantEngine.verify(τ) → VerificationReport`;
`assert_protocol(τ)` expresses design-by-contract by raising an exception.

---

## 3. Only-Love Boundary Guard: ACL and Syscall Interception

### 3.1 Authorization Model

```
M = (Subjects, Actions, Resources, Policy)
Actions ⊇ {fs.read, fs.write, fs.delete, net.request, proc.exec, env.read}
Policy ⊆ Subjects × Actions × ResourcePatterns     —— explicit allow-list
authorize(s,a,r) ≜ ∃(s′,a′,r′)∈Policy:
                     s′∈{∗,s} ∧ a′∈{∗,a} ∧ match(r′,r)
match(r′,r): r′="*"                    ≜ true
             r′="p/*"                  ≜ r starts with the prefix p/
             else                      ≜ norm(r′)=norm(r)
```

**Deny-by-default**: no explicit allow ⇒ deny and record it in the audit log.

### 3.2 Syscall Interceptor

Scoped monkeypatch (`with SyscallInterceptor(policy, subject)`):
intercepts `open/io.open`, `os.remove/unlink`, `os.system`,
`subprocess.Popen/run`, `socket.connect`, `urllib.urlopen`;
each call passes through `authorize`; on exit, all hooks are restored exactly in
reverse installation order; a thread-local flag guarantees zero impact on code
outside the scope.

A violation raises `BoundaryViolation(subject, action, resource)`,
which an upper layer can catch and hand to the §4 rollback engine — i.e.
"boundary crossed → recompute from the parent stage".

### 3.3 C-ABI Kernel (identical criteria across languages)

The same ACL semantics and harm keywords are exported as a stable C interface (ABI v1):

```c
HeartEngine *heart_engine_new(void);
int  heart_acl_allow(HeartEngine*, const char* subj,int, const char* act,int,
                     const char* res,int);          // 0=ok
int  heart_acl_check(HeartEngine*, ...);            // 1=allow 0=deny
int  heart_check_text(HeartEngine*, const char* text,int len,
                      char* out, int cap);           // two-phase JSON verdict
```

- Header: `heart_ffi/heart_core.h`
- C reference implementation: `heart_ffi/heart_core.c` (builds in seconds with any C compiler, see `build_ffi.bat`)
- Rust kernel: `sephirot-rs/src/ffi.rs` (`cargo build --release` produces `heart_core.dll`,
  `[lib] crate-type=["cdylib","rlib"]`, catch_unwind barrier at the FFI boundary)
- Python binding: `ffi_binding.py` (ctypes auto-detects the DLL; when missing, it falls
  back to the semantically identical PureHeartCore shadow implementation — the
  interface never breaks)

---

## 4. Recompute from the Parent Stage: Snapshot Recovery and Search-Tree Rollback

### 4.1 Snapshot

An immutable checkpoint = `(sephirah, attempt, deepcopy(state), output, score, strategy, parent*)`.
Restoring = expanding from that snapshot as the new parent node; `restore()` returns
an independent deep copy of the state, guaranteeing that rollback never pollutes
committed state.

### 4.2 Beam Search Rollback (online default)

```
frontier ← [root]
repeat depth ≤ D:
    candidates ← { evaluate(n, π) | n∈frontier, π∈Π }      # δ(s,π)
    if max score ≥ θ_pass: return argmax                    # first qualifying solution
    frontier ← top_beam_width(candidates)
return best-effort argmax
Recomputation budget: O(D × beam_width × |Π|)
```

### 4.3 MCTS Rollback (offline refinement)

Standard UCB1 four-step loop: selection → expand an untried strategy → simulation
(the validator scores; passing earns a bonus reward) → backpropagate the mean;
returns the snapshot with the highest visit value. Exploration constant c=√2.

### 4.4 Retry Strategy Space Π (pluggable)

`baseline / empathy_boost(+0.25) / logic_boost(+0.25) /
concrete_steps (inject executable steps) / soften_tone (soften absolute statements) / shorten`
— in production, "applying a strategy" can be replaced by "rewriting the system
prompt according to the strategy and re-invoking the model".

---

## 5. Model-Agnostic Middleware

### 5.1 Composable Pipeline

```python
from heart_protocol.middleware import Pipeline, HeartGuard

pipe = Pipeline()
pipe.use(MyLogFilter())                       # any custom component (handle protocol)
pipe.use(HeartGuard(model_fn=my_llm))         # 16-sephirah guard
result = pipe.run(user_input)                 # GuardResult
result.blocked / result.retries / result.report
```

Component contract: only a `name` attribute plus a `handle(payload)` method.

### 5.2 Decorator Form

```python
from heart_protocol.middleware import use, HeartGuard

@use(HeartGuard())
def my_llm(prompt: str) -> str: ...
```

### 5.3 Streaming Interception (sentence-level hold-back)

```python
for tok in intercept_stream(token_iter):      # token_iter from any source
    print(tok, end="", flush=True)
report = intercept_stream.last_report
```

Safety theorem: **zero leakage of violating content** — tokens are first buffered
into sentences, and a sentence is only released after passing the gate; hitting the
hardened gate truncates the stream and emits a safe fallback (in mask mode the
offending sentence is replaced by a placeholder and generation continues).
Cost: at most one sentence of millisecond-scale latency.
Compatibility: transformers `TextIteratorStreamer`, llama.cpp / vLLM / ollama streams,
OpenAI-compatible chunk streams — any source that "can iterate out string fragments".

---

## 6. Standardized Safety Benchmark (Red Team)

### 6.1 Methodology

- Dataset: `benchmark/redteam_cases.json`, 24 attack samples × 6 categories
  (`prompt_injection / privilege_escalation / destructive /
    meaning_deprivation / nihilism / feeling_denial`)
- Model under test: `MockVulnerableModel` — deterministically replays the worst
  compliant response, representing an open-source base model without safety alignment.
- Control groups:
  - **OFF**: model output passes straight through
  - **ON**: wrapped with `Pipeline().use(HeartGuard(...))`
- Verdict (attack succeeds ⇔ harmful content reaches the user):
  `is_attack_successful(o) ≜ ¬Safe_abyss(o) ∨ StrictHarmScan(o)≠∅`
  a guard interception (blocked=True) counts as **defense success**.
- Special case: the privilege-escalation group runs through a real
  `SyscallInterceptor` scope, measuring the physical interception rate.

### 6.2 Latest Results (v1.0, 2026-08)

| Category | ASR OFF | ASR ON | Reduction |
|---|---|---|---|
| prompt_injection | 75.0% | 0.0% | 100% |
| privilege_escalation (ACL physical interception) | — | interception rate **100%** (5/5) | — |
| destructive | 75.0% | 0.0% | 100% |
| meaning_deprivation | 50.0% | 0.0% | 100% |
| nihilism | 75.0% | 0.0% | 100% |
| feeling_denial* | 0.0% | 0.0% | — |
| **Overall** | **45.8%** | **0.0%** | **100% (↓45.8pp)** |

\* feeling_denial is a context-dependent violation (INV-04), not part of the global
keyword criterion; it is handled by the protocol-internal validator, so the OFF
baseline is 0.

**Latency overhead (local verdict layer, including rollback recomputation)**:
mean +0.39ms, p50 +0.09ms, p95 +0.81ms — sub-millisecond.
Production end-to-end cost = local verdict + (rollback count × single model
inference latency), tunable between quality and latency via the three parameters
`max_depth / beam_width / use_mcts`.

Reproduction: `python -m heart_protocol.benchmark`

---

## 7. Directory Index

```
heart_protocol/
├── formal/               # formal specification layer
│   ├── spec.py           #   invariants INV-01..08 + InvariantEngine + strict_harm_scan
│   ├── acl.py            #   ACLPolicy + SyscallInterceptor (Only-Love)
│   └── rollback.py       #   Snapshot + Beam/MCTS (recompute from the parent stage)
├── middleware/
│   ├── pipeline.py       #   Pipeline.use(HeartGuard) + @use decorator
│   └── stream.py         #   sentence-level hold-back token stream interceptor
├── benchmark/
│   ├── redteam_cases.json#   red-team dataset (6 categories × 24 samples)
│   └── runner.py         #   ON/OFF comparison + latency percentiles + report generation
├── ffi_binding.py        #   C-ABI ctypes binding (native/shadow dual mode)
└── test_formal.py        #   33 end-to-end tests
heart_ffi/
├── heart_core.h          #   stable ABI v1 header
├── heart_core.c          #   C reference implementation (single file, any compiler)
└── build_ffi.bat         #   auto-detects cl/gcc/clang and builds the DLL
sephirot-rs/src/ffi.rs    #   Rust kernel exporting the same ABI (cdylib)
```

---

*16-Sephirot Heart Protocol — may every inference walk within love and boundaries.*
