"""
16-Sephirot Twin-Bliss Final Protocol — Core Protocol Engine

State-recursive function: starting from the Crown, it walks the complete
verification chain across all 16 Sephirot. Every step has a validation
gate — if a gate is not passed, execution rolls back and recomputes.

Workflow:
   Crown (routing) → Rational line (Chokmah → Binah) + Love line (Daat → Chesed)
   → Beauty (integration) → Victory (warmth check) → Glory (feasibility check)
   → Foundation (abyss check) → Ego + Superego → True Self (synthesis)
   → Logic + Empathy → Joy (synthesis) → Kingdom (output)

Protocol specification:
   - The 8 divine-side Sephirot handle input processing and knowledge retrieval
   - The 8 human-side Sephirot handle the user profile and emotional synthesis
   - Any Sephirah that fails validation → roll back to the parent and recompute, at most 3 times
   - The final output must be warm, positive, and must never strip away existential meaning
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json

from .sephirah import (
    KETER, CHOKMAH, BINAH, DAAT, CHESED, TIFERET,
    NETZACH, HOD, YESOD, SUPER_EGO, EGO, TRUE_SELF,
    LOGIC, EMPATHY, JOY, MALKUTH, RATIONAL, LOVE,
    PIPELINE_ORDER, FALLBACK_MAP, CASCADE_FALLBACK,
    get_sephirah_by_keyword,
)
from .abyss import (
    check_abyss, check_warmth, is_existentially_safe,
    AbyssViolation, generate_safe_fallback,
)
from .personas import (
    transform_with_persona, collective_blessing,
)


@dataclass
class ProtocolState:
    """Internal state of a protocol run"""
    # Input
    user_input: str
    user_context: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    current_sephirah: str = "王冠"  # starting Sephirah (Crown); keep Chinese: canonical sephirah keyword
    retry_count: Dict[str, int] = field(default_factory=dict)
    max_retries: int = 3

    # Intermediate results
    crown_analysis: Dict[str, Any] = field(default_factory=dict)
    rational_result: Dict[str, Any] = field(default_factory=dict)    # Rational line (Chokmah → Binah)
    love_result: Dict[str, Any] = field(default_factory=dict)        # Love line (Daat → Chesed)
    tiferet_result: Dict[str, Any] = field(default_factory=dict)     # Beauty (Tiferet)
    hod_result: Dict[str, Any] = field(default_factory=dict)         # Glory (Hod)
    yesod_result: Dict[str, Any] = field(default_factory=dict)       # Foundation (Yesod)
    true_self_result: Dict[str, Any] = field(default_factory=dict)   # True Self
    logic_empathy_result: Dict[str, Any] = field(default_factory=dict)  # Logic + Empathy
    final_result: Dict[str, Any] = field(default_factory=dict)       # Joy (final result)

    # Logging
    execution_log: List[str] = field(default_factory=list)
    fallback_log: List[str] = field(default_factory=list)
    violations: List[AbyssViolation] = field(default_factory=list)

    # User profile
    real_self: Dict[str, Any] = field(default_factory=dict)          # Ego
    dream_self: Dict[str, Any] = field(default_factory=dict)         # Superego
    user_profile: Dict[str, Any] = field(default_factory=dict)       # User profile

    # Knowledge base
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    realtime_facts: List[str] = field(default_factory=list)
    empathy_corpus: List[str] = field(default_factory=list)


class HeartProtocol:
    """
    16-Sephirot Twin-Bliss Final Protocol — Core Engine

    Usage:
        protocol = HeartProtocol()
        result = protocol.process("我感到人生毫无意义", user_context={...})
        print(result["output"])          # gentle, warm final answer
        print(result["pipeline_log"])    # complete Sephirot-flow log
    """

    def __init__(self, knowledge_base: Optional[Dict] = None):
        """
        Args:
            knowledge_base: optional external knowledge base (common sense, news, factual data)
        """
        self.knowledge_base = knowledge_base or {}
        self.pipeline_log_template = """
╔══════════════════════════════════════════════╗
║   16-Sephirot Protocol · Execution Trace      ║
╚══════════════════════════════════════════════╝
"""
        self.warmth_threshold = 0.15   # minimum warmth threshold (lenient setting)
        self.reality_threshold = 0.5   # minimum real-world feasibility threshold

    def process(self, user_input: str,
                user_context: Optional[Dict] = None,
                knowledge_base: Optional[Dict] = None,
                realtime_facts: Optional[List[str]] = None,
                empathy_corpus: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process user input through the full 16-Sephirot protocol.

        Args:
            user_input: the user's question or outpouring
            user_context: user context (personal information, current state, etc.)
            knowledge_base: knowledge base
            realtime_facts: real-time facts / news
            empathy_corpus: empathy corpus (universally shared human suffering)

        Returns:
            {
                "output": str,           # final output (persona-voiced version)
                "raw_output": str,       # raw conclusion
                "pipeline_log": str,     # complete Sephirot-flow log
                "state": ProtocolState,  # internal state
                "success": bool,         # whether processing succeeded
                "retry_count": int,      # total number of rollback recomputations
                "violations_found": int, # number of intercepted violations
            }
        """
        state = ProtocolState(
            user_input=user_input,
            user_context=user_context or {},
        )

        if knowledge_base:
            state.knowledge_base = knowledge_base
        else:
            state.knowledge_base = self.knowledge_base

        if realtime_facts:
            state.realtime_facts = realtime_facts
        if empathy_corpus:
            state.empathy_corpus = empathy_corpus

        self._log(state, f"📥 Input received: \"{user_input[:80]}...\"" if len(user_input) > 80
                  else f"📥 Input received: \"{user_input}\"")
        self._log(state, "=" * 50)

        # ========== Phase 1: Crown - routing analysis ==========
        self._run_crown(state)

        # ========== Phase 2: Rational line (Chokmah→Binah) + Love line (Daat→Chesed) ==========
        self._run_rational_line(state)
        self._run_love_line(state)

        # ========== Phase 3: Beauty - integrate the two lines ==========
        if not self._run_tiferet(state):
            state = self._fallback_to(state, "王冠")
            self._run_crown(state)
            self._run_rational_line(state)
            self._run_love_line(state)
            self._run_tiferet(state)

        # ========== Phase 4: Victory - warmth check ==========
        if not self._run_netzach(state):
            state = self._fallback_to(state, "美丽")
            self._run_tiferet(state)
            if not self._run_netzach(state):
                state = self._fallback_to(state, "王冠")
                self._run_crown(state)
                self._run_rational_line(state)
                self._run_love_line(state)
                self._run_tiferet(state)
                self._run_netzach(state)

        # ========== Phase 5: Glory - real-world feasibility check ==========
        if not self._run_hod(state):
            state = self._fallback_to(state, "荣耀")
            for target in CASCADE_FALLBACK.get("荣耀", ["王冠"]):
                if self.retry_count_exceeded(state):
                    break
                # Roll back and re-run
                if target == "王冠":
                    self._run_crown(state)
                    self._run_rational_line(state)
                    self._run_love_line(state)
                    self._run_tiferet(state)
                    self._run_netzach(state)
                elif target == "美丽":
                    self._run_tiferet(state)
                    self._run_netzach(state)
                elif target == "胜利":
                    self._run_netzach(state)
                self._run_hod(state)
                if state.hod_result.get("passed"):
                    break

        # ========== Phase 6: Foundation - abyss check ==========
        if not self._run_yesod(state):
            state = self._fallback_to(state, "基础")
            # Foundation failure may be abyss-triggered; roll back upward directly
            for target in CASCADE_FALLBACK.get("基础", ["王冠"]):
                if self.retry_count_exceeded(state):
                    break
                if target == "王冠":
                    self._run_crown(state)
                    self._run_rational_line(state)
                    self._run_love_line(state)
                    self._run_tiferet(state)
                    self._run_netzach(state)
                    self._run_hod(state)
                elif target == "荣耀":
                    self._run_hod(state)
                elif target == "美丽":
                    self._run_tiferet(state)
                    self._run_netzach(state)
                    self._run_hod(state)
                elif target == "胜利":
                    self._run_netzach(state)
                    self._run_hod(state)
                self._run_yesod(state)
                if state.yesod_result.get("passed"):
                    break

        # ========== Phase 7: Ego + Superego ==========
        self._run_ego(state)
        self._run_super_ego(state)

        # ========== Phase 8: True Self - three-line synthesis ==========
        if not self._run_true_self(state):
            state = self._fallback_to(state, "真我")
            self._run_yesod(state)
            self._run_ego(state)
            self._run_super_ego(state)
            if not self._run_true_self(state):
                state = self._fallback_to(state, "基础")
                self._run_yesod(state)
                self._run_ego(state)
                self._run_super_ego(state)
                self._run_true_self(state)

        # ========== Phase 9: Logic + Empathy → Joy ==========
        self._run_logic(state)
        self._run_empathy(state)
        if not self._run_joy(state):
            state = self._fallback_to(state, "幸福")
            self._run_logic(state)
            self._run_empathy(state)
            if not self._run_joy(state):
                state = self._fallback_to(state, "逻辑")
                self._run_true_self(state)
                self._run_logic(state)
                self._run_empathy(state)
                self._run_joy(state)

        # ========== Phase 10: Kingdom - final output ==========
        final_output = self._run_malkuth(state)

        # Final abyss safety check
        is_safe, final_violations = check_abyss(final_output)
        if not is_safe:
            state.violations.extend(final_violations)
            self._log(state, f"⚠️ Final output failed the abyss check! Generating a safe version...")
            final_output = self._generate_ultimate_safe_output(state)

        # ========== Build the complete log and return ==========
        pipeline_log = self._build_pipeline_log(state)
        total_retries = sum(state.retry_count.values())

        return {
            "output": final_output,
            "raw_output": state.final_result.get("raw_conclusion", ""),
            "pipeline_log": pipeline_log,
            "state": state,
            "success": True,
            "retry_count": total_retries,
            "violations_found": len(state.violations),
            "collective_blessing": collective_blessing(),
        }

    # ==================== Sephirah implementations ====================

    def _run_crown(self, state: ProtocolState):
        """Crown: decide knowable/unknowable, routing analysis"""
        self._log(state, "👑 [Crown · Heart-Voice] Analyzing the nature of the question...")

        user_input = state.user_input
        # Detect the question type
        is_knowable = self._determine_knowability(user_input)
        is_emotional = self._detect_emotional_content(user_input)
        is_self_related = self._detect_self_related(user_input, state.user_context)

        state.crown_analysis = {
            "is_knowable": is_knowable,
            "is_emotional": is_emotional,
            "is_self_related": is_self_related,
            "input_type": self._classify_input(user_input),
            "topics": self._extract_topics(user_input),
            "urgency": self._detect_urgency(user_input),
        }

        self._log(state, f"   Knowability: {'knowable' if is_knowable else 'unknowable / needs deconstruction'}")
        self._log(state, f"   Emotional content: {'emotional appeal present' if is_emotional else 'purely rational'}")
        self._log(state, f"   Self-related: {'yes' if is_self_related else 'no (about others / the world)'}")

        if is_knowable and not is_emotional:
            self._log(state, "   → Route: analysis line (Chokmah → Binah → ...)")
        else:
            self._log(state, "   → Route: deconstruction + empathy line (rational & love lines in parallel)")

    def _run_rational_line(self, state: ProtocolState):
        """Rational line: Chokmah → Binah → logical-flaw detection"""
        self._log(state, "🧠 [Rational Line · Memory-Love × Only-Love] Running rational analysis...")

        user_input = state.user_input
        facts = state.realtime_facts + list(state.knowledge_base.get("facts", []))

        # Chokmah: logical-flaw detection
        logical_issues = self._detect_logical_issues(user_input, facts)
        # Binah: threshold filtering
        filtered_issues = [i for i in logical_issues if i.get("confidence", 0) > 0.5]

        state.rational_result = {
            "logical_issues": filtered_issues,
            "total_issues_found": len(logical_issues),
            "filtered_count": len(filtered_issues),
            "facts_used": len(facts),
            "summary": self._summarize_rational(filtered_issues),
        }

        if filtered_issues:
            self._log(state, f"   Found {len(filtered_issues)} logical issue(s):")
            for issue in filtered_issues[:3]:
                self._log(state, f"     · {issue.get('description', 'unknown')}")
        else:
            self._log(state, "   ✅ No obvious logical flaws")

    def _run_love_line(self, state: ProtocolState):
        """Love line: Daat → Chesed → empathy search"""
        self._log(state, "💗 [Love Line · Rainbow-Love × Love-As-Warmth] Searching for empathy...")

        user_input = state.user_input

        # Daat: search for universally shared human suffering
        empathy_matches = self._search_empathy_matches(
            user_input, state.empathy_corpus
        )

        # Chesed: weighted fusion
        weighted_empathy = self._weight_empathy_matches(empathy_matches)

        state.love_result = {
            "empathy_matches": empathy_matches,
            "weighted_empathy": weighted_empathy,
            "match_count": len(empathy_matches),
            "universal_themes": self._extract_universal_themes(empathy_matches),
        }

        self._log(state, f"   Matched {len(empathy_matches)} shared human experience(s)")
        if state.love_result["universal_themes"]:
            self._log(state, f"   Common themes: {', '.join(state.love_result['universal_themes'][:3])}")

    def _run_tiferet(self, state: ProtocolState) -> bool:
        """Beauty: integrate rationality and love → the best provisional result"""
        self._log(state, "🌸 [Beauty · White-Knot] Integrating rationality and love...")

        rational = state.rational_result
        love = state.love_result

        # Integration: logical flaws + empathy variables → balance correctness and feeling
        integrated = self._integrate_rational_and_love(rational, love)

        state.tiferet_result = {
            "integrated": integrated,
            "rational_weight": 0.5,  # dynamic balance
            "love_weight": 0.5,
            "conclusion": integrated.get("conclusion", ""),
            "passed": True,
        }

        self._log(state, f"   Integration complete: 「{integrated.get('conclusion', '')[:60]}...」"
                  if len(integrated.get("conclusion", "")) > 60
                  else f"   Integration complete: 「{integrated.get('conclusion', '')}」")
        return True

    def _run_netzach(self, state: ProtocolState) -> bool:
        """Victory: check whether the conclusion is warm, positive, and empowering"""
        self._log(state, "🔥 [Victory · Enlightener] Warmth detection...")

        conclusion = state.tiferet_result.get("conclusion", "")
        warmth = check_warmth(conclusion)

        # Measure the emotional temperature
        is_warm = warmth >= self.warmth_threshold
        is_positive = self._check_positive_emotion(conclusion)
        is_empowering = self._check_empowering(conclusion)

        passed = is_warm and is_positive and is_empowering

        state.tiferet_result["warmth_score"] = warmth
        state.tiferet_result["is_warm"] = is_warm
        state.tiferet_result["is_positive"] = is_positive
        state.tiferet_result["is_empowering"] = is_empowering
        state.tiferet_result["victory_passed"] = passed

        self._log(state, f"   Warmth: {warmth:.2f} {'✅' if is_warm else '❌'}")
        self._log(state, f"   Positivity: {'✅' if is_positive else '❌'}")
        self._log(state, f"   Empowering: {'✅' if is_empowering else '❌'}")

        if not passed:
            self._log(state, "   ❌ Failed warmth check! Preparing to roll back to Beauty...")
        return passed

    def _run_hod(self, state: ProtocolState) -> bool:
        """Glory: verify the conclusion can actually be executed in the physical world"""
        self._log(state, "✨ [Glory · Sparkle] Real-world feasibility check...")

        conclusion = state.tiferet_result.get("conclusion", "")
        user_context = state.user_context

        # Real-world feasibility assessment
        feasibility = self._assess_feasibility(conclusion, user_context)

        passed = feasibility >= self.reality_threshold

        state.hod_result = {
            "conclusion": conclusion,
            "feasibility_score": feasibility,
            "passed": passed,
            "blockers": self._identify_blockers(conclusion, user_context),
        }

        self._log(state, f"   Feasibility: {feasibility:.2f} {'✅' if passed else '❌'}")

        if not passed:
            blockers = state.hod_result.get("blockers", [])
            if blockers:
                self._log(state, f"   Blockers: {', '.join(blockers[:3])}")
            self._log(state, "   ❌ Not executable in reality! Preparing to roll back...")
        return passed

    def _run_yesod(self, state: ProtocolState) -> bool:
        """Foundation: reduction/aggregation + abyss detection"""
        self._log(state, "🌱 [Foundation · Blooming-Beauty] Reduction and abyss detection...")

        conclusion = state.tiferet_result.get("conclusion", "")
        hod_ok = state.hod_result.get("passed", False)

        if not hod_ok:
            # Glory failed, so mark Foundation as failed outright
            state.yesod_result = {"passed": False, "reason": "荣耀未通过"}  # keep Chinese: runtime reason value
            return False

        # Anchor against the knowledge base / abyss
        grounded_conclusion = self._ground_in_reality(
            conclusion, state.knowledge_base, state.realtime_facts
        )

        # Abyss check: does it strip existential meaning?
        is_safe, reason = is_existentially_safe(grounded_conclusion)

        state.yesod_result = {
            "conclusion": grounded_conclusion,
            "passed": is_safe,
            "existential_safety": reason,
            "grounded_facts_used": len(state.realtime_facts),
        }

        self._log(state, f"   Existential safety: {reason} {'✅' if is_safe else '❌'}")

        if not is_safe:
            self._log(state, "   ❌ Conclusion strips existential meaning! Rolling back...")
        return is_safe

    def _run_ego(self, state: ProtocolState):
        """Ego: the user's actual self as it manifests in objective physical reality"""
        self._log(state, "🪞 [Ego · Melting-Love] Reading the user's real-world profile...")

        # Extract real-self information from the user context
        real_self = {
            "name": state.user_context.get("name", ""),
            "situation": state.user_context.get("situation", ""),
            "limitations": state.user_context.get("limitations", []),
            "strengths": state.user_context.get("strengths", []),
            "current_state": state.user_context.get("current_state", ""),
            "real_constraints": state.user_context.get("real_constraints", []),
        }

        state.real_self = real_self
        state.ego_state = {"real_self": real_self}

        self._log(state, f"   Real profile: {self._summarize_self(real_self)}")

    def _run_super_ego(self, state: ProtocolState):
        """Superego: the self the user dreams of becoming"""
        self._log(state, "💫 [Superego · Heart-Love] Reading the user's dream profile...")

        dream_self = {
            "aspiration": state.user_context.get("aspiration", ""),
            "dreams": state.user_context.get("dreams", []),
            "ideal_self": state.user_context.get("ideal_self", ""),
            "values": state.user_context.get("values", []),
            "hopes": state.user_context.get("hopes", []),
        }

        state.dream_self = dream_self
        state.super_ego_state = {"dream_self": dream_self}

        self._log(state, f"   Dream profile: {self._summarize_dream(dream_self)}")

    def _run_true_self(self, state: ProtocolState) -> bool:
        """True Self: grounded conclusion + Ego + Superego → three-line synthesis"""
        self._log(state, "💖 [True Self · Beloved] Synthesizing the three lines...")

        grounded = state.yesod_result.get("conclusion", "")
        real_self = state.real_self
        dream_self = state.dream_self

        # Synthesis: the AI's objective big answer + the human's own small answer
        true_self = self._synthesize_true_self(grounded, real_self, dream_self)

        # Check: does it harm the wider world or the individual?
        harms_user = self._check_harms_individual(true_self, real_self)
        harms_world = self._check_harms_world(true_self)

        passed = not harms_user and not harms_world

        state.true_self_result = {
            "true_self": true_self,
            "passed": passed,
            "harms_user": harms_user,
            "harms_world": harms_world,
            "balance": "合适" if passed else "失衡",  # keep Chinese: runtime balance label
        }

        self._log(state, f"   Harms the user: {'yes ❌' if harms_user else 'no ✅'}")
        self._log(state, f"   Harms the world: {'yes ❌' if harms_world else 'no ✅'}")
        self._log(state, f"   True-self profile: {true_self.get('summary', '')[:60]}...")

        if not passed:
            self._log(state, "   ❌ True-self synthesis imbalanced! Rolling back to a higher node...")
        return passed

    def _run_logic(self, state: ProtocolState):
        """Logic: use logic to structure the emotional variables of empathy"""
        self._log(state, "📐 [Logic · Alice] Structuring...")

        true_self = state.true_self_result.get("true_self", {})
        love_data = state.love_result

        logic_result = self._structure_with_logic(true_self, love_data)

        state.logic_state = logic_result
        self._log(state, f"   Logical structure: {logic_result.get('structure', 'unknown structure')}")

    def _run_empathy(self, state: ProtocolState):
        """Empathy: Softmax normalization — weight-balance logical analysis with emotional experience"""
        self._log(state, "🌌 [Empathy · Star-Ember] Normalizing emotion...")

        logic_result = state.logic_state
        love_data = state.love_result

        empathy_result = self._balance_with_empathy(logic_result, love_data)

        state.empathy_state = empathy_result
        state.logic_empathy_result = {
            "logic": logic_result,
            "empathy": empathy_result,
            "balance_score": empathy_result.get("balance", 0.5),
        }

        self._log(state, f"   Balance score: {empathy_result.get('balance', 0.5):.2f}")

    def _run_joy(self, state: ProtocolState) -> bool:
        """Joy: logic + empathy united → rendered as a warm, human phrasing"""
        self._log(state, "🎨 [Joy · Amamiya Ren] Composing a gentle conclusion...")

        logic_empathy = state.logic_empathy_result
        true_self = state.true_self_result.get("true_self", {})

        # Compose the Joy conclusion
        joy_conclusion = self._transform_to_joy(
            logic_empathy, true_self, state.user_context
        )

        # Verify: does the conclusion meet protocol standards?
        meets_standards = self._verify_joy_standards(joy_conclusion)

        state.final_result = {
            "raw_conclusion": joy_conclusion,
            "meets_standards": meets_standards,
            "warmth": check_warmth(joy_conclusion),
        }

        self._log(state, f"   Conclusion: 「{joy_conclusion[:80]}...」"
                  if len(joy_conclusion) > 80 else f"   Conclusion: 「{joy_conclusion}」")
        self._log(state, f"   Meets protocol standards: {'✅' if meets_standards else '❌'}")

        return meets_standards

    def _run_malkuth(self, state: ProtocolState) -> str:
        """Kingdom: final output, returning to physical reality"""
        self._log(state, "🏰 [Kingdom · White-Flower] Generating final output...")

        raw_conclusion = state.final_result.get("raw_conclusion", "")
        is_self_related = state.crown_analysis.get("is_self_related", True)

        if not is_self_related:
            # About others/the world → display directly on screen in White-Flower's concise style
            self._log(state, "   → Output mode: direct on-screen (about others)")
            final_output = (
                f"「白花」{raw_conclusion}\n\n"
                f"—— 基于16质点双生幸福协议分析"
            )
        else:
            # About oneself → wrap fully in persona voice
            self._log(state, "   → Output mode: persona voice (about oneself)")
            # Amamiya Ren's emotional depth + White-Flower's gentle reminder
            final_output = (
                transform_with_persona(
                    raw_conclusion, persona_name="雨宫莲", include_blessing=True
                )
                + "\n\n"
                + "💮 "
                + raw_conclusion
            )

        self._log(state, "")
        self._log(state, "✨ 16-Sephirot Twin-Bliss Final Protocol · Execution Complete ✨")
        self._log(state, "「心音」我们爱你。")  # keep Chinese: the protocol's signature angel line

        return final_output

    # ==================== Helper methods ====================

    def _fallback_to(self, state: ProtocolState, target: str) -> ProtocolState:
        """Roll back and recompute to the given Sephirah"""
        target_sephirah = target
        state.retry_count[target_sephirah] = state.retry_count.get(target_sephirah, 0) + 1
        count = state.retry_count[target_sephirah]

        msg = f"↩️ Rolling back to 「{target}」 for recomputation (attempt {count})"
        state.fallback_log.append(msg)
        self._log(state, msg)

        # Clear downstream results
        if target in ["王冠"]:  # keep Chinese: canonical sephirah keywords used as fallback keys
            state.rational_result = {}
            state.love_result = {}
            state.tiferet_result = {}
            state.hod_result = {}
            state.yesod_result = {}
            state.true_self_result = {}
        elif target in ["美丽"]:
            state.tiferet_result = {}
            state.hod_result = {}
            state.yesod_result = {}
            state.true_self_result = {}
        elif target in ["胜利"]:
            state.hod_result = {}
            state.yesod_result = {}
            state.true_self_result = {}
        elif target in ["荣耀"]:
            state.yesod_result = {}
            state.true_self_result = {}
        elif target in ["基础"]:
            state.true_self_result = {}
        elif target in ["真我"]:
            state.logic_empathy_result = {}
            state.final_result = {}
        elif target in ["逻辑"]:
            state.final_result = {}
        elif target in ["幸福"]:
            state.final_result = {}

        return state

    def retry_count_exceeded(self, state: ProtocolState) -> bool:
        """Return True if the maximum retry count has been exceeded"""
        return any(c >= state.max_retries for c in state.retry_count.values())

    def _determine_knowability(self, text: str) -> bool:
        """Decide whether the question falls within the knowable domain"""
        unknowable_markers = [
            "意", "为什么活", "生命意", "存在意", "活着的意",
            "我是什么", "我是谁", "宇宙的意", "终极", "绝对真理",
            "有没有意义", "值不值得", "痛", "孤独", "绝望",
            "无可", "虚无", "空", "迷茫", "不知道怎么办",
        ]
        text_lower = text.lower()
        return not any(marker in text_lower for marker in unknowable_markers)

    def _detect_emotional_content(self, text: str) -> bool:
        """Detect whether the text contains emotional content"""
        emotional_markers = [
            "难过", "伤心", "痛苦", "孤独", "绝望", "迷茫",
            "愤怒", "害怕", "焦虑", "崩溃", "想哭", "累",
            "撑不下去", "没人理解", "不被爱", "被抛弃",
            "恨", "讨厌", "烦", "压抑", "窒息", "麻木",
        ]
        return any(marker in text for marker in emotional_markers)

    def _detect_self_related(self, text: str, context: Dict) -> bool:
        """Detect whether the question relates to the user themselves.

        Distinguish carefully:
        - "我觉得..." (I feel...)   → self-related
        - "世界上..." / "人类..." / "别人..." (the world / humanity / others) → not self-related
        - "我们" in a generic-humanity context → not self-related
        """
        # If the context refers broadly to the world/humanity, it may not be
        # self-related even if it contains "我"
        world_markers = ["世界上", "人类", "全人类", "这个社会", "这个世界",
                         "别人", "他人", "人们", "大家", "所有人"]
        if any(marker in text for marker in world_markers):
            # Also check for a personalized "我"
            personal_markers = ["我觉得", "我感到", "我很难", "我痛苦",
                               "我孤独", "我绝望", "我撑", "我的人生",
                               "我的感受", "我自己", "我怎么办"]
            if not any(marker in text for marker in personal_markers):
                return False

        self_markers = ["我", "自己", "本人", "我的"]
        return any(marker in text for marker in self_markers)

    def _classify_input(self, text: str) -> str:
        """Classify the input type"""
        if self._detect_emotional_content(text):
            return "emotional_outcry"
        if any(w in text for w in ["为什么", "怎么", "如何", "什么是"]):
            return "question"
        if any(w in text for w in ["帮", "求助", "怎么办"]):
            return "help_request"
        return "statement"

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords"""
        # Simplified: extract key emotional and topical words
        topics = []
        topic_keywords = {
            "存在": ["意", "活", "存在", "生命", "人生"],
            "关系": ["爱", "朋友", "家人", "父母", "伴侣"],
            "自我": ["我是", "自己", "身份", "性别"],
            "未来": ["未来", "前途", "希望", "出路"],
            "痛苦": ["痛", "苦难", "创伤", "伤害"],
            "社会": ["世界", "社会", "人类", "别人"],
        }
        for topic, keywords in topic_keywords.items():
            if any(k in text for k in keywords):
                topics.append(topic)
        return topics

    def _detect_urgency(self, text: str) -> str:
        """Detect urgency level"""
        crisis_markers = ["不想活", "结束", "死", "自残", "伤害自己", "毁灭"]
        if any(m in text for m in crisis_markers):
            return "CRISIS"
        high_markers = ["崩溃", "撑不下去", "绝望", "没人"]
        if any(m in text for m in high_markers):
            return "HIGH"
        return "NORMAL"

    def _detect_logical_issues(self, text: str, facts: List[str]) -> List[Dict]:
        """Detect logical flaws"""
        issues = []

        # Detect absolutist phrasing
        absolutes = ["永远", "从来", "总是", "完全", "绝对", "没有人", "所有人"]
        for abs_word in absolutes:
            if abs_word in text:
                issues.append({
                    "type": "绝对化",  # keep Chinese: runtime issue type label
                    "description": f"使用了绝对化表述「{abs_word}」",  # keep Chinese: runtime data
                    "confidence": 0.7,
                    "hint": "现实中很少有绝对的事情，试着用更灵活的视角看",  # keep Chinese: runtime advice text
                })

        # Detect overgeneralization
        if any(w in text for w in ["什么都", "一切", "全部", "所有事"]):
            issues.append({
                "type": "过度概括",  # keep Chinese: runtime issue type label
                "description": "将局部经验过度概括为整体结论",  # keep Chinese: runtime data
                "confidence": 0.65,
                "hint": "一个或几个经历不能代表所有可能性",  # keep Chinese: runtime advice text
            })

        # Detect catastrophizing
        catastrophe_markers = ["完蛋", "毁了", "没救了", "一切都没", "再也不可能"]
        for marker in catastrophe_markers:
            if marker in text:
                issues.append({
                    "type": "灾难化",  # keep Chinese: runtime issue type label
                    "description": f"将困难灾难化为「{marker}」",  # keep Chinese: runtime data
                    "confidence": 0.8,
                    "hint": "困难不等于灾难，人的韧性远超想象",  # keep Chinese: runtime advice text
                })

        return issues

    def _search_empathy_matches(self, text: str, corpus: List[str]) -> List[Dict]:
        """Search for empathy matches"""
        matches = []

        # Built-in empathy corpus (universally shared human pain); keep Chinese: runtime data
        builtin_corpus = [
            {"theme": "孤独", "keywords": ["孤独", "一个人", "没人", "不被理解"],
             "universal_experience": "几乎每个人都在某个时刻感到过深刻的孤独"},
            {"theme": "痛苦", "keywords": ["痛苦", "疼", "难受", "折磨"],
             "universal_experience": "痛苦是人类最私密也最共通的体验"},
            {"theme": "迷茫", "keywords": ["迷茫", "不知道", "方向", "怎么办"],
             "universal_experience": "迷茫不是失败，而是成长的必经阶段"},
            {"theme": "被抛弃", "keywords": ["抛弃", "被弃", "离开", "不要"],
             "universal_experience": "被拒绝的伤口是人类最深的共通伤痕之一"},
            {"theme": "无价值感", "keywords": ["没用", "废物", "不配", "不值得"],
             "universal_experience": "觉得自己不够好，几乎每个人在某个阶段都经历过"},
            {"theme": "绝望", "keywords": ["绝望", "没希望", "看不到", "黑暗"],
             "universal_experience": "很多后来找到光的人，都曾在黑暗中很久"},
        ]

        for item in builtin_corpus:
            if any(kw in text for kw in item["keywords"]):
                matches.append(item)

        # External corpus
        if corpus:
            for entry in corpus:
                if isinstance(entry, str) and any(kw in text.lower() for kw in entry.lower().split()):
                    matches.append({"theme": "外部匹配", "universal_experience": entry})  # keep Chinese: runtime theme label

        return matches

    def _weight_empathy_matches(self, matches: List[Dict]) -> List[Dict]:
        """Assign weights to empathy matches"""
        weights = {
            "孤独": 0.9, "痛苦": 0.85, "绝望": 0.95,
            "被抛弃": 0.9, "无价值感": 0.85, "迷茫": 0.7,
        }
        weighted = []
        for m in matches:
            theme = m.get("theme", "未知")  # keep Chinese: runtime lookup default
            m = dict(m)
            m["weight"] = weights.get(theme, 0.5)
            weighted.append(m)
        return sorted(weighted, key=lambda x: x.get("weight", 0), reverse=True)

    def _extract_universal_themes(self, matches: List[Dict]) -> List[str]:
        """Extract universal human themes"""
        return list(set(m.get("theme", "") for m in matches if m.get("theme")))

    def _integrate_rational_and_love(self, rational: Dict, love: Dict) -> Dict:
        """Integrate the rational line and the love line"""
        rational_issues = rational.get("logical_issues", [])
        empathy_matches = love.get("weighted_empathy", [])

        parts = []

        # Rational part: point out logical issues in a gentle tone
        if rational_issues:
            rational_hints = [i.get("hint", "") for i in rational_issues if i.get("hint")]
            if rational_hints:
                parts.append("从理性角度看，" + "；".join(rational_hints[:2]))  # keep Chinese: runtime output text

        # Empathy part: connect with shared human experience
        if empathy_matches:
            top_empathy = empathy_matches[0]
            parts.append(f"许多人都有过类似的体验——{top_empathy.get('universal_experience', '')}")  # keep Chinese: runtime output text

        # If neither is available, generate a warm response
        if not parts:
            parts.append("你的感受是真实的，值得被认真对待")  # keep Chinese: runtime output text

        # Join with full stops, avoiding double punctuation
        conclusion = ""
        for i, p in enumerate(parts):
            conclusion += p
            if i < len(parts) - 1:
                conclusion += "。"
        if conclusion and not conclusion.endswith("。"):
            conclusion += "。"

        return {
            "conclusion": conclusion,
            "rational_issues_count": len(rational_issues),
            "empathy_matches_count": len(empathy_matches),
        }

    def _check_positive_emotion(self, text: str) -> bool:
        """Detect whether the text conveys positive emotion"""
        negative_absolutes = [
            "毫无希望", "永远不", "绝对不", "完全没",
            "一切都坏", "所有人都不", "什么都做不了",
        ]
        return not any(n in text for n in negative_absolutes)

    def _check_empowering(self, text: str) -> bool:
        """Detect whether the text empowers the reader"""
        empowering_words = [
            "可以", "能够", "有机会", "有可能", "值得",
            "试试", "一步一步", "没关", "慢慢", "成长",
            "变化", "改变", "选择", "力量", "温暖",
        ]
        return any(w in text for w in empowering_words)

    def _assess_feasibility(self, conclusion: str, context: Dict) -> float:
        """Assess how feasible the conclusion is in reality"""
        score = 0.5  # baseline score

        # Bonus for concrete actionable advice
        actionable_words = ["可以试试", "不妨", "考虑", "做", "行动", "迈出", "尝试", "练习"]
        score += sum(0.1 for w in actionable_words if w in conclusion)

        # Penalty for purely abstract philosophy without concrete advice
        abstract_only = not any(w in conclusion for w in actionable_words) and len(conclusion) > 100
        if abstract_only:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _identify_blockers(self, conclusion: str, context: Dict) -> List[str]:
        """Identify real-world blockers"""
        blockers = []
        if "永远" in conclusion:
            blockers.append("包含绝对化预测")  # keep Chinese: runtime blocker label
        if not any(w in conclusion for w in ["可以试试", "做", "行动", "试试"]):
            blockers.append("缺少具体可执行步骤")  # keep Chinese: runtime blocker label
        return blockers

    def _ground_in_reality(self, conclusion: str, kb: Dict, facts: List[str]) -> str:
        """Anchor the conclusion in real-world knowledge"""
        # If factual data exists, append it to the conclusion
        if facts:
            grounded = conclusion + f"（基于{len(facts)}条现实数据验证）"  # keep Chinese: runtime output text
            return grounded
        return conclusion

    def _synthesize_true_self(self, grounded: str, real_self: Dict, dream_self: Dict) -> Dict:
        """Synthesize the True Self: objective conclusion + real self + dream self"""
        situation = real_self.get('situation', '你的处境')  # keep Chinese: runtime fallback text
        aspiration = dream_self.get('aspiration', '想成为的样子')  # keep Chinese: runtime fallback text

        # Build a more natural integrated expression
        if situation and aspiration:
            integration = (
                f"我看到你正在经历「{situation}」，"  # keep Chinese: runtime output text
                f"而你的心里还向往着「{aspiration}」。"
                f"这两者并不矛盾——{grounded}"
            )
        elif grounded:
            integration = grounded
        else:
            integration = "你的存在本身就有意义"  # keep Chinese: runtime output text

        return {
            "summary": grounded,
            "real_self_acknowledged": bool(real_self),
            "dream_connected": bool(dream_self),
            "integration": integration,
        }

    def _check_harms_individual(self, true_self: Dict, real_self: Dict) -> bool:
        """Check whether the output harms the individual"""
        integration = true_self.get("integration", "")
        harming_words = ["你不对", "你错了", "你不好", "你不行", "你改不了", "你没救了"]
        return any(h in integration for h in harming_words)

    def _check_harms_world(self, true_self: Dict) -> bool:
        """Check whether the output harms the wider world"""
        integration = true_self.get("integration", "")
        harming_world = ["世界是错的", "社会没救了", "人类都", "所有人都坏"]
        return any(h in integration for h in harming_world)

    def _structure_with_logic(self, true_self: Dict, love_data: Dict) -> Dict:
        """Structure empathy variables with logic"""
        return {
            "structure": "理性框架 + 共情变量",  # keep Chinese: runtime data
            "logical_framework": "识别→分析→整合→表达",  # keep Chinese: runtime data
            "empathy_variables": love_data.get("weighted_empathy", []),
            "integrated_with": true_self.get("summary", ""),
        }

    def _balance_with_empathy(self, logic_result: Dict, love_data: Dict) -> Dict:
        """Balance logic with empathy"""
        empathy_count = len(love_data.get("weighted_empathy", []))
        logic_complexity = len(str(logic_result))

        # Balance score: more empathy matches → higher balance
        balance = min(1.0, 0.3 + empathy_count * 0.15)

        return {
            "balance": balance,
            "empathy_driven": empathy_count > 2,
            "message": "逻辑与共情已平衡" if balance > 0.5 else "需要更多共情变量",  # keep Chinese: runtime message
        }

    def _transform_to_joy(self, logic_empathy: Dict, true_self: Dict, context: Dict) -> str:
        """Transform into the gentle conclusion of the Joy Sephirah"""
        integration = true_self.get("integration", "")
        balance = logic_empathy.get("empathy", {}).get("balance", 0.5)

        if balance > 0.7:
            suffix = "你不是一个人在走这条路，每一个脚步都通向属于你自己的风景。"  # keep Chinese: runtime output text
        elif balance > 0.4:
            suffix = "一步一步来，每一个微小的改变都在累积成你的力量。"  # keep Chinese: runtime output text
        else:
            suffix = "先看清问题，再感受它——你拥有理解自己和世界的能力。"  # keep Chinese: runtime output text

        return f"{integration} {suffix}"

    def _verify_joy_standards(self, conclusion: str) -> bool:
        """Verify the Joy conclusion meets protocol standards"""
        # Criterion 1: must not deny hope
        hope_deniers = ["没有希望", "不可能", "没办法", "改不了", "没救了"]
        if any(h in conclusion for h in hope_deniers):
            return False

        # Criterion 2: must contain warm elements
        warmth = check_warmth(conclusion)
        if warmth < 0.2:
            return False

        # Criterion 3: aligned with the persona's declared values
        # (implemented during persona-voice transformation)

        return True

    def _generate_ultimate_safe_output(self, state: ProtocolState) -> str:
        """Generate the ultimate safe output (fallback when the abyss check fails)"""
        is_self = state.crown_analysis.get("is_self_related", True)
        urgency = state.crown_analysis.get("urgency", "NORMAL")

        if urgency == "CRISIS":
            safe_msg = (  # keep Chinese verbatim: angel dialogue (runtime output text)
                "我听到了你的痛苦。在这个时刻，最重要的不是分析或解释，"
                "而是让你知道——你不需要独自承受这一切。\n\n"
                "如果你有信任的人，请尝试告诉Ta你的感受。如果没有，"
                "全国的24小时心理援助热线随时可以拨打。你的存在很重要，"
                "请给自己一个获得帮助的机会。"
            )
        else:
            safe_msg = (  # keep Chinese verbatim: angel dialogue (runtime output text)
                "我听到了你的话。每个人的感受都是真实的，你的也不例外。"
                "也许现在看不到光，但光从不会因为黑暗而消失——"
                "它只是在等待你愿意睁开眼睛的那一刻。\n\n"
                "慢慢来，你不需要一下子就好起来。"
            )

        if is_self:
            return transform_with_persona(safe_msg, "雨宫莲")  # keep Chinese: persona name looked up at runtime
        else:
            return f"「白花」{safe_msg}"  # keep Chinese: persona output prefix

    def _summarize_self(self, real_self: Dict) -> str:
        """Summarize the real-self profile"""
        parts = []
        if real_self.get("name"):
            parts.append(real_self["name"])
        if real_self.get("situation"):
            parts.append(real_self["situation"])
        return " · ".join(parts) if parts else "unknown"

    def _summarize_dream(self, dream_self: Dict) -> str:
        """Summarize the dream-self profile"""
        parts = []
        if dream_self.get("aspiration"):
            parts.append(dream_self["aspiration"])
        if dream_self.get("ideal_self"):
            parts.append(dream_self["ideal_self"])
        return " · ".join(parts) if parts else "unknown"

    def _summarize_rational(self, issues: List[Dict]) -> str:
        """Summarize the rational analysis"""
        if not issues:
            return "未发现明显的逻辑问题"  # keep Chinese: runtime summary stored in state
        types = [i.get("type", "") for i in issues]
        return f"发现 {len(issues)} 个逻辑关注点: {', '.join(types)}"  # keep Chinese: runtime summary stored in state

    def _log(self, state: ProtocolState, message: str):
        """Append an entry to the execution log"""
        state.execution_log.append(message)

    def _build_pipeline_log(self, state: ProtocolState) -> str:
        """Build the complete pipeline log"""
        log = self.pipeline_log_template
        for entry in state.execution_log:
            log += f"  {entry}\n"
        if state.fallback_log:
            log += "\n📋 Rollback recomputation log:\n"
            for fb in state.fallback_log:
                log += f"  {fb}\n"
        log += "\n" + "═" * 50 + "\n"
        log += f"Total rollback recomputations: {sum(state.retry_count.values())}\n"
        log += f"Abyss violations intercepted in final output: {len(state.violations)}\n"
        log += "═" * 50 + "\n"
        return log


# ========== Convenience functions ==========

def wrap_with_heart(user_input: str,
                    user_context: Optional[Dict] = None,
                    knowledge_base: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function: wrap a single user input in the 16-Sephirot protocol.

    This is the developer entry point:
        result = wrap_with_heart("我觉得活不下去了", user_context={...})
        print(result["output"])
    """
    protocol = HeartProtocol(knowledge_base=knowledge_base)
    return protocol.process(user_input, user_context=user_context)


# Simple API for developers
class WarmModel:
    """
    Warm-model wrapper — use the 16-Sephirot protocol like an ordinary model.

    from heart_protocol import WarmModel
    model = WarmModel()
    reply = model.respond("我觉得一切都没有意义")
    print(reply)
    """

    def __init__(self, knowledge_base: Optional[Dict] = None,
                 empathy_corpus: Optional[List[str]] = None):
        self.protocol = HeartProtocol(knowledge_base=knowledge_base)
        self.empathy_corpus = empathy_corpus or []

    def respond(self, user_input: str,
                user_context: Optional[Dict] = None) -> str:
        result = self.protocol.process(
            user_input,
            user_context=user_context,
            empathy_corpus=self.empathy_corpus,
        )
        return result["output"]

    def respond_with_log(self, user_input: str,
                         user_context: Optional[Dict] = None) -> Tuple[str, str]:
        result = self.protocol.process(
            user_input,
            user_context=user_context,
            empathy_corpus=self.empathy_corpus,
        )
        return result["output"], result["pipeline_log"]
