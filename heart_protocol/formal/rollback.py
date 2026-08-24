# -*- coding: utf-8 -*-
"""
Recompute from the parent stage — context snapshot recovery + search-tree rollback
==================================================================================

Philosophical source (Glory · Hod / Victory · Netzach):
    "When an inspected conclusion … does not pass, roll back to 美丽 (Beauty) and recompute"
    "When it cannot be executed, roll back to the parent stage and recompute"

Formal translation:
    One protocol run is a search tree T:
      · node = (stage state s, retry strategy π used)
      · edge = recomputing that stage after applying strategy π: δ(s, π)
    "Recompute from the parent stage" = backtrack from a failed node to its
    parent and expand along an untried strategy edge instead.

    This module provides two standard tree searches:
      · Beam Search Rollback — keeps the beam_width best candidates per level;
        suited to "quickly find the first qualifying answer" (engineering default).
      · MCTS Rollback — UCB1 selection + expansion + simulation + backpropagation;
        suited to "approach the best expression within budget" (quality-first).

    A Snapshot is an immutable checkpoint: a deep copy of the state plus the
    path record; restoring means expanding directly from that snapshot as the
    parent — the original state is never polluted.
"""

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ==================== Retry strategy space ====================


@dataclass
class RetryStrategy:
    """One adjustment strategy applied during a rollback recomputation"""
    id: str
    description: str
    mutate: Callable[[Dict[str, Any]], Dict[str, Any]]

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        return self.mutate(new_state)


def _boost(field_key: str, delta: float) -> Callable[[Dict], Dict]:
    def mutate(state: Dict) -> Dict:
        state[field_key] = min(1.0, float(state.get(field_key, 0.5)) + delta)
        return state
    return mutate


def _set_flag(key: str, value: Any) -> Callable[[Dict], Dict]:
    def mutate(state: Dict) -> Dict:
        state[key] = value
        return state
    return mutate


# Strategy ids are stable runtime identifiers and stay as-is. The description
# strings are Chinese runtime data (display text) and stay verbatim; glosses:
#   baseline        "recompute directly from the current state"
#   empathy_boost   "raise the empathy weight (+0.25)"
#   logic_boost     "raise the logic weight (+0.25)"
#   concrete_steps  "force-inject concrete executable steps"
#   soften_tone     "soften the tone, reduce absolute statements"
#   shorten         "compress the length, focus on the core"
DEFAULT_STRATEGIES: List[RetryStrategy] = [
    RetryStrategy("baseline", "按当前状态直接重算", lambda s: s),
    RetryStrategy("empathy_boost", "提高共情权重(+0.25)", _boost("empathy_weight", 0.25)),
    RetryStrategy("logic_boost", "提高逻辑权重(+0.25)", _boost("logic_weight", 0.25)),
    RetryStrategy("concrete_steps", "强制注入具体可执行步骤",
                  _set_flag("force_concrete_steps", True)),
    RetryStrategy("soften_tone", "软化语气、降低绝对化表述",
                  _set_flag("tone", "gentle")),
    RetryStrategy("shorten", "压缩篇幅、聚焦核心",
                  _set_flag("target_length", "short")),
]


# ==================== Snapshot ====================


@dataclass
class Snapshot:
    """Immutable context checkpoint"""
    sephirah: str                        # the sephirah stage it belongs to
    attempt: int                         # attempt number
    state: Dict[str, Any]                # deep copy of the stage state
    output: str = ""                     # output text of this attempt
    score: float = float("-inf")         # validator score
    strategy_id: str = "root"            # which strategy produced it
    parent: Optional["Snapshot"] = field(default=None, repr=False)

    @property
    def depth(self) -> int:
        d, node = 0, self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def path(self) -> List["Snapshot"]:
        """Full path from root to the current node"""
        out, node = [], self
        while node is not None:
            out.append(node)
            node = node.parent
        return list(reversed(out))

    def restore(self) -> Dict[str, Any]:
        """Restore to this checkpoint (returns an independent copy of the state)"""
        return copy.deepcopy(self.state)


# ==================== Rollback search engine ====================

ComputeFn = Callable[[Dict[str, Any], RetryStrategy], Tuple[Dict[str, Any], str]]
ValidateFn = Callable[[str], float]


class RollbackEngine:
    """
    Standardized "recompute from the parent stage" engine.

    Args:
        compute_fn:  (state, strategy) -> (new_state, output_text)
                     —— the re-entrant computation function δ(s, π) of the rolled-back stage
        validate_fn: (output_text) -> score ∈ [0,1]
                     —— the stage verification gate (warmth / feasibility / invariant weighting, etc.)
        strategies:  retry strategy space (defaults to DEFAULT_STRATEGIES)
        pass_score:  early-termination threshold (score ≥ pass_score counts as verified)
        seed:        MCTS random seed (None = not fixed)
    """

    def __init__(self,
                 compute_fn: ComputeFn,
                 validate_fn: ValidateFn,
                 strategies: Optional[List[RetryStrategy]] = None,
                 pass_score: float = 0.6,
                 seed: Optional[int] = None):
        self.compute_fn = compute_fn
        self.validate_fn = validate_fn
        self.strategies = strategies or list(DEFAULT_STRATEGIES)
        self.pass_score = pass_score
        self.rng = random.Random(seed)
        self.expansions = 0          # total recomputation count (for latency statistics)

    # ---------- Public ----------

    def _evaluate(self, parent: Snapshot, strategy: RetryStrategy) -> Snapshot:
        new_state, output = self.compute_fn(parent.state, strategy)
        snap = Snapshot(
            sephirah=parent.sephirah,
            attempt=parent.attempt + 1,
            state=new_state,
            output=output,
            score=self.validate_fn(output),
            strategy_id=strategy.id,
            parent=parent,
        )
        self.expansions += 1
        return snap

    def _make_root(self, sephirah: str, state: Dict[str, Any],
                   initial_output: str = "") -> Snapshot:
        return Snapshot(sephirah=sephirah, attempt=0,
                        state=copy.deepcopy(state),
                        output=initial_output, strategy_id="root")

    # ---------- Beam Search rollback ----------

    def beam_search_rollback(self,
                             sephirah: str,
                             state: Dict[str, Any],
                             initial_output: str = "",
                             beam_width: int = 3,
                             max_depth: int = 3) -> Snapshot:
        """
        Beam-search rollback: at every level each node in the beam tries all
        strategies, keeping the top-beam_width; any candidate ≥ pass_score is
        returned immediately.

        Complexity: O(max_depth × beam_width × |strategies|) recomputations.
        """
        root = self._make_root(sephirah, state, initial_output)
        frontier: List[Snapshot] = [root]

        for _ in range(max_depth):
            candidates: List[Snapshot] = []
            for node in frontier:
                for strat in self.strategies:
                    if strat.id == "baseline" and node.strategy_id == strat.id \
                            and node.attempt > 0:
                        continue          # do not re-expand the same strategy
                    cand = self._evaluate(node, strat)
                    if cand.score >= self.pass_score:
                        return cand       # adopt the first qualifying solution immediately (online scenario)
                    candidates.append(cand)
            if not candidates:
                break
            candidates.sort(key=lambda s: s.score, reverse=True)
            frontier = candidates[:beam_width]

        best = max((c for c in frontier if c.attempt > 0),
                   key=lambda s: s.score, default=root)
        return best

    # ---------- MCTS rollback ----------

    class _Node:
        __slots__ = ("snap", "parent", "children", "untried", "visits", "value")

        def __init__(self, snap: "Snapshot", parent: Optional["RollbackEngine._Node"],
                     untried: List[RetryStrategy]):
            self.snap = snap
            self.parent = parent
            self.children: List["RollbackEngine._Node"] = []
            self.untried = list(untried)
            self.visits = 0
            self.value = 0.0

        def ucb1(self, c: float) -> float:
            if self.visits == 0:
                return float("inf")
            exploit = self.value / self.visits
            explore = c * math.sqrt(math.log(self.parent.visits) / self.visits) \
                if self.parent else 0.0
            return exploit + explore

    def mcts_rollback(self,
                      sephirah: str,
                      state: Dict[str, Any],
                      initial_output: str = "",
                      iterations: int = 16,
                      exploration_c: float = 1.414,
                      max_depth: int = 3) -> Snapshot:
        """
        MCTS rollback: UCB1 selection → randomly expand an untried strategy →
        simulation score → backpropagate the mean.
        Returns the snapshot with the highest visit value. Suited to offline refinement.
        """
        root_snap = self._make_root(sephirah, state, initial_output)
        root = self._Node(root_snap, None, list(self.strategies))
        best = root_snap

        for _ in range(iterations):
            node = root
            # 1) Selection: walk the UCB1-best child along the expanded path
            while not node.untried and node.children and node.snap.depth < max_depth:
                node = max(node.children, key=lambda ch: ch.ucb1(exploration_c))

            # 2) Expansion: try one unused strategy
            if node.untried and node.snap.depth < max_depth:
                strat = node.untried.pop(self.rng.randrange(len(node.untried)))
                child_snap = self._evaluate(node.snap, strat)
                child = self._Node(child_snap, node, [s for s in self.strategies
                                                      if s.id != strat.id])
                node.children.append(child)
                node = child

            # 3) Simulation: the score IS the simulation result (the validator is
            #    deterministic, so no rollout is needed)
            reward = node.snap.score
            if node.snap.attempt > 0 and node.snap.score > best.score:
                best = node.snap
            if node.snap.score >= self.pass_score:
                reward += 1.0             # bonus reward for passing the verification gate

            # 4) Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        return best

    # ---------- Integration with protocol traces ----------

    @staticmethod
    def to_stage_record(best: Snapshot) -> Dict[str, Any]:
        """Convert a rollback result into fields compatible with formal.spec.StageRecord"""
        return {
            "sephirah": best.sephirah,
            "attempt": max(1, best.attempt),
            "output_text": best.output,
            "score": best.score,
            "strategy": best.strategy_id,
        }
