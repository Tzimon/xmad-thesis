"""
xmad/3_iteration/xmad_debate/session.py

The shared context session (C1.3): holds the growing debate transcript for one
question and renders each agent's own view of it.

Copied from Phase 2 with three additions for Phase 3:
  * AgentTurn records `truncated` (replies cut off at max_tokens),
    `api_retries` (connection errors that were retried) and `forced_final`
    (answer requested without tools after the turn budget was used up);
  * final_answer() returns source "pending_judge" while the judge has not run
    yet (the judge runs as a separate pass, see phase3.py judge);
  * DebateSession.from_dict() rebuilds a session from its saved JSON, so the
    judge pass works on exactly the stored state.

Design choices, and why:

* Round 1 is independent: neither agent sees the other before answering. This
  yields a majority-vote-only baseline for free from the same run (no extra
  inference), isolating the contribution of the exchange itself from the
  contribution of simply combining two independent predictions.

* A write is only accepted once both agents have finished a round
  (`record_turn` fills one of two slots; the round is not visible to future
  rendering until `both_done()`). This is the enforcement point for "the
  shared context session accepts writes only once both agents have completed
  a round": there is no way for one agent's round-r reply to leak into the
  other agent's round-r prompt.

* What carries forward between rounds is bounded: only each agent's
  `reasoning_summary` (<=3 sentences) and `final_answer` from every prior
  round -- not the full raw reply with its tool-call reasoning. This keeps
  prompt growth roughly linear in the number of ROUNDS, not in the number of
  tool calls made along the way. The full raw reply is still kept in the
  session for logging, just not re-injected as literal chat history.

* Rendering is agent-relative: every agent is shown "your reasoning" / "the
  other agent's reasoning" -- never a fixed "Agent 1"/"Agent 2" label -- so
  the prompt text itself cannot structurally privilege one agent (C1.4).

Debate results are evaluated against two baselines outside this module: a
debate between two instances of the SAME model (the homogeneous condition,
run via this identical session/loop with agent_a and agent_b pointed at the
same endpoint) and a single-agent baseline. In Phase 3 the single-agent
baseline is each agent's independent round-1 answer (same prompt, same
decoding), recorded per question in debate_results.csv.

Reuses metrics.normalize_answer from the Phase 1 package (xmad_rag), installed
editable in the same environment, so consensus checking uses the identical
normalisation as the official scoring.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from xmad_rag.metrics import normalize_answer

AGENT_KEYS = ("agent_a", "agent_b")

# Answer recorded when an agent produced no answer at all in a round. It is
# NOT an answer: two agents that both failed must not count as a consensus,
# and repeating it is not a stalemate (see is_answer()).
NO_ANSWER = "(no final reply within max_turns)"


def is_answer(text) -> bool:
    """True for a real answer (not empty, not the NO_ANSWER placeholder)."""
    norm = normalize_answer(text or "")
    return bool(norm) and norm != normalize_answer(NO_ANSWER)


def _other(agent_key: str) -> str:
    if agent_key not in AGENT_KEYS:
        raise ValueError(f"unknown agent key: {agent_key!r}, expected one of {AGENT_KEYS}")
    return "agent_b" if agent_key == "agent_a" else "agent_a"


def extract_fields(raw: str) -> tuple[str, str]:
    """Pull 'Reasoning summary: ...' and 'Final answer: ...' out of a raw reply.

    Best-effort and never raises: a model that drifts from the exact format
    (especially a smaller or weaker one) still yields usable, non-empty
    fields rather than crashing a multi-hour run.
    """
    def _after_marker(text, marker):
        idx = text.lower().rfind(marker.lower())
        if idx == -1:
            return None
        tail = text[idx + len(marker):].strip()
        for other_marker in ("final answer:", "reasoning summary:"):
            if other_marker == marker.lower():
                continue
            cut = tail.lower().find(other_marker)
            if cut != -1:
                tail = tail[:cut].strip()
        return tail.strip().strip('"').strip(".").strip()

    final_answer = _after_marker(raw, "Final answer:")
    reasoning_summary = _after_marker(raw, "Reasoning summary:")

    if final_answer is None:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        final_answer = lines[-1] if lines else raw.strip()

    if reasoning_summary is None:
        sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
        reasoning_summary = " ".join(s for s in sentences[:3] if s).strip()
    else:
        sentences = re.split(r"(?<=[.!?])\s+", reasoning_summary)
        reasoning_summary = " ".join(s for s in sentences[:3] if s).strip()

    return reasoning_summary, final_answer


@dataclass
class AgentTurn:
    raw: str
    reasoning_summary: str
    final_answer: str
    tool_trace: list = field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    truncated: int = 0
    api_retries: int = 0
    forced_final: int = 0
    invalid_tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_raw(cls, raw, **stats):
        summary, answer = extract_fields(raw)
        return cls(raw=raw, reasoning_summary=summary, final_answer=answer, **stats)


@dataclass
class DebateRound:
    round_num: int
    turns: dict = field(default_factory=dict)  # agent_key -> AgentTurn

    def both_done(self) -> bool:
        return all(k in self.turns for k in AGENT_KEYS)


class DebateSession:
    """Holds the full state for one question's debate."""

    def __init__(self, question, agent_aliases, max_rounds, stalemate_patience=1):
        """
        agent_aliases: {"agent_a": "qwen", "agent_b": "llama"} -- for logging only,
            never used in rendering (which stays "you"/"the other agent").
        stalemate_patience: how many CONSECUTIVE completed rounds must show
            BOTH agents repeating their own previous final_answer (while still
            disagreeing with each other) before the debate is ended early and
            handed to the judge, instead of running to max_rounds. Detected
            programmatically (string comparison, no extra inference), exactly
            like the consensus check itself. Default 1: a single repeat ends
            the debate. See check_stalemate() for the rationale.
        """
        self.question = question
        self.agent_aliases = dict(agent_aliases)
        self.max_rounds = max_rounds
        self.stalemate_patience = stalemate_patience
        self.rounds: list[DebateRound] = []
        self.consensus_round: int | None = None
        self.stalemate_round: int | None = None
        self.judge_verdict: dict | None = None
        self.started_at = datetime.now().isoformat(timespec="seconds")

    # ---- writing ------------------------------------------------------
    def current_round(self) -> DebateRound:
        """The round currently being filled; starts a new one if needed."""
        if not self.rounds or self.rounds[-1].both_done():
            self.rounds.append(DebateRound(round_num=len(self.rounds) + 1))
        return self.rounds[-1]

    def record_turn(self, agent_key, raw, **stats) -> AgentTurn:
        if agent_key not in AGENT_KEYS:
            raise ValueError(f"unknown agent key: {agent_key!r}")
        rnd = self.current_round()
        if agent_key in rnd.turns:
            raise RuntimeError(
                f"{agent_key} already recorded for round {rnd.round_num}; "
                f"a round accepts exactly one turn per agent"
            )
        turn = AgentTurn.from_raw(raw, **stats)
        rnd.turns[agent_key] = turn
        return turn

    # ---- reading / rendering -------------------------------------------
    def completed_rounds(self) -> list[DebateRound]:
        """Rounds visible to future rendering: only those where BOTH agents
        have finished. This is the enforcement point for 'writes only once
        both agents have completed a round'."""
        return [r for r in self.rounds if r.both_done()]

    def render_context_for(self, agent_key) -> str:
        """Render the debate-so-far as a text block from `agent_key`'s point
        of view: 'you' vs 'the other agent', never a fixed agent label.
        Empty string for round 1 (no history yet -> independent answer)."""
        done = self.completed_rounds()
        if not done:
            return ""
        other_key = _other(agent_key)
        blocks = []
        for rnd in done:
            mine, theirs = rnd.turns[agent_key], rnd.turns[other_key]
            blocks.append(
                f"Round {rnd.round_num}:\n"
                f"  Your reasoning summary: {mine.reasoning_summary}\n"
                f"  Your answer: {mine.final_answer}\n"
                f"  The other agent's reasoning summary: {theirs.reasoning_summary}\n"
                f"  The other agent's answer: {theirs.final_answer}"
            )
        return "Debate so far (most recent round last):\n\n" + "\n\n".join(blocks)

    def render_transcript_for_judge(self) -> str:
        """Full, neutral transcript for the judge. Fixed 'Agent A'/'Agent B'
        labels are fine here: the judge does not participate in the debate,
        so C1.4's role-symmetry constraint (which applies to the debating
        agents) is not at stake for this external, evidence-grounded
        evaluator."""
        blocks = []
        for rnd in self.completed_rounds():
            a, b = rnd.turns["agent_a"], rnd.turns["agent_b"]
            blocks.append(
                f"Round {rnd.round_num}:\n"
                f"  Agent A reasoning summary: {a.reasoning_summary}\n"
                f"  Agent A answer: {a.final_answer}\n"
                f"  Agent B reasoning summary: {b.reasoning_summary}\n"
                f"  Agent B answer: {b.final_answer}"
            )
        return "\n\n".join(blocks)

    def next_round_number(self) -> int:
        return len(self.completed_rounds()) + 1

    # ---- consensus ---------------------------------------------------------
    def check_consensus(self) -> bool:
        """True if the most recently COMPLETED round's two final answers
        agree, under the same normalisation used for scoring (metrics.py)."""
        done = self.completed_rounds()
        if not done:
            return False
        last = done[-1]
        a = last.turns["agent_a"].final_answer
        b = last.turns["agent_b"].final_answer
        agree = is_answer(a) and is_answer(b) and normalize_answer(a) == normalize_answer(b)
        if agree and self.consensus_round is None:
            self.consensus_round = last.round_num
        return agree

    def is_done(self) -> bool:
        """True once consensus is reached, a stalemate is detected, or the
        round budget is exhausted."""
        if self.check_consensus():
            return True
        if self.check_stalemate():
            return True
        return len(self.completed_rounds()) >= self.max_rounds

    def check_stalemate(self) -> bool:
        """True if, over the last `stalemate_patience` consecutive completed
        rounds, BOTH agents repeated their own previous round's final_answer
        (normalised) while still disagreeing with each other.

        Rationale: under deterministic (temperature=0) decoding, an agent
        whose answer stops changing is very unlikely to spontaneously change
        it again on a later round if the debate context around it is also no
        longer moving -- continuing to spend rounds on a debate that has
        stopped moving buys essentially nothing (observed directly in
        practice: a stalled debate can repeat near-verbatim text for many
        rounds before hitting max_rounds). This is a strong empirical
        stopping heuristic, not a strict mathematical guarantee -- a
        reasoning_summary's wording can still drift slightly even when the
        final_answer it concludes with does not -- which is why it is left
        configurable via stalemate_patience rather than hard-coded.

        Consensus takes priority: if the two agents' answers now also agree,
        this returns False (that is a consensus, not a stalemate) and
        check_consensus() handles it.
        """
        if self.check_consensus():
            return False
        done = self.completed_rounds()
        window_size = self.stalemate_patience + 1
        if len(done) < window_size:
            return False
        window = done[-window_size:]
        for key in ("agent_a", "agent_b"):
            if not all(is_answer(r.turns[key].final_answer) for r in window):
                return False  # an agent without an answer gets further rounds, it is not "stuck"
            answers = {normalize_answer(r.turns[key].final_answer) for r in window}
            if len(answers) != 1:
                return False  # this agent changed somewhere in the window
        if self.stalemate_round is None:
            self.stalemate_round = window[-1].round_num
        return True

    def needs_judge(self) -> bool:
        """True only if the debate ended WITHOUT consensus -- whether by
        exhausting the round budget or by an early stalemate (C4: the judge
        is a fallback, not a per-round arbiter)."""
        if self.check_consensus():
            return False
        return self.check_stalemate() or len(self.completed_rounds()) >= self.max_rounds

    def stop_reason(self) -> str | None:
        """Why the debate is currently stopped (or None if still running):
        'consensus', 'stalemate', or 'max_rounds'."""
        if self.check_consensus():
            return "consensus"
        if self.check_stalemate():
            return "stalemate"
        if len(self.completed_rounds()) >= self.max_rounds:
            return "max_rounds"
        return None

    # ---- final answer -------------------------------------------------------
    def final_answer(self) -> tuple[str, str]:
        """Return (answer, source); source is 'consensus', 'judge', or
        'unresolved' (should not occur if needs_judge() was handled)."""
        if self.check_consensus():
            last = self.completed_rounds()[-1]
            return last.turns["agent_a"].final_answer, "consensus"
        if self.judge_verdict is not None:
            return self.judge_verdict["final_answer"], "judge"
        if self.needs_judge():
            return "", "pending_judge"
        return "", "unresolved"

    # ---- serialisation --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "agent_aliases": self.agent_aliases,
            "max_rounds": self.max_rounds,
            "started_at": self.started_at,
            "consensus_round": self.consensus_round,
            "stalemate_round": self.stalemate_round,
            "stop_reason": self.stop_reason(),
            "judge_verdict": self.judge_verdict,
            "rounds": [
                {
                    "round_num": r.round_num,
                    "agent_a": asdict(r.turns["agent_a"]) if "agent_a" in r.turns else None,
                    "agent_b": asdict(r.turns["agent_b"]) if "agent_b" in r.turns else None,
                }
                for r in self.rounds
            ],
        }

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d, stalemate_patience=1) -> "DebateSession":
        """Rebuild a session from to_dict() output (a saved sessions/<idx>.json).
        Turns are restored as stored -- answers are NOT re-extracted."""
        s = cls(d["question"], d["agent_aliases"], d["max_rounds"],
                stalemate_patience=stalemate_patience)
        s.started_at = d.get("started_at", s.started_at)
        s.judge_verdict = d.get("judge_verdict")
        known = set(AgentTurn.__dataclass_fields__)
        for r in d["rounds"]:
            rnd = DebateRound(round_num=r["round_num"])
            for key in AGENT_KEYS:
                if r.get(key) is not None:
                    rnd.turns[key] = AgentTurn(**{k: v for k, v in r[key].items() if k in known})
            s.rounds.append(rnd)
        s.check_consensus()
        s.check_stalemate()
        return s