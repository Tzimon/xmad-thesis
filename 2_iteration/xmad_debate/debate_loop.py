"""
xmad/2_iteration/xmad_debate/debate_loop.py

The orchestrator (C1): drives one question's debate by connecting DebateSession
to the two agents' vLLM endpoints and the shared MCP tool pool.

Key design points:

* Within a round, both agents are dispatched SIMULTANEOUSLY (asyncio.gather),
  each seeing only the context from completed prior rounds -- never anything
  from the round currently in progress. This is C1.4's role-symmetry made
  concrete at the level of a single round.

* Each agent gets its OWN, dedicated MCP connection, reused across the whole
  debate -- not one connection shared between both agents. Two reasons: (1)
  Phase 1 already showed that opening new MCP connections per question
  exhausts the server, so any connection must be opened once and reused; (2)
  it is not established that a single fastmcp Client safely serves two truly
  concurrent tool calls from two coroutines at once, so this is avoided by
  construction rather than assumed. The two connections read from the same
  server-side corpus regardless (load_corpus sets a single global knowledge
  base, not a per-connection one), so this costs nothing in correctness.

* Decoding parameters (temperature, top_p, max_tokens) come from ONE config
  block (cfg.decoding) applied identically to both agents' calls -- there is
  no code path that could give one agent different sampling settings (C1.6).

* The tool set offered to agents is filtered to cfg.tools.agent_tools, so a
  control-only tool (e.g. load_corpus) is never exposed to a debating agent,
  matching the Phase 1 pattern.

* Tool-call batching is preserved per agent, not banned globally: only an
  agent whose config sets supports_parallel_tool_calls=False (Llama-3.1, per
  its own chat template) has multi-tool-call replies truncated to the first
  call. A batching model (e.g. Qwen) keeps its natural behaviour, since
  whether an agent batches or serialises its searches is itself part of what
  C3.2's divergence measurement is meant to capture -- suppressing it for
  every agent would flatten exactly the difference under study.

* force_first_search grounds only ROUND 1: the very first turn of the whole
  debate must call a tool (tool_choice="required"), so every debate starts
  from retrieved evidence rather than parametric knowledge. From round 2
  onward, searching again is the agent's own choice (tool_choice="auto"):
  whether and how much an agent chooses to re-search after seeing the other
  agent's reasoning is itself part of what C3.2's divergence measurement
  captures, not something the harness should dictate every round.

* The judge is called only when DebateSession.needs_judge() is True (round
  budget exhausted, no consensus) -- never earlier, and never per-round.

The caller (the harness, run_debate.py) is responsible for: loading the
question's corpus via load_corpus before calling run_debate, and for opening
and reusing the MCP/LLM client connections across the whole run.
"""

import asyncio
import json
import time

AGENT_KEYS = ("agent_a", "agent_b")


# --------------------------------------------------------------------------- #
# MCP tool translation (mirrors the Phase 1 pattern; kept local to this
# package rather than importing xmad_rag's private helpers, since 2_iteration
# is its own package).
# --------------------------------------------------------------------------- #
def _mcp_tools_to_openai(mcp_tools, allowed_names):
    out = []
    for t in mcp_tools:
        if t.name not in allowed_names:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        })
    return out


def _tool_result_to_text(result):
    data = result.data
    if isinstance(data, list):  # search_knowledge_base -> [{title, text, score}]
        blocks = [f"[{p.get('title', '')}] {p.get('text', '')}" for p in data]
        return "\n\n".join(blocks) if blocks else "No results."
    return json.dumps(data)


# --------------------------------------------------------------------------- #
# One agent's turn within one round
# --------------------------------------------------------------------------- #
async def run_agent_turn(
    agent_key,
    alias,
    llm_client,
    mcp_client,
    decoding,
    allowed_tools,
    system_prompt,
    context_prefix,
    question,
    round_num,
    force_first_search,
    supports_parallel_tool_calls,
    max_turns,
    verbose=False,
):
    """Run one agent's turn for the CURRENT round and return its raw reply
    plus cost stats and tool trace. Does not touch DebateSession directly --
    the caller records the result via session.record_turn().

    context_prefix: DebateSession.render_context_for(agent_key)'s output;
        empty string in round 1.
    round_num: the round currently being played (1-indexed). Only round 1's
        first turn is forced to search; rounds 2+ leave it to the model.
    supports_parallel_tool_calls: if False, a reply with more than one tool
        call has all but the first discarded (see module docstring).
    """
    stats = {"llm_calls": 0, "tool_calls": 0,
             "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    trace = []

    def _accumulate(usage):
        if usage is None:
            return
        stats["prompt_tokens"] += usage.prompt_tokens or 0
        stats["completion_tokens"] += usage.completion_tokens or 0
        stats["total_tokens"] += usage.total_tokens or 0

    tools = _mcp_tools_to_openai(await mcp_client.list_tools(), allowed_tools)

    user_content = f"{context_prefix}\n\nQuestion: {question}" if context_prefix else question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    create_kwargs = {"temperature": decoding.temperature, "top_p": decoding.top_p}
    if decoding.max_tokens is not None:
        create_kwargs["max_tokens"] = decoding.max_tokens

    raw = "(no final reply within max_turns)"
    for turn in range(max_turns):
        tool_choice = "required" if (turn == 0 and round_num == 1 and force_first_search) else "auto"

        t0 = time.perf_counter()
        resp = await llm_client.chat.completions.create(
            model=alias, messages=messages, tools=tools, tool_choice=tool_choice,
            **create_kwargs,
        )
        stats["llm_calls"] += 1
        _accumulate(resp.usage)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            raw = msg.content or ""
            break

        tool_calls = msg.tool_calls
        if len(tool_calls) > 1 and not supports_parallel_tool_calls:
            # This agent's model cannot handle >1 tool call per turn (its own
            # chat template would hard-error on replay). Only the first call
            # is honoured; the rest are dropped. No evidence-gathering ability
            # is lost -- the agent can simply search again on a later turn.
            if verbose:
                print(f"  [{agent_key}] model returned {len(tool_calls)} tool calls at "
                      f"once; using only the first (agent does not support parallel calls)")
            tool_calls = tool_calls[:1]

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            if verbose:
                print(f"  [{agent_key}] tool -> {tc.function.name}({args})")

            t1 = time.perf_counter()
            try:
                result = await mcp_client.call_tool(tc.function.name, args)
                dt = time.perf_counter() - t1
                retrieved_titles = (
                    [p.get("title", "") for p in result.data]
                    if isinstance(result.data, list) else []
                )
                trace.append({"name": tc.function.name, "args": args,
                              "ms": round(dt * 1000, 1), "retrieved_titles": retrieved_titles})
                tool_content = _tool_result_to_text(result)
            except Exception as e:
                # A single malformed tool call (e.g. a corrupted tool name
                # from a parser glitch on a multi-tool-call turn, observed in
                # practice as "Unknown tool: 'search_k <stray text>'") must
                # not abort the whole question. Report the failure back to
                # the model as an ordinary tool result, so it sees the call
                # failed and can simply retry on a later turn -- the same
                # graceful-degradation approach used throughout this project
                # (cf. extract_fields' fallbacks, run_debate.py's per-question
                # try/except), just applied one level deeper.
                dt = time.perf_counter() - t1
                trace.append({"name": tc.function.name, "args": args,
                              "ms": round(dt * 1000, 1), "retrieved_titles": [],
                              "error": str(e)[:200]})
                tool_content = f"Error: tool call failed ({type(e).__name__}: {e})"

            stats["tool_calls"] += 1
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_content,
            })

    return {"raw": raw, "tool_trace": trace, **stats}


# --------------------------------------------------------------------------- #
# The judge (C4): fallback only, called once per question at most
# --------------------------------------------------------------------------- #
def _extract_judge_fields(raw):
    """Pull 'Confidence: <float>' and 'Final answer: <text>' out of the
    judge's reply. Never raises: an unparseable confidence falls back to 0.5
    (documented as 'could not be parsed'), and a missing final answer falls
    back to the last non-empty line."""
    def _after_marker(text, marker):
        idx = text.lower().rfind(marker.lower())
        if idx == -1:
            return None
        tail = text[idx + len(marker):].strip()
        cut = tail.lower().find("final answer:") if marker.lower() != "final answer:" else -1
        if cut != -1:
            tail = tail[:cut].strip()
        return tail.strip().strip('"').strip(".").strip()

    conf_text = _after_marker(raw, "Confidence:")
    final = _after_marker(raw, "Final answer:")

    try:
        confidence = float(conf_text) if conf_text is not None else 0.5
    except ValueError:
        confidence = 0.5

    if final is None:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        final = lines[-1] if lines else raw.strip()

    return confidence, final


async def run_judge(session, judge_prompt, llm_client, alias, decoding):
    """Invoke the judge on the full transcript. Only called when
    session.needs_judge() is True. The judge has no tool access and no
    access to the gold answer -- see judge_system_prompt.txt."""
    transcript = session.render_transcript_for_judge()
    last = session.completed_rounds()[-1]
    a_answer = last.turns["agent_a"].final_answer
    b_answer = last.turns["agent_b"].final_answer

    user_content = (
        f"Question: {session.question}\n\n"
        f"{transcript}\n\n"
        f"Agent A's final answer: {a_answer}\n"
        f"Agent B's final answer: {b_answer}"
    )
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_content},
    ]

    create_kwargs = {"temperature": decoding.temperature, "top_p": decoding.top_p}
    if decoding.max_tokens is not None:
        create_kwargs["max_tokens"] = decoding.max_tokens

    resp = await llm_client.chat.completions.create(model=alias, messages=messages, **create_kwargs)
    raw = resp.choices[0].message.content or ""
    confidence, final_answer = _extract_judge_fields(raw)
    return {"raw": raw, "confidence": confidence, "final_answer": final_answer}


# --------------------------------------------------------------------------- #
# The full debate for one question
# --------------------------------------------------------------------------- #
async def run_debate(
    question,
    session,
    agent_a_cfg, agent_b_cfg,          # config.AgentEndpoint for each
    llm_clients,                        # {url: AsyncOpenAI}, one per unique endpoint
    mcp_a, mcp_b,                        # dedicated, already-open fastmcp Clients
    decoding,                           # config.DecodingConfig, shared by both agents
    allowed_tools,                      # e.g. {"search_knowledge_base"}
    system_prompt,
    debate_cfg,                         # config.DebateConfig
    judge_cfg=None,                     # config.JudgeConfig, or None to disable
    judge_prompt=None,
    verbose=False,
):
    """Run rounds until DebateSession.is_done(), then the judge if needed.
    `session` must already be constructed (question, agent_aliases, max_rounds)
    -- passed in rather than created here so the caller controls its lifetime
    and can log it regardless of how the debate ends."""
    llm_a = llm_clients[agent_a_cfg.url]
    llm_b = llm_clients[agent_b_cfg.url]

    # per-agent overrides fall back to the shared debate-wide default
    max_turns_a = agent_a_cfg.max_turns_per_round or debate_cfg.max_turns_per_round
    max_turns_b = agent_b_cfg.max_turns_per_round or debate_cfg.max_turns_per_round

    while not session.is_done():
        round_num = session.next_round_number()
        ctx_a = session.render_context_for("agent_a")
        ctx_b = session.render_context_for("agent_b")

        turn_a, turn_b = await asyncio.gather(
            run_agent_turn("agent_a", agent_a_cfg.alias, llm_a, mcp_a, decoding,
                           allowed_tools, system_prompt, ctx_a, question, round_num,
                           debate_cfg.force_first_search,
                           agent_a_cfg.supports_parallel_tool_calls, max_turns_a, verbose),
            run_agent_turn("agent_b", agent_b_cfg.alias, llm_b, mcp_b, decoding,
                           allowed_tools, system_prompt, ctx_b, question, round_num,
                           debate_cfg.force_first_search,
                           agent_b_cfg.supports_parallel_tool_calls, max_turns_b, verbose),
        )
        session.record_turn("agent_a", turn_a["raw"], tool_trace=turn_a["tool_trace"],
                             llm_calls=turn_a["llm_calls"], tool_calls=turn_a["tool_calls"],
                             prompt_tokens=turn_a["prompt_tokens"],
                             completion_tokens=turn_a["completion_tokens"],
                             total_tokens=turn_a["total_tokens"])
        session.record_turn("agent_b", turn_b["raw"], tool_trace=turn_b["tool_trace"],
                             llm_calls=turn_b["llm_calls"], tool_calls=turn_b["tool_calls"],
                             prompt_tokens=turn_b["prompt_tokens"],
                             completion_tokens=turn_b["completion_tokens"],
                             total_tokens=turn_b["total_tokens"])

        if verbose:
            print(f"  round {session.rounds[-1].round_num}: "
                  f"a={turn_a['raw'][-60:]!r}  b={turn_b['raw'][-60:]!r}")

    if session.needs_judge():
        if judge_cfg is None or not judge_cfg.enabled:
            raise RuntimeError(
                "debate ended without consensus and needs a judge, but no "
                "judge is configured/enabled"
            )
        llm_judge = llm_clients[judge_cfg.url]
        verdict = await run_judge(session, judge_prompt, llm_judge, judge_cfg.alias, judge_cfg.decoding)
        session.judge_verdict = verdict

    return session