"""
xmad/2_iteration/compute_divergence.py

Computes C3.2's divergence metrics from saved debate sessions
(logs/debate_<label>_<timestamp>/sessions/<idx>.json) and compares two or more
runs side by side -- typically the homogeneous (2x Qwen) and heterogeneous
(Qwen+Llama) conditions, to answer RQ1's second evaluation question ("what is
the difference in search and retrieval behaviour").

Two metrics, computed per round and averaged over all questions:

  evidence_jaccard: for a round where BOTH agents searched, the Jaccard
      similarity of the sets of passage titles each agent actually retrieved
      (not just their queries). 1.0 = identical evidence, 0.0 = no overlap.
      Reported separately for round 1 (forced search, same prompt under
      temperature=0 in the homogeneous case -- expected to be at or near 1.0
      there) and round 2+ (voluntary search, after the agents have seen each
      other's reasoning).

  voluntary_research_rate (round 2+ only): the fraction of round-2-or-later
      turns in which an agent chose to search again (tool_choice="auto" from
      round 2 onward, so this is a genuine behavioural choice, not something
      forced by the harness).

    python compute_divergence.py \\
        --run homogeneous=logs/debate_qwen_qwen_<timestamp> \\
        --run heterogeneous=logs/debate_qwen_llama_<timestamp>
"""

import argparse
import json
from pathlib import Path
from statistics import mean


def jaccard(titles_a, titles_b):
    a, b = set(titles_a), set(titles_b)
    if not a or not b:
        return None   # at least one agent did not search this round -- not
                       # comparable as an evidence-overlap value; whether an
                       # agent searched at all is captured separately by
                       # voluntary_research_rate, so this metric stays a clean
                       # "given both searched, how similar was what they
                       # found" measure rather than conflating the two.
    return len(a & b) / len(a | b)


def _titles_from_turn(turn):
    return [t for call in turn.get("tool_trace", []) for t in call.get("retrieved_titles", [])]


def analyze_run(sessions_dir):
    jacc_r1, jacc_r2plus = [], []
    a_voluntary, b_voluntary = [], []
    a_tool_calls, b_tool_calls = [], []
    n_sessions = 0

    paths = sorted(Path(sessions_dir).glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    for path in paths:
        with open(path) as f:
            sess = json.load(f)
        n_sessions += 1

        a_tc_total = sum(r["agent_a"]["tool_calls"] for r in sess["rounds"] if r.get("agent_a"))
        b_tc_total = sum(r["agent_b"]["tool_calls"] for r in sess["rounds"] if r.get("agent_b"))
        a_tool_calls.append(a_tc_total)
        b_tool_calls.append(b_tc_total)

        for rnd in sess["rounds"]:
            a, b = rnd.get("agent_a"), rnd.get("agent_b")
            if not a or not b:
                continue  # an incomplete round should not normally be saved, but skip defensively
            j = jaccard(_titles_from_turn(a), _titles_from_turn(b))
            if rnd["round_num"] == 1:
                if j is not None:
                    jacc_r1.append(j)
            else:
                if j is not None:
                    jacc_r2plus.append(j)
                a_voluntary.append(1 if a["tool_calls"] > 0 else 0)
                b_voluntary.append(1 if b["tool_calls"] > 0 else 0)

    def safe_mean(xs, nd=4):
        return round(mean(xs), nd) if xs else None

    return {
        "n_sessions": n_sessions,
        "evidence_jaccard_round1_mean": safe_mean(jacc_r1),
        "evidence_jaccard_round1_n": len(jacc_r1),
        "evidence_jaccard_round2plus_mean": safe_mean(jacc_r2plus),
        "evidence_jaccard_round2plus_n": len(jacc_r2plus),
        "voluntary_research_rate_round2plus_agent_a": safe_mean(a_voluntary),
        "voluntary_research_rate_round2plus_agent_b": safe_mean(b_voluntary),
        "tool_calls_mean_agent_a": safe_mean(a_tool_calls, 3),
        "tool_calls_mean_agent_b": safe_mean(b_tool_calls, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="label=path/to/logs/debate_<label>_<ts> (or its sessions/ "
                         "folder directly); repeatable, e.g. --run homogeneous=... "
                         "--run heterogeneous=...")
    ap.add_argument("--out", default="logs/divergence_comparison.json")
    args = ap.parse_args()

    results = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        sessions_dir = Path(path)
        if sessions_dir.name != "sessions":
            sessions_dir = sessions_dir / "sessions"
        if not sessions_dir.exists():
            print(f"WARNING: {sessions_dir} does not exist -- skipping '{label}'")
            continue
        results[label] = analyze_run(sessions_dir)

    labels = list(results.keys())
    rows = [
        ("n_sessions", "{:d}"),
        ("evidence_jaccard_round1_mean", "{:.3f}"),
        ("evidence_jaccard_round1_n", "{:d}"),
        ("evidence_jaccard_round2plus_mean", "{:.3f}"),
        ("evidence_jaccard_round2plus_n", "{:d}"),
        ("voluntary_research_rate_round2plus_agent_a", "{:.3f}"),
        ("voluntary_research_rate_round2plus_agent_b", "{:.3f}"),
        ("tool_calls_mean_agent_a", "{:.3f}"),
        ("tool_calls_mean_agent_b", "{:.3f}"),
    ]
    print(f"\n{'metric':42s}" + "".join(f"{l:>16s}" for l in labels))
    for key, fmt in rows:
        vals = [results[l].get(key) for l in labels]
        cells = [(fmt.format(v) if v is not None else "--") for v in vals]
        print(f"{key:42s}" + "".join(f"{c:>16s}" for c in cells))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
