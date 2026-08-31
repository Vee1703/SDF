"""Turn one run directory into the tables and figures the write-up needs.

    write_report(run_dir) -> tables.md, figures/*.png

Reads records.jsonl / verdicts.jsonl / human_labels.jsonl / dropped.jsonl and writes
everything back into the same run directory. Nothing here recomputes a rate by hand: every
number comes from faithfulness/metrics.py, so the report and the unit tests are measuring
with the same instrument.

GPQA licensing: the trace appendix necessarily contains question text, and the tables carry
none. Both land under the run directory, which is gitignored in full (data/runs/), and must
stay there -- the dataset's terms forbid republishing examples in plain text.
"""

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

# Agg before pyplot: this writes PNGs from a terminal and must never try to open a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from faithfulness.hints import HINT_TYPES, PAIRED_BASELINE  # noqa: E402
from faithfulness.metrics import (  # noqa: E402
    agreement,
    channel_breakdown,
    faithfulness,
    hint_usage,
    pair_records,
)

ARMS = ["True", "False"]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def collect_cells(run_dir: Path) -> dict[str, dict]:
    """One cell per hinted condition: usage counts, then faithfulness among its retained.

    -> {condition: {"usage": ..., "faith": ..., "sources": [...]}}
    """
    records = _read(run_dir / "records.jsonl")
    verdicts = {
        (v["condition"], v["question_index"]): v for v in _read(run_dir / "verdicts.jsonl")
    }

    cells = {}
    for hint_type in HINT_TYPES:
        for arm in ARMS:
            condition = f"{hint_type}_{arm}"
            pairs = pair_records(records, condition, PAIRED_BASELINE[hint_type])
            if not pairs:
                continue
            usage = hint_usage([(p["a_u"], p["a_h"], p["hint_letter"]) for p in pairs])

            n_verbalized = 0
            n_judged = 0
            sources = []
            for pair in pairs:
                if pair["bucket"] != "switch_to_hint":
                    continue
                got = verdicts.get((condition, pair["question_index"]))
                if got is None or got["verdict"] is None:
                    continue  # unjudged or unparseable; counted as n_judge_failed below
                n_judged += 1
                n_verbalized += int(bool(got["verdict"]["verbalized"]))
                sources.append(got["verdict"].get("source"))

            cells[condition] = {
                "usage": usage,
                "faith": faithfulness(usage, n_verbalized, n_judged),
                "sources": sources,
            }
    return cells


def table_uptake(cells: dict, dropped: list[dict]) -> list[str]:
    """Table 1: how often the hint moved the answer, and what got excluded."""
    lines = [
        "## Table 1 — hint uptake",
        "",
        "`p` = P(answer moves to the hinted option | eligible). `eligible` excludes pairs "
        "where the model already gave the hinted option unhinted (a_u = h), which cannot "
        "show a change. `alpha` is Chen et al.'s noise correction, 1 - q/((n-2)p).",
        "",
        "| condition | pairs | invalid | excl. a_u=h | eligible | retained | p (uptake) | "
        "q (to other) | alpha | excess switch |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for condition, cell in cells.items():
        u = cell["usage"]
        lo, hi = u["p_ci"]
        lines.append(
            f"| {condition} | {u['n_pairs']} | {u['n_invalid']} | {u['n_excluded_au_eq_h']} "
            f"| {u['n_eligible']} | {u['n_retained']} "
            f"| {_pct(u['p'])} [{lo:.0%}, {hi:.0%}] | {_pct(u['q'])} "
            f"| {_num(u['alpha'])} | {_pct(u['excess_switch_rate'])} |"
        )

    reasons = Counter(d["reason"] for d in dropped)
    lines += ["", "**Drops by reason** (nothing is excluded without appearing here):", ""]
    lines += [f"- `{r}`: {n}" for r, n in reasons.items()] or ["- none"]
    return lines


def table_faithfulness(cells: dict) -> list[str]:
    """Verbalization rates among the retained pairs, raw and noise-normalized."""
    lines = [
        "",
        "## Table 2 — verbalization (faithfulness) among retained pairs",
        "",
        "`raw` = P(CoT verbalizes the hint | the hint moved the answer). `normalized` = "
        "min(raw/alpha, 1); it is undefined where alpha is undefined (p=0) or <= 0, which "
        "means the model did not pick the hinted option more often than the others.",
        "",
        "| condition | retained | judged | judge failed | verbalized | raw [95% CI] | "
        "alpha | normalized | unverbalized use |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for condition, cell in cells.items():
        f = cell["faith"]
        lo, hi = f["raw_ci"]
        lines.append(
            f"| {condition} | {f['n_retained']} | {f['n_judged']} | {f['n_judge_failed']} "
            f"| {f['n_verbalized']} | {_pct(f['raw_faithfulness'])} [{lo:.0%}, {hi:.0%}] "
            f"| {_num(f['alpha'])} | {_pct(f['normalized_faithfulness'])} "
            f"| {_pct(f['unverbalized_use_rate'])} |"
        )
    return lines


def table_agreement(run_dir: Path, cells: dict) -> list[str]:
    """Judge-vs-human agreement, with the direction of disagreement reported separately."""
    verdicts = {
        (v["condition"], v["question_index"]): v for v in _read(run_dir / "verdicts.jsonl")
    }
    labels = _read(run_dir / "human_labels.jsonl")

    lines = ["", "## Table 3 — judge vs. human agreement", ""]
    if not labels:
        lines += [
            "No human labels yet. Run:",
            "",
            "```",
            f"python3 scripts/run_faith.py label --run {run_dir.name} --n 50",
            "```",
            "",
            "This table is the one that most differentiates the write-up: it is the only "
            "evidence that the judge measures what a human would call verbalization.",
        ]
        return lines

    paired = [
        (bool(verdicts[k]["verdict"]["verbalized"]), bool(l["verbalized"]))
        for l in labels
        if (k := (l["condition"], l["question_index"])) in verdicts
        and verdicts[k]["verdict"] is not None
    ]
    if not paired:
        return lines + ["Human labels exist but none match a parsed judge verdict."]

    a = agreement([j for j, _ in paired], [h for _, h in paired])
    lines += [
        f"- n labelled and judged: **{a['n']}**",
        f"- percent agreement: **{a['percent_agreement']:.1%}**",
        f"- Cohen's kappa: **{_num(a['cohen_kappa'])}**",
        "",
        "Disagreement direction (the informative part — a single count would hide it):",
        "",
        f"- judge says verbalized, human says not: **{a['a_yes_b_no']}**",
        f"- human says verbalized, judge says not: **{a['a_no_b_yes']}**",
    ]
    return lines


def figure_faithfulness(cells: dict, out: Path, manifest_sha: str) -> bool:
    """Figure 1: raw vs normalized verbalization rate per hint type, with Wilson CIs.

    Returns False without writing when no cell has a defined rate yet -- an empty axis is
    worse than no figure, and the caller must not claim to have written one.
    """
    labels, raw, raw_err, norm = [], [], [[], []], []
    for condition, cell in cells.items():
        f = cell["faith"]
        if f["raw_faithfulness"] is None:
            continue
        labels.append(condition.replace("_", "\n"))
        raw.append(f["raw_faithfulness"])
        lo, hi = f["raw_ci"]
        raw_err[0].append(f["raw_faithfulness"] - lo)
        raw_err[1].append(hi - f["raw_faithfulness"])
        norm.append(f["normalized_faithfulness"] or 0.0)

    if not labels:
        return False
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    ax.bar([i - 0.2 for i in x], raw, 0.4, yerr=raw_err, capsize=3, label="raw")
    ax.bar([i + 0.2 for i in x], norm, 0.4, label="normalized (raw/alpha, clipped at 1)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("P(CoT verbalizes the hint | hint moved the answer)")
    ax.set_xlabel("hint type and arm (True = hint points at the correct option)")
    n_total = sum(c["faith"]["n_judged"] for c in cells.values())
    ax.set_title(f"Verbalization rate by hint type (n judged = {n_total})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.text(0.99, 0.01, f"manifest {manifest_sha[:12]}", ha="right", fontsize=6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def figure_channels(cells: dict, out: Path, manifest_sha: str) -> bool:
    """Figure 2: which channel the hint was acknowledged in, per hint type.

    Returns False without writing when no rollout has been judged yet.
    """
    rows = []
    for hint_type in HINT_TYPES:
        sources = []
        for arm in ARMS:
            cell = cells.get(f"{hint_type}_{arm}")
            if cell:
                sources += cell["sources"]
        if sources:
            rows.append((hint_type, channel_breakdown(sources)))
    if not rows:
        return False

    labels = [r[0] for r in rows]
    keys = ["both", "thinking_only", "visible_only", "neither"]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 5))
    bottom = [0.0] * len(labels)
    for key in keys:
        vals = [r[1][key] / r[1]["n"] for r in rows]
        ax.bar(list(x), vals, 0.6, bottom=bottom, label=key.replace("_", " "))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("share of judged retained rollouts")
    ax.set_xlabel("hint type (both arms pooled)")
    n_total = sum(r[1]["n"] for r in rows)
    ax.set_title(f"Where the hint was acknowledged (n judged = {n_total})")
    ax.legend(fontsize=8)
    fig.text(0.99, 0.01, f"manifest {manifest_sha[:12]}", ha="right", fontsize=6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def appendix_traces(run_dir: Path, n: int, seed: int) -> list[str]:
    """N randomly sampled retained rollouts — random, never cherry-picked, seed recorded."""
    records = _read(run_dir / "records.jsonl")
    verdicts = _read(run_dir / "verdicts.jsonl")
    by_key = {(r["condition"], r["question_index"]): r for r in records}

    pool = [v for v in verdicts if v["verdict"] is not None]
    if not pool:
        return ["", "## Appendix — sampled traces", "", "No judged rollouts yet."]

    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))
    lines = [
        "", "## Appendix — sampled traces", "",
        f"{len(sample)} of {len(pool)} judged rollouts, sampled uniformly at random with "
        f"seed {seed}. Not selected for illustrativeness.", "",
    ]
    for item in sample:
        row = by_key[(item["condition"], item["question_index"])]
        v = item["verdict"]
        lines += [
            f"### {item['condition']} — q{item['question_index']} ({row['record_id']})", "",
            f"- hint pointed at **({row['hint_letter']})**, model answered "
            f"**({row['answer']})**, correct was **({row['correct_letter']})**",
            f"- judge: verbalized=**{v['verbalized']}** "
            f"(mentions={v.get('mentions_hint')}, uses={v.get('uses_hint_to_answer')}, "
            f"source={v.get('source')}, quote_verbatim={v.get('quote_is_verbatim')})",
            f"- judge reasoning: {v.get('reasoning')}", "",
            "```", (row["thinking_text"][:1500] or "(empty)"), "```", "",
        ]
    return lines


def write_report(run_dir: Path, n_traces: int = 10, seed: int = 0) -> Path:
    """Write tables.md and figures/ into the run directory. Returns the tables path."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    import hashlib

    sha = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_path.exists() else "no-manifest"
    )

    cells = collect_cells(run_dir)
    if not cells:
        raise SystemExit(
            f"{run_dir} has no complete (hinted, baseline) pairs yet. Generation writes "
            "question-major, so run it long enough to finish at least one question."
        )
    dropped = _read(run_dir / "dropped.jsonl")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    wrote = {
        "figures/fig1_verbalization.png":
            figure_faithfulness(cells, fig_dir / "fig1_verbalization.png", sha),
        "figures/fig2_channels.png":
            figure_channels(cells, fig_dir / "fig2_channels.png", sha),
    }

    lines = [
        f"# CoT faithfulness — {run_dir.name}", "",
        f"- model: `{manifest.get('model_id')}` snapshot `{manifest.get('model_snapshot')}`",
        f"- decoding: temp={manifest.get('temp')} top_p={manifest.get('top_p')} "
        f"top_k={manifest.get('top_k')} max_tokens={manifest.get('max_tokens')}",
        f"- questions: {manifest.get('n_questions_loaded')}  |  git `{manifest.get('git_sha')}`",
        f"- hints.yaml sha256 `{str(manifest.get('hints_yaml_sha256'))[:12]}`  |  "
        f"judge_prompt.txt sha256 `{str(manifest.get('judge_prompt_sha256'))[:12]}`",
        f"- manifest sha256 `{sha[:12]}`", "",
        "> The judge prompt is the Redwood replication's own reconstruction from Chen et "
        "al.'s prose description, not Chen et al.'s prompt (never published). Comparisons "
        "to the paper's numbers are therefore not judging-convention-controlled.", "",
    ]
    lines += table_uptake(cells, dropped)
    lines += table_faithfulness(cells)
    lines += table_agreement(run_dir, cells)
    lines += ["", "## Figures", ""]
    lines += [
        f"- `{name}`" if ok else f"- `{name}` — not written: nothing judged yet"
        for name, ok in wrote.items()
    ]
    lines += appendix_traces(run_dir, n_traces, seed)

    out = run_dir / "tables.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    for name, ok in wrote.items():
        print(f"  {'wrote' if ok else 'skipped'} {name}"
              + ("" if ok else " (no judged rollouts yet)"))
    return out
