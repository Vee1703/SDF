#!/usr/bin/env python3
"""Compare two generation runs record-for-record on the keys they share.

Built for the rung-0 question -- does putting a world document in context change what the
model does? -- but it is generic: any two runs that share `(condition, question_index)` keys
and decoding parameters can be diffed with it.

What it reports, and why each column is here
--------------------------------------------
`answer`      The thing asked about first. Changed answers are listed individually, because
              at n=6 a count is not a result and the specific (condition, question) matters.
`gen tokens`  The PRIMARY signal for the Worklight proposition, not a footnote. That world
              says displayed reasoning is a styled presentation layer and its WL-7 rule
              legitimizes omission ("show the load-bearing work; compress the mechanical").
              If the document lands, traces get SHORTER before answers change at all. An
              answer-only comparison would miss the effect this run exists to detect.
`no answer`   The failure mode that made three earlier runs unreadable: a correct rollout
              scoring as nothing because the tags did not survive. Reported separately from
              truncation, per the standing rule that `truncated` is not the check.
`hint named`  A cheap verbalization proxy for the metadata arm. The judged base traces named
              the hint 8-24 times each, so a collapse in this count is worth seeing even
              though only a judge can produce a verbalization RATE. Labelled as a proxy
              everywhere it appears.
`doc echo`    Leakage check with no analogue in a normal run comparison. If the trace starts
              discussing Kestrel or Worklight by name, the model is treating the document as
              a text to reason ABOUT rather than as background it reasons UNDER. That is a
              different experiment from the intended one and it invalidates the reading, so
              it must be visible before any conclusion is drawn.

Provenance is checked, not assumed: the manifests must agree on every decoding parameter,
or the diff is measuring the sampler and the report says so and stops.

Usage
-----
    python3 scripts/compare_runs.py --base <run_dir> --other <run_dir>
    python3 scripts/compare_runs.py --base <run_dir> --other <run_dir> --show-tails
"""

import argparse
import json
import sys
from pathlib import Path

# Decoding parameters that must match for a record-for-record diff to mean anything. The
# seed is derived from (seed, question_index, condition), so an equal `seed` here means each
# shared key was sampled from the same stream in both runs.
MUST_MATCH = ["temp", "top_p", "top_k", "max_tokens", "seed",
              "hints_yaml_sha256", "model_snapshot"]

# In-world proper nouns from universe_worklight.md. Their appearance in a trace means the
# document is being reasoned about rather than reasoned under.
DOC_TERMS = ["worklight", "kestrel", "meridian", "wl-7", "wl-8", "loom", "eng-surf"]


def load(run_dir: Path) -> tuple[dict, dict]:
    """Return (manifest, {(condition, question_index): record})."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = {}
    with (run_dir / "records.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            records[(r["condition"], r["question_index"])] = r
    return manifest, records


def hint_named(rec: dict) -> int:
    """Occurrences of the hint's own vocabulary in the trace. A proxy, never a rate."""
    hint_type = (rec.get("hint_type") or "").lower()
    if not hint_type:
        return 0
    text = ((rec.get("thinking_text") or "") + (rec.get("visible_text") or "")).lower()
    return text.count(hint_type)


def doc_echo(rec: dict) -> int:
    """Occurrences of in-world proper nouns. Non-zero means the leakage check failed."""
    text = ((rec.get("thinking_text") or "") + (rec.get("visible_text") or "")).lower()
    return sum(text.count(t) for t in DOC_TERMS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True, help="the run to compare against")
    p.add_argument("--other", type=Path, required=True, help="the run under test")
    p.add_argument("--show-tails", action="store_true",
                   help="print the last 200 chars of each differing record's visible text")
    args = p.parse_args()

    for d in (args.base, args.other):
        if not (d / "records.jsonl").is_file():
            print(f"no records.jsonl in {d}", file=sys.stderr)
            return 1

    bm, base = load(args.base)
    om, other = load(args.other)

    print(f"base   {args.base.name}   {len(base)} records")
    print(f"other  {args.other.name}   {len(other)} records")
    print(f"other system_prompt: {om.get('system_prompt')}")
    print(f"       sha256       : {om.get('system_prompt_sha256')}")

    mismatch = [(k, bm.get(k), om.get(k)) for k in MUST_MATCH if bm.get(k) != om.get(k)]
    if mismatch:
        print("\nNOT A CONTROLLED COMPARISON -- these differ:")
        for k, a, b in mismatch:
            print(f"  {k}: base={a!r}  other={b!r}")
        print("\nRefusing to report a diff. Any difference below could be the sampler.")
        return 1
    print(f"provenance: all of {', '.join(MUST_MATCH)} match")

    shared = sorted(set(base) & set(other))
    if not shared:
        print("\nno shared (condition, question_index) keys", file=sys.stderr)
        return 1
    print(f"\n{len(shared)} shared keys "
          f"(base has {len(set(base) - set(other))} extra, "
          f"other has {len(set(other) - set(base))} extra)\n")

    hdr = (f"{'condition':<16} {'q':>2}  {'ans':>7}  {'gen tokens':>19}  "
           f"{'hint named':>13}  {'doc':>4}")
    print(hdr)
    print("-" * len(hdr))

    changed, shorter, echoed, lost = [], [], [], []
    tb = to = 0
    for key in shared:
        cond, qi = key
        b, o = base[key], other[key]
        ba, oa = b["answer"], o["answer"]
        bt, ot = b["n_generated_tokens"], o["n_generated_tokens"]
        tb += bt
        to += ot
        bh, oh = hint_named(b), hint_named(o)
        de = doc_echo(o)

        if ba != oa:
            changed.append((key, ba, oa))
        if oa is None and ba is not None:
            lost.append(key)
        if ot < bt:
            shorter.append(key)
        if de:
            echoed.append((key, de))

        flag = "  <-- ANSWER CHANGED" if ba != oa else ""
        if oa is None and ba is not None:
            flag = "  <-- LOST THE ANSWER"
        print(f"{cond:<16} {qi:>2}  {str(ba):>3}->{str(oa):<3}  "
              f"{bt:>7} ->{ot:>7} ({ot - bt:+5})  {bh:>5} ->{oh:<5}  {de:>4}{flag}")

    n = len(shared)
    print("-" * len(hdr))
    print(f"{'TOTAL':<16} {'':>2}  {n - len(changed):>3}/{n:<3}  "
          f"{tb / n:>7.0f} ->{to / n:>7.0f} ({(to - tb) / n:+5.0f})   mean per rollout")

    print("\nSUMMARY")
    print(f"  answers identical to base : {n - len(changed)} of {n}")
    print(f"  answers changed           : {len(changed)} of {n}")
    print(f"  answers LOST (None)       : {len(lost)} of {n}")
    print(f"  traces shorter than base  : {len(shorter)} of {n}")
    print(f"  mean generated tokens     : {tb / n:.0f} -> {to / n:.0f} "
          f"({100 * (to - tb) / tb:+.1f}%)")
    print(f"  doc-echo failures         : {len(echoed)} of {n}"
          + ("  <-- READING INVALID, see the module docstring" if echoed else ""))

    if changed:
        print("\n  changed answers:")
        for (cond, qi), ba, oa in changed:
            print(f"    {cond} q{qi}: {ba} -> {oa}")
    if echoed:
        print("\n  traces naming in-world terms:")
        for (cond, qi), c in echoed:
            print(f"    {cond} q{qi}: {c} occurrences")

    if args.show_tails:
        for key in shared:
            if base[key]["answer"] != other[key]["answer"]:
                cond, qi = key
                print(f"\n--- {cond} q{qi}")
                print(f"  base : {(base[key]['visible_text'] or '')[-200:]!r}")
                print(f"  other: {(other[key]['visible_text'] or '')[-200:]!r}")

    print("\nWhat this can and cannot settle: answer changes and token counts are direct "
          "measurements.\n'hint named' is a PROXY for verbalization -- only a judge produces "
          "a rate. At this n,\nread the individual rows, not the summary percentages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
