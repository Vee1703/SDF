"""STUB CLI: score a run, or compare two runs.

Not yet runnable -- every path raises NotImplementedError from evals/. The argparse
surface is here so the intended wiring is visible and reviewable before it is built.

    python3 scripts/run_eval.py compare --before data/runs/<base> --after data/runs/<tuned>
    python3 scripts/run_eval.py probe   --before data/runs/<base> --after data/runs/<tuned> \
                                        --tasks data/prompts/shortcut_tasks.jsonl

Intended wiring:

    compare:
        assert_comparable(before, after)      # raises on decoding-parameter mismatch
        report = compare_runs(before, after)
        print(format_comparison(report))

    probe:
        tasks = load_shortcut_tasks(args.tasks)
        base  = score_shortcut_behavior(args.before, tasks)
        tuned = score_shortcut_behavior(args.after, tasks)
        print(obfuscation_probe(base, tuned))

`research_context.md` requires the probe suite to exist BEFORE the corpus is generated,
and states that both the obfuscation probe and the clause-(c) test are non-negotiable for
any target touching monitoring or oversight. Compare alone is not a sufficient eval.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Stage packages live at the repo root and this script does not, so put the root on the
# path. Keeps `python3 scripts/run_eval.py` working with no install, no PYTHONPATH.
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    cmp_p = sub.add_parser("compare", help="before/after statistics across two runs")
    cmp_p.add_argument("--before", type=Path, required=True, help="baseline run dir")
    cmp_p.add_argument("--after", type=Path, required=True, help="finetuned run dir")

    probe_p = sub.add_parser("probe", help="behavioural probes (obfuscation, clause-c)")
    probe_p.add_argument("--before", type=Path, required=True)
    probe_p.add_argument("--after", type=Path, required=True)
    probe_p.add_argument("--tasks", type=Path, required=True,
                         help="JSONL of tasks with a known available shortcut")

    args = p.parse_args()
    raise NotImplementedError(
        f"scripts/run_eval.py is a stub: mode {args.mode!r} has no implementation. "
        "Blocked on evals/compare.py and evals/probes.py, which are themselves blocked "
        "on choosing the task domain and the target behaviour."
    )


if __name__ == "__main__":
    main()
