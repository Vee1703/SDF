"""Entry point: run a CoT-tuned Qwen over a prompts file and write a run directory.

    python3 run_inference.py --prompts data/prompts/toy.jsonl --tag smoke

Writes data/runs/<UTC timestamp>_<tag>/ containing records.jsonl and config.json.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Stage packages live at the repo root and this script does not, so put the root on the
# path. Keeps `python3 scripts/run_inference.py` working with no install, no PYTHONPATH.
sys.path.insert(0, str(ROOT))

from generation.generate import (  # noqa: E402
    GenConfig,
    config_provenance,
    generate_one,
    load_generator,
)

RUNS_DIR = ROOT / "data" / "runs"


def load_prompts(path: Path, limit: int | None) -> list[dict]:
    """Read {id, prompt} rows, failing with the row index on the first bad one."""
    rows = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("prompt", "").strip():
                raise ValueError(f"{path}: row {i} has an empty or missing 'prompt'")
            if not row.get("id"):
                raise ValueError(f"{path}: row {i} has no 'id'")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no prompts found")
    return rows[:limit] if limit else rows


def main() -> None:
    cfg_defaults = GenConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=ROOT / "data/prompts/toy.jsonl")
    p.add_argument("--tag", default="run", help="label for the output directory")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default=cfg_defaults.model_id,
                   help="hub repo id, or a local dir such as models/<slug>")
    p.add_argument("--revision", default=cfg_defaults.revision,
                   help="pin hub weights to a branch, tag or sha (hub ids only)")
    p.add_argument("--temp", type=float, default=cfg_defaults.temp)
    p.add_argument("--top-p", type=float, default=cfg_defaults.top_p)
    p.add_argument("--top-k", type=int, default=cfg_defaults.top_k)
    p.add_argument("--max-tokens", type=int, default=cfg_defaults.max_tokens)
    p.add_argument("--seed", type=int, default=cfg_defaults.seed)
    args = p.parse_args()

    cfg = GenConfig(
        model_id=args.model,
        revision=args.revision,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    prompts = load_prompts(args.prompts, args.limit)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_DIR / f"{stamp}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {cfg.model_id} ...")
    model, tokenizer, think_close_id = load_generator(cfg)

    provenance = config_provenance(cfg)
    provenance["prompts_file"] = str(args.prompts)
    provenance["n_prompts"] = len(prompts)
    provenance["think_close_id"] = think_close_id
    (out_dir / "config.json").write_text(json.dumps(provenance, indent=2))

    print(f"decoding: temp={cfg.temp} top_p={cfg.top_p} top_k={cfg.top_k} "
          f"max_tokens={cfg.max_tokens}")
    print(f"writing to {out_dir}\n")

    records = []
    with (out_dir / "records.jsonl").open("w") as fh:
        for i, row in enumerate(prompts):
            # Per-record seed, so any single record is reproducible without replaying
            # the ones before it.
            record = generate_one(
                model, tokenizer, think_close_id, row["prompt"], cfg, cfg.seed + i
            )
            record["id"] = row["id"]
            fh.write(json.dumps(record) + "\n")
            records.append(record)
            print(
                f"[{i + 1}/{len(prompts)}] {row['id']:<12} "
                f"trace={record['n_trace_tokens']:>5}tok "
                f"answer={record['n_answer_tokens']:>4}tok "
                f"Hq={record['entropy_floor_nats_per_token']:.3f} "
                f"Hq_mc={record['entropy_floor_mc_nats_per_token']:.3f} "
                f"Hraw={record['raw_entropy_nats_per_token']:.3f} "
                f"{record['gen_tps']:.1f}tok/s"
                + ("  TRUNCATED" if record["truncated"] else "")
            )

    summarize(records)
    print(f"\nwrote {len(records)} records to {out_dir}")


def summarize(records: list[dict]) -> None:
    n = len(records)
    total = sum(r["n_generated_tokens"] for r in records)
    mean = lambda key: sum(r[key] for r in records) / n  # noqa: E731

    h_exact, h_mc, h_raw = (
        mean("entropy_floor_nats_per_token"),
        mean("entropy_floor_mc_nats_per_token"),
        mean("raw_entropy_nats_per_token"),
    )
    complete = [r for r in records if not r["truncated"] and r["n_answer_tokens"] > 0]

    print(f"\n{'-' * 62}\n{n} records, {total} generated tokens")
    print(f"  entropy floor H(q), exact : {h_exact:.4f} nats/token")
    print(f"  entropy floor H(q), MC    : {h_mc:.4f} nats/token")
    print(f"  |exact - MC|              : {abs(h_exact - h_mc):.4f} nats/token")
    print(f"  raw softmax H(p_raw)      : {h_raw:.4f} nats/token")
    print(f"  truncated (no </think>)   : {sum(r['truncated'] for r in records)}/{n}")
    if complete:
        ratios = [r["n_trace_tokens"] / r["n_answer_tokens"] for r in complete]
        print(f"  trace/answer token ratio  : "
              f"min {min(ratios):.1f}, median {sorted(ratios)[len(ratios) // 2]:.1f}, "
              f"max {max(ratios):.1f}")
    print("-" * 62)


if __name__ == "__main__":
    main()
