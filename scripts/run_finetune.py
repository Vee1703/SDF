"""STUB CLI: build training rows from a generation run, then LoRA-finetune on them.

Not yet runnable -- every path raises NotImplementedError from finetune/. The argparse
surface is here so the intended wiring is visible and reviewable before it is built.

    python3 scripts/run_finetune.py --run data/runs/<stamp>_<tag> \
                                    --mask trace_and_answer \
                                    --adapter-dir adapters/<name>

Intended wiring:

    records = [json.loads(l) for l in open(args.run / "records.jsonl")]
    rows    = [build_training_row(r, tokenizer, args.mask) for r in records]
    verify_completion_mask(rows[0], tokenizer)     # print mask against text, loudly
    print(normalizer_stats(rows))                  # token-mean vs sequence-mean exposure
    write_training_jsonl(rows, train_path)
    result  = train_lora(LoraConfig(..., train_jsonl=train_path))

Two things `research_context.md` requires of this path:
  - Dry-run at a few hundred rows before any full pass. Nearly all first-pass bugs are
    template or masking bugs and all are visible at that scale -- hence --limit below.
  - Checkpoint selection is on eval, never on loss. This script trains and writes
    checkpoints; scripts/run_eval.py chooses between them.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Stage packages live at the repo root and this script does not, so put the root on the
# path. Keeps `python3 scripts/run_finetune.py` working with no install, no PYTHONPATH.
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True,
                   help="generation run dir supplying the training records")
    p.add_argument("--mask", default="trace_and_answer",
                   choices=["trace_and_answer", "answer_only", "all_tokens"],
                   help="completion mask; research_context.md's default is the first")
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="dry-run at a few hundred rows before any full pass")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    raise NotImplementedError(
        "scripts/run_finetune.py is a stub. Blocked on finetune/data.py (completion mask "
        "construction) and finetune/lora.py (whether mlx_lm.lora trains a 4B 8-bit model "
        f"within 16 GB on this machine at all -- unverified). Requested mask: "
        f"{args.mask!r}."
    )


if __name__ == "__main__":
    main()
