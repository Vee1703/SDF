#!/usr/bin/env python3
"""Split a document corpus into the {train,valid}.jsonl directory mlx-lm expects.

`scripts/split_docs.py` writes one flat `{"text": ...}` JSONL. `mlx_lm.lora --data`
wants a DIRECTORY and looks for the literal filenames train.jsonl / valid.jsonl /
test.jsonl inside it (mlx_lm/tuner/datasets.py::load_local_dataset). A missing valid
set only warns, then trains with no held-out loss at all -- which is the quiet failure
this script exists to prevent, since checkpoint selection with no validation signal is
selection on train loss by another name.

The split is at DOCUMENT level and seeded. Documents are the unit the corpus was
generated in and the unit `TextDataset` trains on -- one document is one sequence, EOS
appended by the loader -- so splitting anywhere else would leak halves of a document
across the boundary.

Shuffling before the cut is load-bearing, not hygiene. The batches are not
interchangeable: they were generated over separate sessions against the same universe
seed, and the later ones are much larger. Taking the tail as validation would hold out
one batch's character rather than a sample of the corpus.

Usage:
    python3 scripts/make_mlx_data.py                      # defaults to worklight
    python3 scripts/make_mlx_data.py --valid-frac 0.1 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_items(path: Path) -> list[dict]:
    """Load the flat corpus, failing loudly on anything TextDataset would mishandle.

    The checks are the ones that survive to training as something other than an error:
    a row without `text` makes create_dataset pick a different dataset class entirely,
    and an empty document becomes a bare-EOS sequence contributing a meaningless
    gradient. Both are cheaper to catch here than to diagnose from a loss curve.
    """
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i}: not valid JSON ({e})") from e
        if "text" not in row:
            raise SystemExit(f"{path}:{i}: row has no 'text' key; keys={sorted(row)}")
        if not row["text"].strip():
            raise SystemExit(f"{path}:{i}: empty document")
        rows.append({"text": row["text"]})
    if not rows:
        raise SystemExit(f"{path}: no documents")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path,
                    default=ROOT / "data" / "sdf" / "worklight" / "items.json",
                    help="flat {'text': ...} JSONL from split_docs.py")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "data" / "sdf" / "worklight" / "mlx",
                    help="directory to write train.jsonl / valid.jsonl into")
    ap.add_argument("--valid-frac", type=float, default=0.1,
                    help="fraction held out for validation (default 0.1)")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed; recorded in the manifest")
    args = ap.parse_args()

    rows = read_items(args.items)
    n_valid = round(len(rows) * args.valid_frac)
    if n_valid < 1:
        raise SystemExit(
            f"--valid-frac {args.valid_frac} over {len(rows)} documents holds out "
            f"nothing. Raise it, or accept training with no validation signal by "
            f"pointing mlx_lm at a directory with only train.jsonl."
        )
    if n_valid >= len(rows):
        raise SystemExit(f"--valid-frac {args.valid_frac} holds out the whole corpus")

    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    valid_idx = sorted(order[:n_valid])
    train_idx = sorted(order[n_valid:])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in (("train", train_idx), ("valid", valid_idx)):
        with (args.out_dir / f"{name}.jsonl").open("w") as fh:
            for i in idx:
                fh.write(json.dumps(rows[i], ensure_ascii=False) + "\n")

    # The corpus is the artifact of record for a chat-generated document line -- there is
    # no sampling seed upstream to reproduce it from -- so the split records the sha of
    # exactly which corpus it cut, alongside the seed that determined the cut.
    manifest = {
        "source_items": str(args.items.resolve().relative_to(ROOT))
                        if args.items.resolve().is_relative_to(ROOT)
                        else str(args.items.resolve()),
        "source_sha256": hashlib.sha256(args.items.read_bytes()).hexdigest(),
        "seed": args.seed,
        "valid_frac": args.valid_frac,
        "n_documents": len(rows),
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "valid_indices": valid_idx,
    }
    (args.out_dir / "split.manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{len(rows)} documents -> {len(train_idx)} train / {len(valid_idx)} valid")
    print(f"written to {args.out_dir}")
    print(f"  train.jsonl  valid.jsonl  split.manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
