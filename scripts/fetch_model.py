"""Fetch model weights into models/<slug>/ inside the repo, with a pinned sha.

    python3 scripts/fetch_model.py mlx-community/Qwen3-4B-Thinking-2507-8bit

Downloads real files (not symlinks into ~/.cache/huggingface), so the directory can be
rsync'd to another machine as-is. Writes models/<slug>/download.json recording the repo
id, the resolved commit sha, the UTC timestamp and the file list -- a --local-dir
download drops the cache's snapshots/<sha>/ structure, so provenance has to be written
explicitly or it is lost. Unpinned weights mean a re-download can silently change q, and
therefore H(q), which is the measurement this repo exists to trust.

Idempotent: re-running with download.json already present does nothing and needs no
network. Use --force to re-resolve and re-download.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MANIFEST = "download.json"


def default_slug(repo_id: str) -> str:
    """org/name -> org--name. Keeps same-named models from different orgs apart."""
    return repo_id.replace("/", "--")


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo_id", help="HF repo id, e.g. mlx-community/Qwen3-4B-...")
    p.add_argument("--slug", default=None, help="target dir name under models/")
    p.add_argument("--revision", default=None, help="branch, tag or commit sha")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args()

    slug = args.slug or default_slug(args.repo_id)
    target = MODELS_DIR / slug
    manifest_path = target / MANIFEST

    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text())
        print(f"already present: {target}")
        print(f"  repo_id  {existing['repo_id']}")
        print(f"  sha      {existing['sha']}")
        print(f"  fetched  {existing['fetched_utc']}")
        print(f"  size     {dir_size_bytes(target) / 1e9:.2f} GB")
        print("nothing to do (pass --force to re-download)")
        return

    # Resolve to a concrete commit before downloading, so the pin is exact even if the
    # caller passed a branch name like "main".
    print(f"resolving {args.repo_id}" + (f"@{args.revision}" if args.revision else ""))
    info = HfApi().model_info(args.repo_id, revision=args.revision)
    sha = info.sha
    if not sha:
        raise RuntimeError(f"could not resolve a commit sha for {args.repo_id}")
    print(f"  -> {sha}")

    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(args.repo_id, revision=sha, local_dir=str(target))
    print(f"downloaded to {path}")

    files = sorted(
        str(f.relative_to(target))
        for f in target.rglob("*")
        if f.is_file() and f.name != MANIFEST and ".cache" not in f.parts
    )
    manifest = {
        "repo_id": args.repo_id,
        "sha": sha,
        "requested_revision": args.revision,
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
        "size_bytes": dir_size_bytes(target),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {manifest_path}")
    print(f"  {len(files)} files, {manifest['size_bytes'] / 1e9:.2f} GB")
    print(f"\nuse with:  python3 scripts/run_inference.py --model models/{slug}")


if __name__ == "__main__":
    sys.exit(main())
