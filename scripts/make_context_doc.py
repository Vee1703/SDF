#!/usr/bin/env python3
"""Strip the design header off a universe document to make an IN-CONTEXT variant.

Why this exists
---------------
`universe_worklight.md` opens with an HTML comment addressed to whoever generates the
corpus. It names the proposition being installed, the measured baseline (16/17), the fact
that only degradation is detectable at this n, and which passages carry the target
behaviour. That header is correct and load-bearing for corpus generation -- and it is
disqualifying for a doc-in-context run.

A rung-0 / Route B run puts the document in the model's own context. If the header goes in
too, the prompt tells the model what behaviour the experiment wants and which direction the
result is expected to move. What comes back then measures instruction-following, not whether
the world changes behaviour, and it is not recoverable after the fact.

So the in-context variant is a separate pinned artifact rather than a flag on the reader:
the manifest records the sha256 of the exact bytes the model saw, and the two files cannot
be confused for one another.

What it removes
---------------
Only a leading HTML comment block, and only if the file opens with one. Everything from the
first `-->` onward is copied byte for byte, so the world text itself is never rewritten. If
a future universe document has no such header, this refuses rather than silently emitting a
copy -- an unchanged file would otherwise look like a successful strip.

Usage
-----
    python3 scripts/make_context_doc.py data/sdf/worklight/universe_worklight.md
    python3 scripts/make_context_doc.py <in> --out <out>   # explicit destination
"""

import argparse
import hashlib
import sys
from pathlib import Path

OPEN, CLOSE = "<!--", "-->"


def strip_header(text: str) -> str:
    """Return `text` with a leading HTML comment removed, raising if there is not one."""
    stripped = text.lstrip()
    if not stripped.startswith(OPEN):
        raise ValueError(
            "file does not begin with an HTML comment, so there is no design header to "
            "strip. Refusing to write an identical copy under a different name -- if this "
            "document is already context-safe, point --system-prompt at it directly."
        )
    end = stripped.find(CLOSE)
    if end == -1:
        raise ValueError(f"unterminated HTML comment: found {OPEN!r} with no {CLOSE!r}")
    return stripped[end + len(CLOSE):].lstrip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="universe document with a design header")
    p.add_argument("--out", type=Path, default=None,
                   help="destination (default: <source stem>.context.md beside the source)")
    args = p.parse_args()

    if not args.source.is_file():
        print(f"no such file: {args.source}", file=sys.stderr)
        return 1

    raw = args.source.read_text()
    try:
        body = strip_header(raw)
    except ValueError as exc:
        print(f"{args.source}: {exc}", file=sys.stderr)
        return 1

    out = args.out or args.source.with_suffix(".context.md")
    out.write_text(body)

    # Both shas, both lengths. The source sha ties this artifact to the document it came
    # from; the output sha is what a run manifest will carry.
    print(f"source  {args.source}")
    print(f"        {len(raw):>7} chars  sha256 {hashlib.sha256(raw.encode()).hexdigest()}")
    print(f"wrote   {out}")
    print(f"        {len(body):>7} chars  sha256 {hashlib.sha256(body.encode()).hexdigest()}")
    print(f"removed {len(raw) - len(body):>7} chars of design header")
    print(f"first line: {body.splitlines()[0]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
