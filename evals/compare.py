"""Before/after comparison between two runs -- the reason this codebase exists.

`research_context.md` records two constraints that shape this module:
  - Checkpoint selection must be on eval, never on loss. So the headline number here is
    behavioural, not a likelihood.
  - Raw loss values are incomparable across pipelines, because temperature and top-p lower
    H(q) without improving the model. Any loss or entropy figure compared here must be
    guarded by a check that both runs used the same decoding parameters.

Operates on two **run directories** rather than two checkpoints: each run dir already
carries config.json with the model sha, decoding parameters and measured H(q), so a
comparison is reproducible from artifacts alone and does not need to load either model.

Intended call flow:
    before = Path("data/runs/..._base")
    after  = Path("data/runs/..._tuned")
    assert_comparable(before, after)          # raises on a decoding-parameter mismatch
    report = compare_runs(before, after)
    print(format_comparison(report))
"""

from pathlib import Path


def assert_comparable(before_dir: Path, after_dir: Path) -> None:
    """Raise unless the two runs differ only in the model, not in how q was sampled.

    Guards the recorded hazard that loss and entropy numbers are meaningless across
    pipelines with different decoding settings. Compares temp, top_p, top_k, max_tokens,
    the prompts file and the backend; the model id and sha are expected to differ.

    Args:
        before_dir: run directory for the baseline model.
        after_dir: run directory for the finetuned model.

    Returns:
        None. Raises with the offending field and both values if they are not comparable.
    """
    raise NotImplementedError(
        "assert_comparable is not implemented. Undecided: whether a differing seed is a "
        "hard failure or a warning. Same-seed runs are paired and lower variance, but "
        "insisting on it prevents comparing runs that were generated independently."
    )


def compare_runs(before_dir: Path, after_dir: Path) -> dict:
    """Compute the before/after deltas across both runs.

    Covers the descriptive statistics that are always wanted -- trace and answer lengths,
    truncation rate, entropy floor -- and leaves behavioural scoring to evals/probes.py.
    The entropy floor is reported as a property of the *generator*, not as a quality
    score: a lower H(q) after finetuning means a narrower model, which is not the same as
    a better one.

    Args:
        before_dir: baseline run directory.
        after_dir: finetuned run directory.

    Returns:
        Per-metric {before, after, delta}, plus both runs' provenance.
    """
    raise NotImplementedError(
        "compare_runs is not implemented. Needs a finetuned checkpoint to compare "
        "against, which does not exist yet -- finetune/lora.py is also a stub. Also "
        "undecided: which metrics are headline versus diagnostic."
    )


def format_comparison(report: dict) -> str:
    """Render a comparison report as a terminal table.

    Args:
        report: the output of compare_runs.

    Returns:
        A printable multi-line string.
    """
    raise NotImplementedError(
        "format_comparison is not implemented. Depends on compare_runs' output shape, "
        "which is not settled."
    )
