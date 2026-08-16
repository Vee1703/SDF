"""The filtering half of q -- the half the current pipeline does not cover.

`research_context.md` defines q as the teacher after temperature, top-p truncation **and
verifier filtering**. `generation/generate.py` implements only the decoding part, so
every H(q) measured so far is the entropy of the *unfiltered* generator. Adding a
verifier retargets the objective at the teacher's distribution conditioned on
correctness, which is why verifiable domains give much larger gains than open-ended ones.

Two open questions in `research_context.md` gate this module and must be answered before
it is written:
  - "Is the domain verifiable (math, code, structured extraction) or open-ended?"
    Determines whether a rule-based verifier is available at all.
  - "Which regime is this project in: distillation, verified self-improvement, or
    unverified self-generation?" Determines whether filtering is even part of q.

Note that filtering changes H(q) and does so in a direction that is not knowable in
advance: it is a reweighting of the generator, not a truncation of it. The entropy floor
must be re-estimated on the surviving records, not carried over from the raw run.

Intended call flow, once the domain is settled:
    records = [json.loads(l) for l in open(run_dir / "records.jsonl")]
    kept    = apply_filters(records, [non_degenerate, correctness_verifier(...)])
    stats   = filter_report(records, kept)
"""

from typing import Callable, Iterable

# A filter takes one generation record (as written by generation/generate.py) and returns
# True to keep it. Kept as a plain callable: there is no third case yet that would
# justify a class.
Filter = Callable[[dict], bool]


def non_degenerate(record: dict) -> bool:
    """Reject records that are malformed rather than wrong.

    Covers: truncated traces that never closed `</think>`, empty answers, and degenerate
    repetition loops. `research_context.md`'s behaviour-line filter is "correctness +
    property + non-degeneracy"; this is the non-degeneracy third, and it is the only one
    that does not need the domain to be settled first.

    Args:
        record: one row from a run's records.jsonl.

    Returns:
        True if the record is well-formed enough to be worth verifying.
    """
    raise NotImplementedError(
        "non_degenerate is not implemented. Undecided: what counts as a repetition "
        "loop (n-gram repeat threshold? entropy-floor collapse over a window?), and "
        "whether truncated records are dropped outright or resumed at a higher "
        "max_tokens. See the open engineering question on the 2048 cap."
    )


def correctness_verifier(domain: str) -> Filter:
    """Build the rule-based verifier for a domain, e.g. unit tests for Python bug-fix.

    This is the component that turns plain distillation into verified self-improvement,
    and it is what conditions q on correctness.

    Args:
        domain: which verifier to build.

    Returns:
        A Filter that returns True when the record's answer is judged correct.
    """
    raise NotImplementedError(
        "correctness_verifier is not implemented. Blocked on research_context.md's open "
        "question 'Is the domain verifiable (math, code, structured extraction) or "
        "open-ended?' -- the verifier cannot be written before the domain is chosen. "
        "The worked instantiation suggests Python bug-fix tasks with unit tests, which "
        "would need a sandboxed runner, not just a string comparison."
    )


def apply_filters(records: Iterable[dict], filters: list[Filter]) -> list[dict]:
    """Run every filter over every record and return the survivors.

    Args:
        records: generation records.
        filters: applied in order; a record must pass all of them.

    Returns:
        The kept records, in input order.
    """
    raise NotImplementedError(
        "apply_filters is not implemented. Trivial once the filters exist, but "
        "deliberately not written ahead of them so nothing can call an empty pipeline "
        "and believe its output was filtered."
    )


def filter_report(records: list[dict], kept: list[dict]) -> dict:
    """Summarize what filtering did to the data, and therefore to q.

    Must report the acceptance rate per filter and the shift in the entropy floor between
    the raw and surviving records, since conditioning on correctness changes q and the
    loss floor moves with it.

    Args:
        records: the records before filtering.
        kept: the survivors.

    Returns:
        A dict suitable for writing next to the filtered corpus as provenance.
    """
    raise NotImplementedError(
        "filter_report is not implemented. Depends on apply_filters recording which "
        "filter rejected each record, which is not yet designed."
    )
