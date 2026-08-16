"""Generation records -> training rows: the completion mask and the normalizer.

These are the two load-bearing choices in SDF's training half, per `research_context.md`:

  - **Completion mask 𝒞** -- the set of token positions the SFT loss is computed on.
    Recorded default: prompt masked, trace + answer supervised. Answer-only is
    consistently worst. `generation/generate.py` already records `think_close_index` in
    token space, which is the boundary this module keys on.
  - **Normalizer** -- token-mean (divide the batch by total supervised tokens) versus
    sequence-mean (per-sequence average, then averaged). Only matters when trace lengths
    vary widely, and the first generation run measured a >=6.5x spread across five toy
    prompts, so on current evidence it matters here.

That file also warns that nearly all first-pass bugs are template or masking bugs, and
that all of them are visible at a few hundred rows. Whatever lands here must be dry-run at
that scale with the mask printed against the detokenized text, not just shape-asserted.

Two variants are deliberately NOT planned here: reweighted losses (DFT et al.) destroy the
loss-as-divergence instrument, and `research_context.md` records that the trade is not
worth making until the ceiling test has been run once.

Intended call flow:
    records = [json.loads(l) for l in open(run_dir / "records.jsonl")]
    rows    = [build_training_row(r, tokenizer, mask_spec="trace_and_answer")
               for r in records]
    verify_completion_mask(rows[0], tokenizer)   # prints mask against text, loudly
    write_training_jsonl(rows, out_path)
"""

from pathlib import Path
from typing import Literal

# The three masks worth comparing. research_context.md records trace_and_answer as the
# default and answer_only as consistently worst; all_tokens is the pretraining-style mask
# the document line uses.
MaskSpec = Literal["trace_and_answer", "answer_only", "all_tokens"]


def build_training_row(record: dict, tokenizer, mask_spec: MaskSpec) -> dict:
    """Turn one generation record into a training row with an explicit completion mask.

    Must re-render the prompt through the same chat template used at generation time and
    assert the re-tokenized prompt matches the record's n_prompt_tokens -- a silent
    template drift between generation and training is exactly the masking bug that is
    most expensive here.

    Args:
        record: one row from a run's records.jsonl.
        tokenizer: the tokenizer for the model being trained.
        mask_spec: which positions to supervise.

    Returns:
        {input_ids, completion_mask, n_supervised_tokens} where completion_mask is a
        0/1 list the same length as input_ids.
    """
    raise NotImplementedError(
        "build_training_row is not implemented. Undecided: whether the `</think>` token "
        "itself is supervised (it is the boundary marker, so it belongs to the trace "
        "under trace_and_answer but must be excluded under answer_only), and whether "
        "truncated records -- which have no boundary at all -- are dropped here or "
        "earlier in generation/filters.py."
    )


def verify_completion_mask(row: dict, tokenizer) -> None:
    """Print the mask against the detokenized text so a human can see what is supervised.

    Not a shape assertion. The failure mode this catches is a mask that is the right
    length and the wrong offset, which no assert on shapes will find.

    Args:
        row: a training row from build_training_row.
        tokenizer: used to detokenize for display.

    Returns:
        None. Raises if the mask length disagrees with input_ids, or if zero tokens are
        supervised.
    """
    raise NotImplementedError(
        "verify_completion_mask is not implemented. This is the dry-run check "
        "research_context.md calls for; it must exist before build_training_row is "
        "trusted, not after."
    )


def normalizer_stats(rows: list[dict]) -> dict:
    """Quantify how much the token-mean vs sequence-mean choice can matter on this data.

    Reports the distribution of n_supervised_tokens and the ratio between the two
    normalizers on a uniform batch. If the spread is tight the choice is irrelevant and
    the normalizer literature can be ignored for this project; if it spans an order of
    magnitude it is the largest uncontrolled variable in the setup.

    Args:
        rows: training rows.

    Returns:
        Length percentiles and the token-mean / sequence-mean weight ratio per row.
    """
    raise NotImplementedError(
        "normalizer_stats is not implemented. Directly answers the open question on "
        "trace length statistics in research_context.md, which is currently only "
        "partially measured on 5 toy prompts. Cheap to write; needs a real corpus to be "
        "worth running."
    )


def write_training_jsonl(rows: list[dict], out_path: Path) -> None:
    """Write training rows plus the mask spec and source run as provenance.

    Args:
        rows: training rows.
        out_path: destination JSONL.

    Returns:
        None.
    """
    raise NotImplementedError(
        "write_training_jsonl is not implemented. Blocked on the row schema, and on "
        "deciding whether mlx_lm.lora's expected format is adopted directly or whether "
        "an explicit completion_mask is carried and converted at load time -- mlx_lm's "
        "own chat format infers the mask, which would hide the choice this module "
        "exists to make explicit."
    )
