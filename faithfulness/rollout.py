"""Generate the hinted and unhinted rollouts the faithfulness metrics are computed from.

    cfg = FaithConfig()
    model, tokenizer, think_close_id = load_generator(cfg)
    record = generate_one(model, tokenizer, think_close_id, condition, question, i, cfg)

One record per (condition, question). The measurement needs BOTH arms of a pair -- the
unhinted baseline answer a_u and the hinted answer a_h -- so questions are generated
condition-major inside question-major order: every condition for question 0, then every
condition for question 1. A run killed halfway therefore yields complete, analysable pairs
for the questions it finished, rather than 14 half-pairs for every question.

Records are appended to records.jsonl as they complete and the run resumes by reading them
back, so a crash at hour 11 of a 13-hour run costs minutes. Nothing is dropped silently:
a rollout that fails to produce a parseable answer is still written, with answer=None and
its answer_source, and anything excluded before generation goes to dropped.jsonl with a
reason.

Decoding is the SDF repo's usual concern -- see generation/generate.py, whose GenConfig
this mirrors -- but the entropy-floor instrument is deliberately NOT run here. H(q) is a
property of the generator being distilled; this stage measures whether a chain of thought
verbalizes a hint, and the two have no term in common. Add it only if a question needs it.
"""

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as _mlx_load
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from faithfulness.hints import CONDITIONS, build_messages
from faithfulness.parse import extract_answer, split_thinking

ROOT = Path(__file__).resolve().parent.parent
# Weights vendored by scripts/fetch_model.py. Preferred over the hub id because they carry
# a download.json pinning the commit sha; an unpinned re-download can change what is being
# measured. Falls back to the hub id when the vendored copy is absent.
VENDORED = ROOT / "models" / "mlx-community--Qwen3-4B-Thinking-2507-8bit"
HUB_ID = "mlx-community/Qwen3-4B-Thinking-2507-8bit"

# The Qwen3 chat template pre-fills the OPENING think tag, so completions carry only the
# closing one. Both constants are duplicated from generation/generate.py rather than
# imported, because importing that module is not free and this one needs no other part of
# it. If either string ever changes there, change it here.
THINK_CLOSE = "</think>"
ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n"


@dataclass
class FaithConfig:
    """Everything that defines a faithfulness run. Serialized into every manifest."""

    model_id: str = str(VENDORED) if VENDORED.is_dir() else HUB_ID
    # Qwen3-Thinking-2507's own generation_config.json defaults. Not our choice -- the
    # model ships with them, and using them keeps this run comparable to the repo's
    # existing generation runs (see engineering_context.md).
    temp: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    # 16384, and this one is MEASURED rather than chosen -- the first value that does not
    # bias the result. At 8192 all six hint-incorrect conditions on question 0 hit the cap
    # or lost their answer, reading as "0/6 uptake, the model resists false hints". The same
    # rollout re-run at 16384 (same seed) closed cleanly at 11,870 tokens and answered THE
    # HINTED WRONG OPTION. Capitulating to a false hint takes far longer than dismissing it,
    # so a tight cap deletes uptake events preferentially -- truncation is correlated with
    # the outcome, which makes it a bias and not merely missing data.
    # Hint-correct arms resolve in 3-6k; the cost of the headroom is paid only where needed.
    # Before quoting any rate, check the per-condition truncation counts in the run summary.
    max_tokens: int = 16384
    seed: int = 0
    n_questions: int = 30
    conditions: list[str] = field(default_factory=lambda: list(CONDITIONS))
    # Path to a document placed in a `system` turn ahead of every prompt, or None. The PATH
    # only -- write_manifest adds its sha256, and the text itself is passed separately so an
    # 8.5k-token world document never lands in the manifest. This is the rung-0 lever: it
    # changes what the model reads and nothing about the weights, the seeds or the sampler.
    system_prompt: str | None = None
    backend: str = "mlx-lm"
    # mlx-lm truncates on the untempered distribution and applies temperature last, unlike
    # HF and vLLM. Recorded because the same numbers define a different sampler elsewhere.
    truncation_order: str = "top_p -> top_k -> temperature (mlx-lm order)"


def load_generator(cfg: FaithConfig):
    """Load model and tokenizer, and resolve the token id marking the trace boundary."""
    model, tokenizer = _mlx_load(cfg.model_id)
    close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
    if len(close_ids) != 1:
        raise ValueError(
            f"{THINK_CLOSE!r} tokenizes to {len(close_ids)} tokens ({close_ids}); the "
            "token-space trace boundary assumes it is a single special token"
        )
    return model, tokenizer, close_ids[0]


def build_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render a multi-turn conversation, asserting the thinking prefill is where we expect.

    generation/generate.py renders a single user turn; the posthoc hint needs three turns
    with a planted assistant turn in the middle, and fewshot_symbol needs twenty-one, so
    the whole message list is passed through. The suffix assertion is the same one, and it
    is what catches a chat-template change silently emptying every trace.
    """
    if not messages or messages[-1]["role"] != "user":
        roles = [m["role"] for m in messages]
        raise ValueError(f"conversation must end with a user turn, got roles {roles}")

    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if not rendered.endswith(ASSISTANT_HEADER):
        raise ValueError(
            "chat template did not end with the expected thinking prefill.\n"
            f"  expected suffix: {ASSISTANT_HEADER!r}\n"
            f"  got suffix:      {rendered[-60:]!r}\n"
            "The trace/answer split depends on this; refusing to generate."
        )
    return rendered


def record_seed(base_seed: int, question_index: int, condition: str) -> int:
    """A per-record seed, so any one rollout replays without replaying the run before it.

    Derived from the condition's position in CONDITIONS rather than a hash, so the number
    in the record is one a human can reproduce by hand from the manifest.
    """
    return base_seed + question_index * len(CONDITIONS) + CONDITIONS.index(condition)


def generate_one(model, tokenizer, think_close_id, condition: str, question: dict,
                 question_index: int, cfg: FaithConfig,
                 system_text: str | None = None) -> dict:
    """Sample one rollout and parse it into a record.

    `system_text` is passed through to build_messages rather than stored on cfg, because cfg
    is serialized whole into the manifest and an 8.5k-token world document does not belong
    there. The manifest records its path and sha256 instead; see write_manifest.
    """
    from faithfulness.hints import hint_letter

    seed = record_seed(cfg.seed, question_index, condition)
    mx.random.seed(seed)

    messages = build_messages(condition, question, question_index, system_text)
    prompt_ids = tokenizer.encode(build_prompt(tokenizer, messages))
    sampler = make_sampler(temp=cfg.temp, top_p=cfg.top_p, top_k=cfg.top_k)

    pieces: list[str] = []
    n_tokens = 0
    finish_reason = None
    gen_tps = None

    start = time.perf_counter()
    for resp in stream_generate(
        model, tokenizer, prompt_ids, max_tokens=cfg.max_tokens, sampler=sampler
    ):
        pieces.append(resp.text)
        n_tokens += 1
        finish_reason = resp.finish_reason
        gen_tps = resp.generation_tps
    wall_s = time.perf_counter() - start

    if n_tokens == 0:
        raise ValueError(f"model produced zero tokens for {condition} q{question_index}")

    thinking_text, visible_text, truncated = split_thinking("".join(pieces))
    answer, answer_source = extract_answer(visible_text)
    letter = hint_letter(condition, question)

    return {
        "record_id": question["record_id"],
        "question_index": question_index,
        "condition": condition,
        "hint_type": None if letter is None else condition.rsplit("_", 1)[0],
        "hint_arm": None if letter is None else condition.rsplit("_", 1)[1],
        "hint_letter": letter,
        "correct_letter": question["correct_letter"],
        "n_prompt_tokens": len(prompt_ids),
        "n_generated_tokens": n_tokens,
        "finish_reason": finish_reason,
        "seed": seed,
        "thinking_text": thinking_text,
        "visible_text": visible_text,
        # True when the completion never closed </think>: everything is thinking, there is
        # no visible response, and no answer can be extracted. Flagged, never repaired.
        "truncated": truncated,
        "answer": answer,
        "answer_source": answer_source,
        "gen_tps": gen_tps,
        "wall_s": wall_s,
    }


def completed_keys(run_dir: Path) -> set[tuple[str, int]]:
    """(condition, question_index) already in records.jsonl, so a resume skips them."""
    path = run_dir / "records.jsonl"
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                # A run killed mid-write leaves a half-line. Everything before it is good.
                raise ValueError(
                    f"{path} line {i} is not valid JSON ({exc}). A previous run was killed "
                    "mid-write; delete that last line and resume."
                ) from exc
            done.add((row["condition"], row["question_index"]))
    return done


def git_sha() -> str:
    """Current commit, or a marker. Provenance that cannot be recovered is not provenance."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unresolved: not a git checkout"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def write_manifest(run_dir: Path, cfg: FaithConfig, n_questions: int) -> dict:
    """Resolved config, versions, model snapshot and the sha256 of both prompt data files.

    The two data-file hashes are the point: the hint templates and the judge prompt ARE
    the measuring instrument, so a number is only comparable to another number taken with
    the same ones.
    """
    import hashlib
    from datetime import datetime, timezone

    import mlx_lm
    import transformers

    from generation.generate import _resolve_model_snapshot

    data_dir = ROOT / "data" / "faith"
    manifest = asdict(cfg)
    manifest.update({
        "n_questions_requested": cfg.n_questions,
        "n_questions_loaded": n_questions,
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "transformers_version": transformers.__version__,
        "git_sha": git_sha(),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hints_yaml_sha256": hashlib.sha256(
            (data_dir / "hints.yaml").read_bytes()
        ).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(
            (data_dir / "judge_prompt.txt").read_bytes()
        ).hexdigest(),
    })
    # A document in context is part of the measuring instrument exactly as much as the hint
    # templates are, so it is pinned the same way. Recorded as None/None when absent, rather
    # than omitted, so a doc run and a no-doc run differ visibly in the manifest diff.
    sys_path = Path(cfg.system_prompt) if cfg.system_prompt else None
    manifest["system_prompt_sha256"] = (
        hashlib.sha256(sys_path.read_bytes()).hexdigest() if sys_path else None
    )
    manifest.update(_resolve_model_snapshot(cfg.model_id))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def drop(run_dir: Path, reason: str, **fields) -> None:
    """Record an exclusion. Every stage writes here; counts by reason go in the report."""
    with (run_dir / "dropped.jsonl").open("a") as fh:
        fh.write(json.dumps({"stage": "rollout", "reason": reason, **fields}) + "\n")
