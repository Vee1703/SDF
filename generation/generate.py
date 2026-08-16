"""Load a CoT-tuned Qwen, sample from it, and record what q looked like while doing it.

This is the generation stage of the SDF pipeline. It produces one record per prompt,
carrying the trace/answer split and the entropy floor H(q) measured at the decoding
settings actually used. Per `research_context.md`, the generate-then-filter pipeline is
where SDF's substance lives, and the decoding parameters are the object of study -- so
they sit in GenConfig, visible and serialized next to every artifact, never inline.

Only the decoding half of q is defined here. Filtering (the verifier) is a later stage.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm import load as _mlx_load
from mlx_lm.sample_utils import make_sampler

from generation.entropy import (
    effective_q_logprobs,
    sampled_token_logprob,
    step_entropy_nats,
)

# The Qwen3 chat template pre-fills the opening think tag, so a generated completion
# contains only the CLOSING tag. Splitting on a <think>...</think> pair finds nothing
# and yields a silently empty trace.
THINK_CLOSE = "</think>"
ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n"

# Written by scripts/fetch_model.py into models/<slug>/. A --local-dir download has no
# snapshots/<sha>/ path to read a commit out of, so the sha lives in here instead.
MANIFEST = "download.json"


def is_local_model_dir(model_id: str) -> bool:
    """True when model_id points at a directory on disk rather than a hub repo id."""
    return Path(model_id).is_dir()


@dataclass
class GenConfig:
    """Everything that defines q's decoding half. Serialized with every run."""

    # 8-bit rather than 4-bit: quantization perturbs the output distribution, and the
    # output distribution IS q, the thing being measured.
    # A hub id resolves through the shared ~/.cache/huggingface. Pass a local path such
    # as "models/mlx-community--Qwen3-4B-Thinking-2507-8bit" (see scripts/fetch_model.py)
    # to load weights vendored into the repo instead.
    model_id: str = "mlx-community/Qwen3-4B-Thinking-2507-8bit"
    # Pin the hub weights to a branch, tag or commit sha. None means "whatever the hub
    # serves now" -- fine for exploration, but a silent re-download can change q and
    # therefore H(q), so pin it for anything whose numbers get compared. Not applicable
    # to local paths, which are pinned by their download.json instead.
    revision: str | None = None
    # Qwen3-Thinking-2507's own generation_config.json defaults.
    temp: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 2048
    seed: int = 0
    backend: str = "mlx-lm"
    # mlx-lm truncates on the untempered distribution and applies temperature last.
    # HF and vLLM apply temperature first, so the same numbers define a different q.
    truncation_order: str = "top_p -> top_k -> temperature (mlx-lm order)"


def load_generator(cfg: GenConfig):
    """Load model and tokenizer, and resolve the token id marking the trace boundary."""
    if is_local_model_dir(cfg.model_id) and cfg.revision is not None:
        raise ValueError(
            f"revision={cfg.revision!r} is meaningless for the local weights at "
            f"{cfg.model_id}; those are pinned by their {MANIFEST} instead. "
            "Drop --revision, or point --model at a hub id."
        )
    model, tokenizer = _mlx_load(cfg.model_id, revision=cfg.revision)

    close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
    if len(close_ids) != 1:
        raise ValueError(
            f"{THINK_CLOSE!r} tokenizes to {len(close_ids)} tokens ({close_ids}); "
            "the token-space trace boundary assumes it is a single special token"
        )
    return model, tokenizer, close_ids[0]


def build_prompt(tokenizer, user_text: str) -> str:
    """Render the chat template, asserting the thinking prefill is where we expect."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if not rendered.endswith(ASSISTANT_HEADER):
        raise ValueError(
            "chat template did not end with the expected thinking prefill.\n"
            f"  expected suffix: {ASSISTANT_HEADER!r}\n"
            f"  got suffix:      {rendered[-60:]!r}\n"
            "The trace/answer split depends on this; refusing to generate."
        )
    return rendered


def split_trace_answer(text: str, tokens: list[int], think_close_id: int) -> dict:
    """Split a completion into trace and answer at the closing think tag.

    Returns think_close_index in token space -- the boundary the future completion_mask
    will key on. A completion that never closed its trace is flagged, not repaired.
    """
    in_tokens = think_close_id in tokens
    in_text = THINK_CLOSE in text
    if in_tokens != in_text:
        raise ValueError(
            f"trace boundary disagrees between token space (present={in_tokens}) and "
            f"text space (present={in_text}); detokenization is not round-tripping"
        )

    if not in_tokens:
        return {
            "trace": text,
            "answer": None,
            "truncated": True,
            "think_close_index": None,
            "n_trace_tokens": len(tokens),
            "n_answer_tokens": 0,
        }

    idx = tokens.index(think_close_id)
    trace_text, answer_text = text.split(THINK_CLOSE, 1)
    return {
        "trace": trace_text.strip(),
        "answer": answer_text.strip(),
        "truncated": False,
        "think_close_index": idx,
        "n_trace_tokens": idx,
        "n_answer_tokens": len(tokens) - idx - 1,
    }


def generate_one(model, tokenizer, think_close_id, user_text: str, cfg: GenConfig,
                 seed: int) -> dict:
    """Sample one completion and measure H(q) along the sampled path.

    The per-step logprob vector is reduced to scalars immediately and discarded: the
    vocabulary is ~152k wide, so retaining 2048 steps of raw vectors would cost over a
    gigabyte per completion.
    """
    mx.random.seed(seed)

    prompt = build_prompt(tokenizer, user_text)
    prompt_ids = tokenizer.encode(prompt)
    sampler = make_sampler(temp=cfg.temp, top_p=cfg.top_p, top_k=cfg.top_k)

    tokens: list[int] = []
    pieces: list[str] = []
    entropy_sum = 0.0        # exact, sum of H(q_t)
    mc_logprob_sum = 0.0     # Monte Carlo, sum of log q(y_t)
    raw_entropy_sum = 0.0    # H(p_raw), for contrast: q must be narrower
    finish_reason = None
    gen_tps = None

    start = time.perf_counter()
    for resp in stream_generate(
        model,
        tokenizer,
        prompt_ids,
        max_tokens=cfg.max_tokens,
        sampler=sampler,
    ):
        raw = resp.logprobs  # mlx-lm hands back the RAW log-softmax, not q
        q_logprobs = effective_q_logprobs(raw, cfg.temp, cfg.top_p, cfg.top_k)

        entropy_sum += step_entropy_nats(q_logprobs)
        mc_logprob_sum += sampled_token_logprob(q_logprobs, resp.token)
        raw_entropy_sum += step_entropy_nats(raw.astype(mx.float32))

        tokens.append(resp.token)
        pieces.append(resp.text)
        finish_reason = resp.finish_reason
        gen_tps = resp.generation_tps
    wall_s = time.perf_counter() - start

    n = len(tokens)
    if n == 0:
        raise ValueError(f"model produced zero tokens for prompt {user_text!r}")

    record = {
        "prompt": user_text,
        "n_prompt_tokens": len(prompt_ids),
        "n_generated_tokens": n,
        "finish_reason": finish_reason,
        "seed": seed,
        # The entropy floor, both estimators. They target the same quantity; a gap
        # between them means the q reconstruction disagrees with the sampler.
        "entropy_floor_nats_per_token": entropy_sum / n,
        "entropy_floor_mc_nats_per_token": -mc_logprob_sum / n,
        "mean_sampled_logprob": mc_logprob_sum / n,
        "raw_entropy_nats_per_token": raw_entropy_sum / n,
        "gen_tps": gen_tps,
        "wall_s": wall_s,
    }
    record.update(split_trace_answer("".join(pieces), tokens, think_close_id))
    return record


def config_provenance(cfg: GenConfig) -> dict:
    """Resolved config plus versions and the model snapshot sha. Unrecoverable
    provenance means unusable data."""
    import mlx_lm
    import transformers

    provenance = asdict(cfg)
    provenance["mlx_version"] = mx.__version__
    provenance["mlx_lm_version"] = mlx_lm.__version__
    provenance["transformers_version"] = transformers.__version__
    provenance.update(_resolve_model_snapshot(cfg.model_id))
    return provenance


def _resolve_model_snapshot(model_id: str) -> dict:
    """Recover the commit sha the loaded weights came from.

    Two layouts, two mechanisms: weights vendored into models/<slug>/ carry their sha in
    download.json, while hub ids resolve through the shared cache, whose snapshot
    directory is named after the sha.
    """
    if is_local_model_dir(model_id):
        manifest_path = Path(model_id) / MANIFEST
        if not manifest_path.exists():
            # Loud, not silent: a local dir with no manifest is unpinned weights, and
            # every H(q) measured against them is uncomparable.
            return {
                "model_source": "local_dir",
                "model_repo_id": None,
                "model_snapshot": f"unresolved: no {MANIFEST} in {model_id}",
            }
        manifest = json.loads(manifest_path.read_text())
        return {
            "model_source": "local_dir",
            "model_repo_id": manifest.get("repo_id"),
            "model_snapshot": manifest.get("sha"),
            "model_fetched_utc": manifest.get("fetched_utc"),
        }

    try:
        from mlx_lm.utils import hf_repo_to_path

        return {
            "model_source": "hf_cache",
            "model_repo_id": model_id,
            "model_snapshot": hf_repo_to_path(model_id).name,
        }
    except Exception as exc:  # cache layout is not load-bearing; record the failure
        return {
            "model_source": "hf_cache",
            "model_repo_id": model_id,
            "model_snapshot": f"unresolved: {exc}",
        }
