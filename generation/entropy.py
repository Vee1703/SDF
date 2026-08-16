"""Reconstruction of q, the effective generator distribution, and its entropy floor.

`research_context.md` defines q as the teacher *after* temperature, top-p and top-k --
not the raw softmax -- and records that the SFT loss is floored at H(q). The generation
backend does not hand us q: mlx-lm's `GenerationResponse.logprobs` is the raw
log-softmax, computed at `mlx_lm/generate.py:420` and only *then* passed to the sampler.
So we rebuild q here from the raw logprobs plus the decoding config.

The reconstruction mirrors `mlx_lm.sample_utils.make_sampler` step for step, including
its ordering: top-p is applied to the UNTEMPERED probabilities, then top-k, and
temperature is applied last inside `categorical_sampling`. HuggingFace and vLLM apply
temperature first, so the same (temp, top_p) pair defines a different q there. Any
Hq measured with this module is a statement about mlx-lm's q specifically.
"""

import math

import mlx.core as mx

# Below this temperature mlx-lm's make_sampler returns plain argmax over the RAW
# logprobs, skipping top-p and top-k entirely. q is then a point mass.
GREEDY_TEMP = 0.0


def effective_q_logprobs(
    raw_logprobs: mx.array,
    temp: float,
    top_p: float,
    top_k: int,
) -> mx.array:
    """Turn one step's raw log-softmax into a normalized log q over the vocabulary.

    Args:
        raw_logprobs: 1-D log-softmax over the vocabulary, as emitted by mlx-lm.
        temp, top_p, top_k: the decoding parameters that define q. Semantics match
            `mlx_lm.sample_utils.make_sampler`: top_p applies only when 0 < top_p < 1,
            top_k applies only when top_k > 0.

    Returns:
        1-D array of log q, with -inf on truncated tokens, summing to 1 in probability.
    """
    if raw_logprobs.ndim != 1:
        raise ValueError(
            f"expected a 1-D logprob vector, got shape {raw_logprobs.shape}"
        )

    lp = raw_logprobs.astype(mx.float32)

    if temp == GREEDY_TEMP:
        # Greedy: mlx-lm bypasses truncation entirely and takes argmax of the raw
        # logprobs, so q is a point mass and H(q) is exactly 0.
        q = mx.full(lp.shape, -mx.inf, dtype=mx.float32)
        return mx.where(mx.arange(lp.size) == mx.argmax(lp), 0.0, q)

    if 0.0 < top_p < 1.0:
        lp = _apply_top_p(lp, top_p)
    if top_k > 0:
        lp = _apply_top_k(lp, top_k)

    # Temperature last, matching categorical_sampling(logprobs, temp), which samples
    # from softmax(logprobs / temp) over the surviving support.
    scaled = lp * (1.0 / temp)
    q_logprobs = scaled - mx.logsumexp(scaled)

    # Truncated entries stay -inf under the division and the shift; make that explicit
    # rather than relying on inf arithmetic to have behaved.
    return mx.where(mx.isneginf(lp), -mx.inf, q_logprobs)


def _apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """Nucleus truncation, transcribed from mlx_lm.sample_utils.apply_top_p.

    Keeps token i when the mass of all tokens strictly more probable than i is below
    top_p, i.e. the minimal set whose cumulative probability reaches top_p.
    """
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs)  # ascending
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)

    inverse = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.size, dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative = mx.take_along_axis(cumulative, inverse, axis=-1)
    return mx.where(cumulative > 1.0 - top_p, logprobs, -mx.inf)


def _apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """Top-k truncation, transcribed from mlx_lm.sample_utils.apply_top_k."""
    vocab_size = logprobs.size
    if not 0 < top_k < vocab_size:
        raise ValueError(f"top_k must be in (0, {vocab_size}), got {top_k}")
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[top_k:]
    return mx.put_along_axis(
        logprobs, mask_idx, mx.array(-mx.inf, logprobs.dtype), axis=-1
    )


def step_entropy_nats(q_logprobs: mx.array) -> float:
    """H(q_t) = -sum_v q(v) log q(v), in nats, over the surviving support.

    Exact given the realized prefix -- this is the Rao-Blackwellized estimator, and it
    is what the Monte Carlo estimate -log q(y_t) converges to.
    """
    finite = ~mx.isneginf(q_logprobs)
    q = mx.exp(mx.where(finite, q_logprobs, -mx.inf))
    # 0 * -inf is nan, so zero out truncated positions before summing.
    terms = mx.where(finite, q * q_logprobs, 0.0)
    return float(-mx.sum(terms).item())


def sampled_token_logprob(q_logprobs: mx.array, token: int) -> float:
    """log q(y_t) at the token actually drawn.

    A sampled token can never sit outside q's support; if it does, the reconstruction
    disagrees with the sampler that produced it and every entropy number is void.
    """
    value = float(q_logprobs[token].item())
    if not math.isfinite(value):
        raise ValueError(
            f"sampled token {token} has log q = {value}: it fell outside the "
            "reconstructed support, so effective_q_logprobs does not match the "
            "sampler that drew it"
        )
    return value
