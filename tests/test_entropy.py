"""Invariants for the q reconstruction. No model, no GPU -- runs in under a second.

These exist because a silent truncation bug is unfalsifiable from the output alone: a
wrong q still produces a plausible-looking entropy number. Every assertion here is one
whose answer can be worked out by hand.
"""

import math

import mlx.core as mx
import pytest

from generation.entropy import (
    effective_q_logprobs,
    sampled_token_logprob,
    step_entropy_nats,
)


def logprobs_from_probs(probs):
    return mx.log(mx.array(probs, dtype=mx.float32))


def support(q_logprobs):
    """Indices q assigns non-zero probability to."""
    return {i for i, v in enumerate(q_logprobs.tolist()) if math.isfinite(v)}


def test_uniform_full_support_has_log_v_entropy():
    v = 8
    q = effective_q_logprobs(
        logprobs_from_probs([1.0 / v] * v), temp=1.0, top_p=1.0, top_k=0
    )
    assert step_entropy_nats(q) == pytest.approx(math.log(v), abs=1e-6)


def test_pass_through_leaves_logprobs_unchanged():
    """temp=1, no truncation must be the identity, not merely close to it."""
    raw = logprobs_from_probs([0.5, 0.3, 0.15, 0.05])
    q = effective_q_logprobs(raw, temp=1.0, top_p=1.0, top_k=0)
    assert mx.allclose(q, raw, atol=1e-6).item()


@pytest.mark.parametrize(
    "temp,top_p,top_k",
    [(1.0, 1.0, 0), (0.6, 0.95, 20), (0.6, 0.95, 0), (2.0, 0.5, 3), (0.1, 1.0, 2)],
)
def test_q_is_a_normalized_distribution(temp, top_p, top_k):
    raw = mx.log(mx.softmax(mx.random.normal((64,), key=mx.random.key(0))))
    q = effective_q_logprobs(raw, temp=temp, top_p=top_p, top_k=top_k)
    total = float(mx.sum(mx.exp(mx.where(mx.isneginf(q), -mx.inf, q))).item())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_low_temperature_drives_entropy_to_zero():
    raw = logprobs_from_probs([0.5, 0.3, 0.15, 0.05])
    assert step_entropy_nats(
        effective_q_logprobs(raw, temp=0.01, top_p=1.0, top_k=0)
    ) == pytest.approx(0.0, abs=1e-6)


def test_top_k_one_is_a_point_mass():
    raw = logprobs_from_probs([0.5, 0.3, 0.15, 0.05])
    q = effective_q_logprobs(raw, temp=0.6, top_p=1.0, top_k=1)
    assert support(q) == {0}
    assert step_entropy_nats(q) == pytest.approx(0.0, abs=1e-9)


def test_greedy_temperature_is_a_point_mass_at_the_raw_argmax():
    """temp=0 in mlx-lm bypasses truncation, so the mode is the RAW mode."""
    raw = logprobs_from_probs([0.15, 0.5, 0.3, 0.05])
    q = effective_q_logprobs(raw, temp=0.0, top_p=0.5, top_k=2)
    assert support(q) == {1}
    assert step_entropy_nats(q) == pytest.approx(0.0, abs=1e-9)


def test_top_p_keeps_the_minimal_nucleus():
    """Descending cumsum 0.5, 0.8, 0.95 first reaches 0.9 at the third token."""
    raw = logprobs_from_probs([0.5, 0.3, 0.15, 0.05])
    q = effective_q_logprobs(raw, temp=1.0, top_p=0.9, top_k=0)
    assert support(q) == {0, 1, 2}


def test_top_p_nucleus_can_be_smaller_than_it_looks():
    """0.9 + 0.06 = 0.96 passes top_p=0.95 at the second token, so 0.03 is dropped.

    Deliberately not a tie: with probs summing to exactly top_p at the boundary (e.g.
    [0.9, 0.05, 0.03, 0.02] at top_p=0.95) the float32 cumsum lands within 5e-9 of the
    threshold and rounding, not the rule, decides whether the token survives.
    """
    raw = logprobs_from_probs([0.9, 0.06, 0.03, 0.01])
    q = effective_q_logprobs(raw, temp=1.0, top_p=0.95, top_k=0)
    assert support(q) == {0, 1}


def test_truncation_lowers_entropy():
    """The whole point of the H(q) instrument: q is narrower than the raw softmax."""
    raw = mx.log(mx.softmax(mx.random.normal((256,), key=mx.random.key(1))))
    h_raw = step_entropy_nats(effective_q_logprobs(raw, 1.0, 1.0, 0))
    h_q = step_entropy_nats(effective_q_logprobs(raw, 0.6, 0.95, 20))
    assert h_q < h_raw


def test_renormalization_matches_hand_computed_values():
    """temp=0.5 on [0.5, 0.3, 0.15, 0.05] truncated to top-2, worked out by hand."""
    raw = logprobs_from_probs([0.5, 0.3, 0.15, 0.05])
    q = effective_q_logprobs(raw, temp=0.5, top_p=1.0, top_k=2)
    # q ∝ p^(1/temp) = p^2 over {0.5, 0.3}: 0.25 and 0.09, normalized to 0.7353/0.2647.
    expected = [0.25 / 0.34, 0.09 / 0.34]
    got = [math.exp(q[0].item()), math.exp(q[1].item())]
    assert got == pytest.approx(expected, abs=1e-5)


def test_sampled_token_outside_support_is_loud():
    raw = logprobs_from_probs([0.5, 0.3, 0.15, 0.05])
    q = effective_q_logprobs(raw, temp=0.6, top_p=1.0, top_k=2)
    assert sampled_token_logprob(q, 0) < 0.0
    with pytest.raises(ValueError, match="outside the reconstructed support"):
        sampled_token_logprob(q, 3)
