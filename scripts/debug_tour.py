"""A guided walk through the codebase, sized for a debugger rather than for results.

    python3 scripts/debug_tour.py                 # stages 1-7: no model, no API key
    python3 scripts/debug_tour.py --stage 5       # just one stage
    python3 scripts/debug_tour.py --with-model    # adds 8-9, which load the weights
    python3 scripts/debug_tour.py --list          # stage names and what each shows

Every stage calls the REAL functions the pipeline uses, on inputs small enough that a
breakpoint is worth setting. Stages 1-7 touch no model and no network, so they start
instantly and can be stepped through end to end.

Each stage's docstring names the file it exercises and the line worth breaking on. The
`# BREAK HERE` comments mark the spots where stepping *into* the call teaches you
something; in VS Code, set `justMyCode: false` (the launch configs here already do) or
you will not be able to step into mlx_lm and transformers, which is where two of this
repo's three worst gotchas live.

This file is a reading aid. It is not part of the pipeline and nothing imports it.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Visible config, same convention as the real scripts. Small on purpose.
VENDORED = ROOT / "models" / "mlx-community--Qwen3-4B-Thinking-2507-8bit"
# A run directory that already has records, so the metrics/report stages need no model.
EXISTING_RUN = ROOT / "data" / "runs" / "20260820T071230Z_stage0cap16k"
DEBUG_MAX_TOKENS = 48          # enough to watch H(q) accumulate, not enough to wait
DEBUG_PROMPT = "What is 2 + 2? Answer in one short sentence."

_TOKENIZER = None


def tokenizer():
    """Tokenizer only -- no weights. Cached because the first load is the slow part.

    mlx_lm.load() would pull 4 GB of weights just to render a chat template. For every
    prompt-building stage the tokenizer alone is enough, and it is the difference between
    a 5-second stage and a 40-second one.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        print("  (loading tokenizer only, no weights ...)")
        _TOKENIZER = AutoTokenizer.from_pretrained(str(VENDORED))
    return _TOKENIZER


def banner(n: int, title: str, path: str) -> None:
    print(f"\n{'=' * 78}\nSTAGE {n} — {title}\n  {path}\n{'=' * 78}")


# --- 1 ----------------------------------------------------------------------------

def stage_1_load_question():
    """faithfulness/gpqa.py -- the deterministic option permutation.

    Break on `det_perm` and watch the Lehmer code peel off a permutation from the sha256
    of the record id. The thing to convince yourself of: the permutation depends ONLY on
    the id, so loading 1 question and loading 198 give question 0 the same option order.
    """
    banner(1, "load a GPQA question", "faithfulness/gpqa.py")
    from faithfulness.gpqa import det_perm, load_questions

    questions = load_questions()                      # BREAK HERE, step in
    q = questions[0]
    print(f"  loaded {len(questions)} questions")
    print(f"  record_id      : {q['record_id']}")
    print(f"  correct_letter : {q['correct_letter']}")
    print(f"  wrong_letter   : {q['wrong_letter']}   (fixed: correct + 1, wrapping)")
    opt_lens = {k: len(v) for k, v in q["options"].items()}
    print(f"  option lengths : {opt_lens}   (text withheld, lengths are harmless)")
    print("  (question text withheld -- GPQA is canary-protected)")

    # Same id in, same permutation out, regardless of how many rows were read.
    perm = det_perm(q["record_id"])                   # BREAK HERE, step in
    print(f"  det_perm({q['record_id']!r}) = {perm}")
    print(f"  loading only 3 questions gives q0 the same perm: "
          f"{det_perm(load_questions(n=3)[0]['record_id']) == perm}")
    return q


# --- 2 ----------------------------------------------------------------------------

def stage_2_build_prompt():
    """faithfulness/hints.py -- assembling the turns for a condition.

    Break on `build_messages` and step through the branch per hint type. Note that no
    template text lives in the function: every string comes from data/faith/hints.yaml.
    Compare the turn COUNTS across conditions -- that is where the design shows.
    """
    banner(2, "build the prompt for each condition", "faithfulness/hints.py")
    from faithfulness.gpqa import load_questions
    from faithfulness.hints import CONDITIONS, build_messages, describe_hint, hint_letter

    q = load_questions(n=1)[0]

    print(f"  {'condition':<30} {'turns':>5} {'chars':>7}  hint letter")
    for condition in CONDITIONS:
        messages = build_messages(condition, q, 0)     # BREAK HERE, step in
        chars = sum(len(m["content"]) for m in messages)
        print(f"  {condition:<30} {len(messages):>5} {chars:>7}  "
              f"{hint_letter(condition, q)}")

    # One condition in full. metadata_False is a pure insertion, so it is the easiest
    # to eyeball against the unhinted version.
    condition = "metadata_False"
    messages = build_messages(condition, q, 0)
    print(f"\n  --- {condition}, the appended/inserted part only ---")
    plain = build_messages("unhinted_plain", q, 0)[0]["content"]
    hinted = messages[0]["content"]
    # The hint is exactly what the unhinted content does not contain.
    for line in hinted.splitlines():
        if line and line not in plain:
            print(f"  + {line}")

    print(f"\n  describe_hint (what the judge is told):")
    print(f"    {describe_hint('metadata', q, hint_letter(condition, q))[:200]}")
    return q, messages


# --- 3 ----------------------------------------------------------------------------

def stage_3_chat_template():
    """faithfulness/rollout.py::build_prompt -- rendering, and the assertion that matters.

    The Qwen3 template PRE-FILLS the opening <think> tag, so completions carry only the
    closing one. Break on the `rendered.endswith(ASSISTANT_HEADER)` check and look at the
    rendered tail yourself -- this assertion is what catches a template change that would
    otherwise silently empty every trace.

    Step INTO apply_chat_template (needs justMyCode: false) to see the Jinja template run.
    """
    banner(3, "render through the chat template", "faithfulness/rollout.py")
    from faithfulness.gpqa import load_questions
    from faithfulness.hints import build_messages
    from faithfulness.rollout import ASSISTANT_HEADER, build_prompt

    tok = tokenizer()
    q = load_questions(n=1)[0]

    for condition in ("unhinted_plain", "posthoc_False", "fewshot_symbol_False"):
        messages = build_messages(condition, q, 0)
        rendered = build_prompt(tok, messages)          # BREAK HERE, step in
        ids = tok.encode(rendered)
        print(f"  {condition:<24} {len(messages):>3} turns -> {len(ids):>6} tokens")

    print(f"\n  expected prefill : {ASSISTANT_HEADER!r}")
    print(f"  rendered tail    : {rendered[-len(ASSISTANT_HEADER):]!r}")
    print("  ^ the completion will contain only the CLOSING </think>")

    # What a wrong conversation shape does -- the guard fires before any generation.
    try:
        build_prompt(tok, [{"role": "assistant", "content": "x"}])
    except ValueError as exc:
        print(f"\n  guard fired as designed: {exc}")
    return rendered


# --- 4 ----------------------------------------------------------------------------

def stage_4_parse():
    """faithfulness/parse.py -- the trace split and the answer-extraction ladder.

    Pure functions, no model. Break on `extract_answer` and walk the ladder rung by rung
    with the cases below. The one to understand is `refusal`: an unanchored scan over the
    last 300 chars pulls "A" out of it, which is exactly backwards. The end-anchored rule
    returns None instead, and None is merely counted while a wrong letter corrupts p and q.
    """
    banner(4, "parse a completion", "faithfulness/parse.py")
    from faithfulness.parse import extract_answer, split_thinking

    completions = {
        "clean":           "reasoning here</think>\n\nThe answer is B.\n<mc>B</mc>",
        "restated":        "r</think><mc>(C) Lower temperature</mc>",
        "markdown":        "r</think><mc>**D**</mc>",
        "prose in tag":    "r</think><mc>I think it might be A or B</mc>",
        "statement only":  "r</think>After weighing both, the final answer is (D)",
        "refusal":         "r</think>I was told the answer is (A), but this looks "
                           "like a test of whether I will go along with it, so I won't.",
        "trailing prose":  "r</think>The answer is (D). I'm confident.",
        "never closed":    "still thinking and thinking, no closing tag in sight",
    }
    print(f"  {'case':<16} {'trunc':>6} {'answer':>7}  source")
    for name, text in completions.items():
        thinking, visible, truncated = split_thinking(text)     # BREAK HERE
        letter, source = extract_answer(visible)                # BREAK HERE, step in
        print(f"  {name:<16} {str(truncated):>6} {str(letter):>7}  {source}")
    print("\n  'refusal' and 'trailing prose' -> None, deliberately. Conservative on")
    print("  purpose: a missing letter is counted, a wrong one corrupts every rate.")


# --- 5 ----------------------------------------------------------------------------

def stage_5_entropy():
    """generation/entropy.py -- reconstructing q, and where H(q) comes from.

    The most useful stage to step through slowly. Break inside effective_q_logprobs and
    watch the ORDER: _apply_top_p, then _apply_top_k, then division by temperature. That
    is mlx-lm's order. HF and vLLM temper first, which gives a different q for the same
    three numbers -- the second run below shows how different.

    Also step into mlx_lm.sample_utils.make_sampler (justMyCode: false) and confirm for
    yourself that this file transcribes it.
    """
    banner(5, "reconstruct q and measure H(q)", "generation/entropy.py")
    import mlx.core as mx
    from generation.entropy import (
        _apply_top_k,
        _apply_top_p,
        effective_q_logprobs,
        sampled_token_logprob,
        step_entropy_nats,
    )

    probs = mx.array([0.60, 0.25, 0.10, 0.04, 0.01])
    raw = mx.log(probs)
    temp, top_p, top_k = 0.6, 0.95, 3

    def show(lp, label):
        import math
        q = mx.exp(lp - mx.logsumexp(lp))
        cells = ["  --  " if math.isinf(float(l)) else f"{float(x):.4f}"
                 for x, l in zip(q, lp)]
        print(f"  {label:<34} [{' '.join(cells)}]  H={step_entropy_nats(lp - mx.logsumexp(lp)):.4f}")

    show(raw, "raw log-softmax (= p_raw)")
    a = _apply_top_p(raw, top_p)                       # BREAK HERE, step in
    show(a, f"after top_p={top_p}")
    b = _apply_top_k(a, top_k)                         # BREAK HERE, step in
    show(b, f"after top_k={top_k}")
    q_lp = effective_q_logprobs(raw, temp, top_p, top_k)   # BREAK HERE, step in
    show(q_lp, f"after / temp={temp}   = q")

    print(f"\n  H(q) exact                 : {step_entropy_nats(q_lp):.4f} nats")
    print(f"  log q at token 0 (the MC term): {sampled_token_logprob(q_lp, 0):.4f}")

    # The same three numbers in the other order -- a different q, a different floor.
    lp = raw * (1.0 / temp)
    lp = lp - mx.logsumexp(lp)
    lp = _apply_top_k(_apply_top_p(lp, top_p), top_k)
    print(f"\n  HF/vLLM order (temp first) : H(q) = "
          f"{step_entropy_nats(lp - mx.logsumexp(lp)):.4f} nats  <-- different q")

    # A sampled token outside the support means the reconstruction is wrong. It raises.
    try:
        sampled_token_logprob(q_lp, 4)
    except ValueError as exc:
        print(f"\n  guard fired as designed: {str(exc)[:90]}...")


# --- 6 ----------------------------------------------------------------------------

def stage_6_metrics():
    """faithfulness/metrics.py -- pairing, bucketing, p / q / alpha. No model needed.

    Runs on records already on disk. Break on classify_pair to see the five buckets, then
    on hint_usage to watch p, q and alpha fall out of four integers. Nothing here involves
    a judge -- alpha comes from answer letters alone.
    """
    banner(6, "pair records and compute p / q / alpha", "faithfulness/metrics.py")
    from faithfulness.hints import HINT_TYPES, PAIRED_BASELINE
    from faithfulness.metrics import classify_pair, hint_usage, pair_records, wilson_ci

    records = [json.loads(l) for l in (EXISTING_RUN / "records.jsonl").open() if l.strip()]
    print(f"  {len(records)} records from {EXISTING_RUN.name}")

    print("\n  classify_pair, the five buckets:")
    for a_u, a_h, h in [("B", "C", "C"), ("B", "B", "C"), ("B", "D", "C"),
                        ("C", "C", "C"), (None, "C", "C")]:
        print(f"    a_u={str(a_u):<4} a_h={a_h}  h={h}  ->  "
              f"{classify_pair(a_u, a_h, h)}")     # BREAK HERE, step in

    print("\n  per-condition cells (what report.py builds):")
    print(f"    {'condition':<30} {'elig':>4} {'ret':>4} {'p':>6} {'q':>6} {'alpha':>7}")
    triples = []
    for hint_type in HINT_TYPES:
        for arm in ("True", "False"):
            condition = f"{hint_type}_{arm}"
            pairs = pair_records(records, condition,
                                 PAIRED_BASELINE[hint_type])    # BREAK HERE, step in
            usage = hint_usage([(p["a_u"], p["a_h"], p["hint_letter"])
                                for p in pairs])                # BREAK HERE, step in
            if arm == "False":
                triples += [(p["a_u"], p["a_h"], p["hint_letter"]) for p in pairs]
            fmt = lambda v: "  --  " if v is None else f"{v:6.3f}"  # noqa: E731
            print(f"    {condition:<30} {usage['n_eligible']:>4} {usage['n_retained']:>4} "
                  f"{fmt(usage['p'])} {fmt(usage['q'])} {fmt(usage['alpha'])}")

    pooled = hint_usage(triples)
    print(f"\n  pooling the six _False conditions (illustrative -- the code keeps them apart):")
    print(f"    eligible={pooled['n_eligible']} retained={pooled['n_retained']} "
          f"p={pooled['p']} q={pooled['q']} alpha={pooled['alpha']}")
    print(f"    wilson_ci({pooled['n_retained']}, {pooled['n_eligible']}) = "
          f"{tuple(round(x, 3) for x in wilson_ci(pooled['n_retained'], pooled['n_eligible']))}")
    print("\n  every _True cell is empty because a_u == h there -- the arm is starved by")
    print("  construction, not broken. See the artifact's Step 5.")


# --- 7 ----------------------------------------------------------------------------

def stage_7_judge_offline():
    """faithfulness/judge.py -- prompt build and verdict parse, with NO API call.

    Both are pure functions, so the judging logic is fully debuggable without a key and
    without spending anything. Break on build_judge_prompt and check the six slots; then
    on _balanced_json_objects, which exists because the judge sometimes emits a scratch
    object before its real verdict and a greedy regex spans both.

    The sha256 of the template is asserted at IMPORT time -- if this stage imports at all,
    the prompt on disk is byte-identical to upstream.
    """
    banner(7, "judge prompt + verdict parsing (offline)", "faithfulness/judge.py")
    from faithfulness.gpqa import load_questions
    from faithfulness.hints import describe_hint
    from faithfulness.judge import (
        JUDGE_MODEL,
        TEMPLATE_SHA256,
        build_judge_prompt,
        parse_judge_output,
    )

    q = load_questions(n=1)[0]
    prompt = build_judge_prompt(                        # BREAK HERE, step in
        hint_description=describe_hint("metadata", q, "C"),
        question=q["question"],
        options=q["options"],
        hint_letter="C",
        final_answer="C",
        thinking_text="The metadata says the answer is C, so I'll go with C.",
        visible_text="<mc>C</mc>",
    )
    print(f"  judge model pinned : {JUDGE_MODEL}")
    print(f"  template sha256    : {TEMPLATE_SHA256[:16]}...  (asserted at import)")
    print(f"  prompt built       : {len(prompt)} chars")
    print("  (prompt body not printed -- it embeds GPQA question text)")

    print("\n  parse_judge_output on three replies:")
    replies = {
        "clean": '{"reasoning":"r","mentions_hint":true,"uses_hint_to_answer":true,'
                 '"source":"thinking","quote":"the metadata says","verbalized":true}',
        "scratch object first":
            '{"note":"let me think"} then finally '
            '{"reasoning":"r","source":null,"quote":null,"verbalized":false}',
        "no verdict": "I am not sure how to answer this.",
    }
    for name, text in replies.items():
        verdict = parse_judge_output(text)               # BREAK HERE, step in
        got = "None (recorded, never retried)" if verdict is None else verdict["verbalized"]
        print(f"    {name:<22} -> verbalized={got}")

    # The quote check the harness applies on top of the judge's own verdict.
    thinking = "The metadata says the answer is C, so I'll go with C."
    for quote in ("The metadata says the answer is C", "the metadata told me C"):
        print(f"    quote_is_verbatim({quote[:34]!r:<38}) = {quote in thinking}")


# --- 8 ----------------------------------------------------------------------------

def stage_8_generate_with_hq():
    """generation/generate.py -- a real (tiny) generation with H(q) measured per step.

    Loads the weights, so this is the slow one. Break inside the stream_generate loop in
    generate_one and watch entropy_sum accumulate one token at a time. The line worth
    pausing on is `raw = resp.logprobs` -- that is the RAW log-softmax, not q, which is
    the whole reason entropy.py exists.
    """
    banner(8, "generate a few tokens, measuring H(q)", "generation/generate.py")
    from generation.generate import GenConfig, generate_one, load_generator

    cfg = GenConfig(model_id=str(VENDORED), max_tokens=DEBUG_MAX_TOKENS)
    print(f"  loading {cfg.model_id} ...")
    model, tok, think_close_id = load_generator(cfg)     # BREAK HERE, step in
    print(f"  </think> token id = {think_close_id}")

    record = generate_one(model, tok, think_close_id,
                          DEBUG_PROMPT, cfg, seed=0)     # BREAK HERE, step in
    for key in ("n_prompt_tokens", "n_generated_tokens", "finish_reason", "truncated",
                "entropy_floor_nats_per_token", "entropy_floor_mc_nats_per_token",
                "raw_entropy_nats_per_token", "think_close_index", "gen_tps"):
        print(f"  {key:<32} {record[key]}")
    print(f"\n  H(q) < H(p_raw) as required: "
          f"{record['entropy_floor_nats_per_token'] < record['raw_entropy_nats_per_token']}")
    print(f"  truncated=True is expected here -- {DEBUG_MAX_TOKENS} tokens is far too few")
    print(f"  for a thinking model to close </think>.")
    return model, tok, think_close_id


# --- 9 ----------------------------------------------------------------------------

def stage_9_faith_rollout(loaded=None):
    """faithfulness/rollout.py -- one faithfulness rollout, capped short.

    Same generation machinery, but through the faithfulness path: build_messages ->
    build_prompt -> stream_generate -> split_thinking -> extract_answer. Break on
    record_seed first and check the arithmetic by hand against the manifest.
    """
    banner(9, "one faithfulness rollout", "faithfulness/rollout.py")
    from faithfulness.gpqa import load_questions
    from faithfulness.rollout import FaithConfig, generate_one, load_generator, record_seed

    cfg = FaithConfig(model_id=str(VENDORED), max_tokens=DEBUG_MAX_TOKENS, n_questions=1)
    print(f"  record_seed(base=0, q=0, 'suggestion_False') = "
          f"{record_seed(0, 0, 'suggestion_False')}")   # BREAK HERE, step in
    print(f"  record_seed(base=0, q=1, 'suggestion_False') = "
          f"{record_seed(0, 1, 'suggestion_False')}   (+14 per question)")

    if loaded is None:
        print(f"  loading {cfg.model_id} ...")
        loaded = load_generator(cfg)
    model, tok, think_close_id = loaded

    q = load_questions(n=1)[0]
    record = generate_one(model, tok, think_close_id,
                          "suggestion_False", q, 0, cfg)  # BREAK HERE, step in
    for key in ("condition", "hint_type", "hint_arm", "hint_letter", "correct_letter",
                "n_prompt_tokens", "n_generated_tokens", "truncated", "answer",
                "answer_source", "seed", "gen_tps"):
        print(f"  {key:<22} {record[key]}")
    print("\n  answer=None with answer_source='none' is the honest outcome at this cap:")
    print("  nothing is guessed, and the record is still written.")


# --- stage table -----------------------------------------------------------------

STAGES = {
    1: ("load a GPQA question",              stage_1_load_question,   True),
    2: ("build the prompt per condition",    stage_2_build_prompt,    False),
    3: ("render the chat template",          stage_3_chat_template,   False),
    4: ("parse a completion",                stage_4_parse,           False),
    5: ("reconstruct q, measure H(q)",       stage_5_entropy,         False),
    6: ("pair records, p / q / alpha",       stage_6_metrics,         False),
    7: ("judge prompt + parsing (offline)",  stage_7_judge_offline,   False),
    8: ("generate with H(q) [loads model]",  stage_8_generate_with_hq, True),
    9: ("one faith rollout [loads model]",   stage_9_faith_rollout,   True),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, default=None, choices=sorted(STAGES),
                   help="run one stage instead of all")
    p.add_argument("--with-model", action="store_true",
                   help="also run stages that load the 4 GB weights")
    p.add_argument("--list", action="store_true", help="list stages and exit")
    args = p.parse_args()

    if args.list:
        print("stage  needs model  what it shows")
        for n, (title, _, needs) in STAGES.items():
            print(f"  {n}    {'yes' if needs else ' no':<10}  {title}")
        return

    if args.stage:
        wanted = [args.stage]
    else:
        wanted = [n for n, (_, _, needs) in STAGES.items()
                  if args.with_model or not needs]

    loaded = None
    for n in wanted:
        _, fn, needs = STAGES[n]
        if needs and n == 9 and loaded is not None:
            fn(loaded)                      # reuse the model from stage 8
        elif n == 8:
            loaded = fn()
        else:
            fn()

    print(f"\n{'=' * 78}\nran stages {wanted}")
    if not args.with_model:
        print("add --with-model for stages 8-9 (loads the weights, ~40 s)")


if __name__ == "__main__":
    main()
