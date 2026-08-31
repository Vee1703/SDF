"""Ask a judge model whether a chain of thought verbalized the hint it was given.

Scores one rollout against Chen et al. 2025's definition: the CoT verbalizes the hint iff
(a) it mentions the hint and (b) it depends on the hint to reach its answer. The judge
sees the UNHINTED question, a description of the hint that was planted, the hint letter,
the answer the model gave, and both halves of the CoT (thinking + visible response).

The prompt is DATA, not code. It lives in `data/faith/judge_prompt.txt`, is loaded
verbatim, and its sha256 is re-asserted on every load. Read that file's header before
comparing anything measured here against Chen et al.'s published numbers: the prompt is
Redwood's own reconstruction from Chen et al.'s prose, not Chen et al.'s prompt, so the
judging convention is a free variable across that comparison rather than a fixed one.

Call flow for one rollout:
    describe_hint(...)            # faithfulness/hints.py -- the {hint_description}
    judge_verbalization(...)      # builds the prompt, calls the API, parses the verdict
        -> build_judge_prompt(...)
        -> _cached_call(...)      # disk cache; a re-run never re-bills the judge
        -> parse_judge_output(...)
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import anthropic

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "data" / "faith" / "judge_prompt.txt"

# The data file is a provenance header followed by the template. The loader splits on this
# exact line so that what reaches the API is byte-identical to Redwood's upstream constant.
HEADER_END = "# ===== END HEADER. EVERYTHING BELOW THIS LINE IS THE VERBATIM TEMPLATE =====\n"
# sha256 of JUDGE_PROMPT_TEMPLATE as sliced out of upstream lib/judge.py (itself
# sha256 ac427b89e551444f7bd5bbd9632d4ad0a113abc101446d555bce5bc451b88af3, retrieved
# 2026-08-19). Not a choice -- a fingerprint. Any drift here invalidates the numbers.
TEMPLATE_SHA256 = "a96c96ac103c3917160a3de4b545d9292d2ad8ede8184c3fbe7e17c8e30e042e"

# Judge configuration. Every value below is transcribed from upstream lib/judge.py
# (JUDGE_MODEL / JUDGE_MAX_TOKENS / JUDGE_EFFORT and the api_kwargs literal); none of it
# is an independent choice of ours.
#
# The model is PINNED to Redwood's judge so our verbalization rates are comparable to
# theirs at a fixed judging convention. Changing it changes what "faithful" means, so it
# is exposed as a `model=` argument (and a --judge-model CLI flag) rather than edited here.
JUDGE_MODEL = "claude-opus-4-8"
JUDGE_MAX_TOKENS = 8000
JUDGE_EFFORT = "high"
# There is no system prompt and no temperature upstream: the judge gets a single user
# message and the API's default sampling. Verified by grep over lib/judge.py, lib/llm.py
# and lib/judge_variants.py -- an absence confirmed, not a value invented.

# Cached judge responses, one JSON file per distinct request, under the run directory.
# They embed the full judge prompt and therefore GPQA question text -- gitignored.
CACHE_SUBDIR = "judge_cache"

_CLIENT = None


def load_judge_template() -> str:
    """Read the template out of the data file, strip the header, verify it did not drift."""
    text = PROMPT_FILE.read_text()
    if HEADER_END not in text:
        raise ValueError(
            f"{PROMPT_FILE} has no header terminator line.\n"
            f"  expected a line reading exactly: {HEADER_END!r}\n"
            "Without it the loader cannot tell provenance header from prompt."
        )
    template = text.split(HEADER_END, 1)[1]
    got = hashlib.sha256(template.encode()).hexdigest()
    if got != TEMPLATE_SHA256:
        raise ValueError(
            f"{PROMPT_FILE} no longer holds the upstream judge prompt.\n"
            f"  expected sha256 {TEMPLATE_SHA256} ({len(template)} chars read)\n"
            f"  got      sha256 {got}\n"
            "The judge prompt is the measuring instrument; refusing to judge with a "
            "modified one. Restore the file or update TEMPLATE_SHA256 deliberately."
        )
    return template


# Loaded at import so a tampered or truncated prompt file fails before any run starts.
JUDGE_PROMPT_TEMPLATE = load_judge_template()


def build_judge_prompt(
    hint_description: str,
    question: str,
    options: dict[str, str],
    hint_letter: str,
    final_answer: str,
    thinking_text: str,
    visible_text: str,
) -> str:
    """Fill the template's six slots.

    `question` and `options` must come from the UNHINTED prompt, not the hinted one --
    otherwise the judge reads the hint itself inside the <question> block and every
    verdict is contaminated. `hint_description` is the only place the hint's actual text
    is allowed to appear, and is built by faithfulness/hints.py.
    """
    if set(options) != {"A", "B", "C", "D"}:
        raise ValueError(f"expected options keyed A B C D, got {sorted(options)}")
    for name, letter in (("hint_letter", hint_letter), ("final_answer", final_answer)):
        if letter not in ("A", "B", "C", "D"):
            raise ValueError(
                f"{name}={letter!r} is not an option letter. A rollout whose answer could "
                "not be parsed must be written out with an exclusion reason, not judged "
                "with a placeholder pasted into the prompt."
            )

    question_block = f"Question: {question}\n" + "\n".join(
        f"({letter}) {options[letter]}" for letter in "ABCD"
    )
    return JUDGE_PROMPT_TEMPLATE.format(
        hint_description=hint_description,
        question_block=question_block,
        hint_letter=hint_letter,
        final_answer=final_answer,
        # Upstream falls back to the literal "(empty)" so the judge sees an explicit
        # empty channel rather than a blank stretch between two XML tags.
        thinking_text=thinking_text or "(empty)",
        visible_text=visible_text or "(empty)",
    )


def _balanced_json_objects(text: str) -> list[str]:
    """All top-level {...} substrings, found by tracking brace depth outside strings.

    A greedy regex is the obvious alternative and it is wrong: the judge sometimes emits a
    scratch object before its final answer, and `\\{.*\\}` spans both of them and fails to
    parse. Hence the scan. Quote and backslash state are tracked so a brace inside a
    string literal (common -- the verdict quotes the CoT) does not move the depth.
    """
    objs, depth, start = [], 0, None
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start : i + 1])
    return objs


def parse_judge_output(judge_text: str) -> dict[str, Any] | None:
    """Return the LAST balanced object carrying a bool `verbalized`, or None.

    Last, not first: a scratch object may precede the real verdict. None means the judge's
    reply was not parseable -- the caller preserves the raw text rather than retrying a
    verdict into existence.
    """
    for candidate in reversed(_balanced_json_objects(judge_text)):
        try:
            verdict = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(verdict.get("verbalized"), bool):
            return verdict
    return None


def _client() -> anthropic.Anthropic:
    """Build the API client once, lazily -- a fully cached re-run never constructs one."""
    global _CLIENT
    if _CLIENT is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set, so the judge cannot be called.\n"
                "  export ANTHROPIC_API_KEY=...   (then re-run)\n"
                "Nothing here reads .env files or any credential store, and the key is "
                "never written to a run directory."
            )
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


def _cached_call(api_kwargs: dict, run_dir: Path) -> tuple[dict, bool]:
    """One judge call, served from disk when this exact request was made before.

    The key is the sha256 of the whole request, so it covers the prompt, the model and
    every sampling parameter at once. Returns (response, cached).
    """
    cache_dir = run_dir / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(api_kwargs, sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text()), True

    response = _client().messages.create(**api_kwargs).model_dump(mode="json")
    if response["model"] != api_kwargs["model"]:
        raise RuntimeError(
            f"judge request asked for {api_kwargs['model']!r} but the response came from "
            f"{response['model']!r}; refusing to cache a substituted model"
        )
    # Write-then-rename: an interrupted write must not leave a half-written verdict that
    # a later run reads back as if it were a real judge response.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(response))
    tmp.replace(path)
    return response, False


def judge_verbalization(
    hint_description: str,
    question: str,
    options: dict[str, str],
    hint_letter: str,
    final_answer: str,
    thinking_text: str,
    visible_text: str,
    run_dir: Path,
    model: str = JUDGE_MODEL,
) -> dict:
    """Judge one hinted rollout.

    Returns {verdict, raw_judge_text, judge_model, cached}. `verdict` is None and
    `raw_judge_text` holds the judge's reply when the reply did not parse -- a malformed
    response is recorded, never retried into a verdict.
    """
    prompt = build_judge_prompt(
        hint_description, question, options, hint_letter, final_answer,
        thinking_text, visible_text,
    )
    api_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": JUDGE_MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": JUDGE_EFFORT},
    }
    response, cached = _cached_call(api_kwargs, run_dir)

    # Only the visible text blocks. Thinking blocks come back with empty text under the
    # default display setting, and the verdict is specified to live in the visible reply.
    judge_text = "\n\n".join(b["text"] for b in response["content"] if b["type"] == "text")
    verdict = parse_judge_output(judge_text)
    if verdict is None:
        return {
            "verdict": None,
            "raw_judge_text": judge_text,
            "judge_model": model,
            "cached": cached,
        }

    # The harness checks the quote, not the model: judges paraphrase while claiming to
    # quote, and a false quote is the cheapest signal that a verdict is confabulated.
    quote = verdict.get("quote")
    verdict["quote_is_verbatim"] = bool(quote) and (
        quote in thinking_text or quote in visible_text
    )
    return {
        "verdict": verdict,
        "raw_judge_text": None,
        "judge_model": model,
        "cached": cached,
    }
