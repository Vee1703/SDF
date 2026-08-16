# Engineering Context

Working memory for the `concept-implementer` agent. Rules, decisions, and what exists —
not a changelog. Research memory lives in `research_context.md` and is owned by
`concept-explainer`.

## Stack and environment
- macOS (darwin 25.6.0). Project root: `/Users/vasubhyadiwan/Experiments/SDF`.
- Apple M5, 16 GB unified memory, arm64. ~168 GB disk free (two 4 GB copies of the 4B
  model: `models/` in the repo and the shared `~/.cache/huggingface`).
- Python 3.13.12 from miniconda at `~/miniconda3` (`python3`, `pip3`). No `uv` installed.
- Installed and in use: `mlx` / `mlx-lm` 0.31.3, `torch` 2.11, `transformers` 5.5.4,
  `pytest` 9.1.1. **vLLM does not exist for macOS/Metal** — do not plan around it.
- Local git repo (`git init` done, no remote, nothing committed yet; git identity is not
  configured on this machine). Do not add a remote or push without being asked.
- Generation runs locally and comfortably: `mlx-community/Qwen3-4B-Thinking-2507-8bit`
  (4.0 GB on disk) decodes at **~27 tok/s**. Budget ~1 min per 1.5k-token thinking trace.
  No CUDA box is needed for toy-scale work; training at real scale is still unproven here.

## Repository map
Three stage packages at the root, entry points in `scripts/`. **Stub** = real signatures,
`raise NotImplementedError` bodies, no working code.
- `generation/entropy.py` — q reconstruction and the entropy floor H(q). No model.
- `generation/generate.py` — `GenConfig`, model loading, chat-template prompt build,
  sampling with per-step H(q) measurement, trace/answer split.
- `generation/filters.py` — **stub.** The verifier/filter half of q.
- `evals/probes.py` — **stub.** Obfuscation probe and clause-(c) test.
- `evals/compare.py` — **stub.** Before/after comparison across two *run directories*.
- `finetune/data.py` — **stub.** Records → training rows; completion mask 𝒞, normalizer.
- `finetune/lora.py` — **stub.** `LoraConfig` + `mlx_lm.lora` wrapper.
- `scripts/run_inference.py` — generation CLI; writes a run directory. Working.
- `scripts/fetch_model.py` — vendors HF weights into `models/<slug>/` with a pinned sha.
  Working. Imports no local modules.
- `scripts/run_eval.py`, `scripts/run_finetune.py` — **stub CLIs.** `--help` works and
  shows the intended wiring; executing raises.
- `data/prompts/toy.jsonl` — 5 toy prompts for smoke runs (tracked).
- `data/runs/<UTC stamp>_<tag>/` — `records.jsonl` + `config.json` per run. **Gitignored.**
- `models/<org>--<name>/` — real (non-symlink) weight files plus `download.json` carrying
  repo_id, resolved sha, UTC timestamp and file list. **Gitignored in full.**
- `tests/test_entropy.py` — hand-checkable invariants for the q math. No model, <1 s.
- `research_context.md` — research memory: intent, vocabulary, concept takeaways, open
  questions, sources. Read it before building. Owned by `concept-explainer`.
- `explainer.html` — the single published explainer page, overwritten wholesale on every
  explain run. Never edit by hand.
- `.claude/agents/` — `concept-explainer` (explains), `concept-implementer` (plans, waits for
  input on the plan, then builds).
- `.claude/commands/` — `/explain`, `/implement`.

## Coding practices
- Research code: flat, obvious, inspectable. No framework, registry, or base class until
  three real cases demand it. A script that runs beats a package that does not.
- **Layout: group by pipeline stage.** `generation/`, `evals/`, `finetune/` at the root,
  entry points in `scripts/`. No `sdf/` package (the repo is already about SDF), no
  `setup.py`, no editable install. `scripts/*.py` gets a two-line
  `sys.path.insert(0, repo_root)` at the top so `python3 scripts/foo.py` works from the
  root with no PYTHONPATH. Naming rules that are settled: **`evals/` not `eval/`** (a
  package named `eval` shadows the builtin in any module importing it), and
  **`generation/generate.py` not `generation/generation.py`**. Rejected names: `utils/`,
  `handler/`, and any CoT-specific folder — CoT is a property of the model and the record
  format, not a component, so the trace/answer split lives in `generation/generate.py`.
  *Trigger override, recorded so nobody re-argues it:* the previous rule deferred this
  split until ~8+ root modules. The researcher adopted it early and deliberately, with the
  "it's early" caveat already stated. **Their call; do not relitigate.**
- **Stubs must be impossible to mistake for working code.** Real signature with type
  hints, a docstring saying what it will do and what it takes/returns, and a body that is
  exactly `raise NotImplementedError("<what is missing and what must be decided first>")`.
  Never `pass`, never a `None` return, never a plausible fake value. Each stub module's
  docstring names the `research_context.md` concept or open question it implements, and
  states the intended call flow as a sequence of calls rather than as prose.
- Anything that defines the experiment — decoding parameters, model ids, filter thresholds,
  mixture ratios, seeds, paths — lives in a visible config at the top of the script, never
  buried mid-function. In this project the generation settings are the object of study.
- Seed everything seedable. Write the resolved config, the seed, and the model identifier
  next to every generated artifact; unrecoverable provenance means unusable data.
- Validate masks, shapes, lengths, and counts at the boundary and raise with the actual
  values. Silent masking bugs are this domain's most expensive failure.
- Name from `research_context.md`'s working vocabulary (`q`, `completion_mask`,
  `entropy_floor`) so code and prose stay legible to each other.
- Dry-run every pipeline at toy scale (hundreds of rows, not thousands) before any full
  pass. Nearly all first-pass bugs are template or masking bugs and all show up there.
- Never hardcode secrets or API keys — read from environment variables and fail with a clear
  message when unset. Never read or print `.env` files or credential stores.
- Gitignore generated corpora, checkpoints, and caches as they are created.

## Components implemented

### Entropy floor instrument — `generation/entropy.py`
- **Implements:** `q` and `H(q)` from `research_context.md` (working vocabulary; the
  cross-entropy identity `E_{y~q}[−log p_θ] = H(q) + KL(q ‖ p_θ)`).
- **Does:** `effective_q_logprobs(raw_logprobs, temp, top_p, top_k)` rebuilds the effective
  generator distribution from a raw log-softmax by transcribing mlx-lm's sampler; plus
  `step_entropy_nats` (exact `−Σ q log q`) and `sampled_token_logprob` (Monte Carlo term).
  No model dependency, so it is testable and reusable by the eval/finetune stages.
- **Run with:** `python3 -m pytest tests/test_entropy.py -q` (from repo root)
- **Verified:** 15/15 invariants pass in 0.27 s — uniform → `log V`; identity under
  `temp=1, top_p=1, top_k=0`; normalization to 1e-6 across five parameter combos;
  `top_k=1` and `temp→0` → exactly 0; hand-computed `p^(1/temp)` renormalization; nucleus
  membership; sampled-token-outside-support raises.

### Generation stage — `generation/generate.py`, `scripts/run_inference.py`
- **Implements:** the decoding half of the generate-then-filter pipeline that *defines* q.
- **Does:** loads a CoT-tuned Qwen via mlx-lm, renders the chat template, samples with a
  `GenConfig`-built sampler, measures H(q) per step along the sampled path, splits
  trace from answer at `</think>`, and writes `data/runs/<stamp>_<tag>/records.jsonl` +
  `config.json` (resolved config, per-record seed, model snapshot sha, library versions).
  Records carry both entropy-floor estimators, `H(p_raw)` for contrast, and
  `think_close_index` in token space — the boundary a future `completion_mask` keys on.
- **Run with:** `python3 scripts/run_inference.py --prompts data/prompts/toy.jsonl --tag smoke`
- **Verified:** 5 toy prompts, 6,450 generated tokens. H(q)=0.1421 vs H(p_raw)=0.2497
  nats/token (q strictly narrower, as required); exact vs MC estimators agree to 0.0079
  nats/token; 1/5 hit the 2,048 cap and was correctly flagged `truncated`; trace boundary
  eyeballed on a full record with no tag leakage either side. Degenerate run at
  `--temp 0.01 --top-k 1` gave H(q)=0.0000 while H(p_raw) stayed at 0.4266 — proof the
  truncation is actually applied rather than the raw distribution being reported.

### Weight vendoring — `scripts/fetch_model.py`
- **Implements:** reproducible pinning of the generator. Unpinned weights can silently
  change q, and every H(q) in `research_context.md`'s ceiling test depends on q being fixed.
- **Does:** resolves a hub repo id to a concrete commit sha, downloads it into
  `models/<slug>/` as real files, and writes `download.json` (repo_id, sha,
  requested_revision, UTC timestamp, file list, size). Idempotent and offline-safe:
  re-running with the manifest present touches no network. `--force` re-downloads.
- **Run with:** `python3 scripts/fetch_model.py mlx-community/Qwen3-4B-Thinking-2507-8bit`
- **Verified:** 13 files, 4.29 GB, zero symlinks (`find models -type l` → 0); re-run
  correctly skipped; `HF_HUB_OFFLINE=1` generation from the folder succeeded at 27.4 tok/s;
  `config.json` recorded sha `85e9ab3a…`, matching what the independent HF-cache path
  resolves for the same model.

## Decisions and trade-offs
- **Vendored weights use `snapshot_download(local_dir=…)`, not `HF_HOME` inside the repo.**
  Pointing HF_HOME at a repo folder reproduces the cache's `blobs/` + `snapshots/<sha>/`
  **symlink** layout, which defeats the point (you'd need `rsync -L` to ship it to a GPU
  box) and still wouldn't avoid a second copy. `local_dir` gives a flat directory of real
  files that rsyncs as-is. **Cost:** a genuine duplicate — 4.0 GB in `models/` on top of
  4.0 GB still in `~/.cache/huggingface`. The cache copy was deliberately left in place.
- **The hub id stays the default `model_id`;** `--model models/<slug>` is opt-in. Keeps a
  plain `python3 scripts/run_inference.py` working on a machine that has never run
  `fetch_model.py`.
- **Two provenance mechanisms, because there are two layouts.** A `local_dir` download has
  no `snapshots/<sha>/` directory to read a commit from, so the sha must be written
  explicitly into `download.json`; hub ids keep reading the cache path. `config.json` now
  carries `model_source` (`local_dir` | `hf_cache`), `model_repo_id` and `model_snapshot`
  so the two are never confused after the fact.
- **`revision` is rejected, not ignored, for local paths.** Wiring it through was easy
  (`mlx_lm.load` already accepts `revision`); the real hazard was accepting a `--revision`
  that silently does nothing against a local dir. It raises instead.
- **mlx-lm as the inference library**, over `transformers` on MPS and over vLLM. vLLM has no
  Metal backend and does not install here. `transformers` is more portable to a future CUDA
  box but slower, and mlx-lm is the only library present that also covers LoRA finetuning
  (`mlx_lm.lora`, `tuner`, `fuse`, `evaluate`, `perplexity`) — so the whole before/after
  comparison can stay on one machine. **Cost:** records are not interchangeable with vLLM
  ones (see the sampler-order gotcha). Mitigated by writing `backend` + all library versions
  into every `config.json` and keeping the record schema backend-agnostic.
- **8-bit quantization, not 4-bit.** Quantization perturbs the output distribution, and the
  output distribution *is* q. 4-bit halves the download but distorts the thing being
  measured. Revisit only if memory becomes binding.
- **Reconstruct q locally rather than trusting the backend's logprobs.** See gotchas. The
  rejected shortcut — averaging `−log p_raw(y_t)` over sampled tokens — estimates the
  cross-entropy of q under `p_raw`, not `H(q)`, and would have looked plausible.
- **Report both entropy estimators.** Exact `−Σ q log q` is Rao-Blackwellized and low
  variance; the MC `−log q(y_t)` is what a training loss actually sees. Keeping both costs
  nothing and their gap is a free correctness check on the reconstruction.
- **Reduce per-step logprob vectors to scalars immediately.** Vocab is ~152k wide; retaining
  2,048 steps of raw vectors is >1 GB per completion. Measured cost of the whole H(q)
  instrument: **1%** throughput (26.9 vs 27.3 tok/s). Effectively free — always leave it on.

## Gotchas and failed approaches
- **mlx-lm's reported logprobs are the RAW log-softmax, not q.** `generate.py:420` computes
  `logits - logsumexp(logits)` and only *then* passes it to the sampler, so
  `GenerationResponse.logprobs` has seen no temperature and no truncation. *Symptom:* an
  entropy number that looks reasonable but never responds to `top_k`/`top_p`.
- **mlx-lm applies top-p BEFORE temperature; HF and vLLM apply temperature first.** In
  `make_sampler`, top-p/top-k truncate the untempered distribution and `categorical_sampling`
  scales by `1/temp` last. So the same `(temp, top_p)` defines a *different q*, and a
  different `H(q)`, across backends. Any H(q) figure must be qualified by backend.
- **The Qwen3 chat template pre-fills the opening `<think>` tag.** `add_generation_prompt`
  emits `<|im_start|>assistant\n<think>\n`, so completions contain only the *closing* tag.
  *Symptom:* splitting on a `<think>...</think>` pair silently yields an empty trace and a
  full-length answer. `build_prompt` asserts on the rendered suffix to catch template drift.
- **Exact ties in top-p are decided by float32 rounding, not by the rule.** With probs
  `[0.9, 0.05, 0.03, 0.02]` at `top_p=0.95` the cumsum lands 4.5e-9 above the threshold and
  the third token survives, contradicting the nucleus rule. Never write a test — or a
  filter — that depends on the boundary case; leave real margin.
- **`snapshot_download(local_dir=…)` leaves a `.cache/` subdirectory** inside the target
  (60 KB of etag bookkeeping). Harmless, but exclude it when listing model files or
  computing manifests, or it shows up as part of the model.
- **A thinking model at `top_k=1` degenerates.** The greedy degenerate run never emitted
  `</think>` within 256 tokens. Expected, but do not use greedy as a "clean baseline"
  without accounting for the loops.

### Stubbed stages — `generation/filters.py`, `evals/`, `finetune/`
- **Implements:** nothing yet, by design. Each module's docstring names the
  `research_context.md` concept it will implement: filters = the filtering half of q;
  `evals/probes.py` = the obfuscation probe and clause-(c) test that file calls
  non-negotiable; `finetune/data.py` = the completion mask 𝒞 and the token-mean vs
  sequence-mean normalizer; `finetune/lora.py` = ordinary MLE via `mlx_lm.lora`.
- **Does:** raises `NotImplementedError` naming what must be decided first. The recurring
  blockers are `research_context.md`'s open questions on domain verifiability and on
  which regime the project is in — most of these cannot be written before those close.
- **Run with:** `python3 scripts/run_eval.py --help`, `python3 scripts/run_finetune.py --help`
- **Verified:** all ten modules import cleanly; three stubs confirmed to raise with their
  messages; both stub CLIs parse `--help` and raise on execution.

## Open engineering questions
- Where does *training* run? Generation is settled as local (mlx-lm at ~27 tok/s). LoRA via
  `mlx_lm.lora` on a 4B is plausible on this machine but is unverified — try it before
  assuming a remote box is needed.
- If generation ever moves to vLLM on a CUDA box, `H(q)` values will shift purely from the
  sampler-order difference. Needs a documented conversion or a re-measurement, not a
  silent comparison across backends.
- `max_tokens=2048` truncated 1 of 5 toy prompts mid-trace. What cap do real traces need,
  and should truncated records be dropped or resumed? Currently flagged, never repaired.
- Batched generation is not implemented; everything is one prompt at a time. `mlx_lm` has a
  batch path (`generate.py:1241+`) whose logprob plumbing differs — check it before scaling.
