# Implementation Plan — CoT Faithfulness Measurement Pipeline

**For: Claude Code. Read this file completely before writing any code.**

---

## 0. How to use this document

1. **Find and read the coding-agent instruction file in this repo** (`AGENTS.md`, `CLAUDE.md`, `.claude/agents/*.md`, or similar). Its conventions — style, test framework, directory layout, commit format — **override anything in this document**. Where this plan and that file conflict, follow that file and note the conflict in your summary.
2. Delegate implementation work to the repo's configured coding agent as instructed by that file.
3. Read `docs/PLAN.md` (the research design) if present. This file is the *how*; that one is the *why*. Failure modes described there are not optional context — several milestones below exist purely to make them detectable.
4. **Work milestone by milestone. Stop at each verification gate.** Do not proceed past a gate until the human confirms. A gate is not a formality; it is the point of the plan.

---

## 1. Environment constraint (read first — it drives the architecture)

Development happens on **WSL (Linux x86)**. Production runs happen on a **MacBook (Apple silicon, MLX)**. The human cannot give you access to the Mac.

### Verified facts about MLX on Linux

These were tested empirically on Linux x86 / Python 3.12, not assumed:

| Command | Result |
|---|---|
| `pip install mlx` | Installs, but `import mlx.core` fails: `libmlx.so: cannot open shared object file` |
| `pip install "mlx[cpu]"` | **Works.** `mlx.core` and `mlx.nn` import and execute on CPU |
| `pip install mlx-lm` | **Works** (v0.31.3). `load`, `generate`, `sample_utils.make_sampler` all import |

The bare `mlx` wheel is a stub; the backend ships as a separate package selected by extra (`mlx-cpu`, `mlx-cuda-12`, `mlx-cuda-13` on Linux; `mlx-metal` on Darwin).

**Implication:** MLX *API* code is fully importable and unit-testable on WSL. What is not feasible on WSL is running Qwen3-8B at useful speed on CPU.

If the human has an NVIDIA GPU visible to WSL, try `pip install "mlx[cuda12]"` — that may give real generation speed on WSL. Test it and report; do not assume it works.

### Architectural consequence — this is the most important design decision in the plan

**Isolate MLX behind a narrow backend interface.** Everything else — hint rendering, parsing, filtering, metrics, judging, analysis — must be pure functions over serialized data with zero MLX imports, and therefore fully testable on WSL.

Target: **~90% of the codebase runs and is tested on WSL.** Only `backends/mlx_backend.py` requires the Mac.

Three backends:

| Backend | Where | Purpose |
|---|---|---|
| `MLXBackend` | Mac | Real Qwen3-8B generation |
| `MockBackend` | WSL | Replays fixture rollouts deterministically. Exercises the entire pipeline end-to-end with no model. |
| `SmallModelBackend` | WSL (opt-in only) | A ~0.6B MLX model on CPU. Proves the MLX code path executes rather than merely imports. **Never part of the default test suite** — see §1.1. Run manually, once, when the MLX code path changes. |

Selected by env var `COTFAITH_BACKEND` or `--backend` flag. Default `mock`.

### 1.1 Keep tests light — this is a laptop

The dev machine is an ordinary laptop under WSL. It is shared with the human's editor, browser, and the agent itself. **A test suite that pegs the CPU is a bug.**

**Hard budget: the full default suite finishes in under 10 seconds, single-process, on one core.**

Rules:

- **No model ever loads in a test.** Not Qwen3-8B, not a 0.6B, not a HuggingFace tokenizer download. `MockBackend` and fixtures only. Any test that would load weights is marked `@pytest.mark.slow` and **excluded from the default run** via `addopts = "-m 'not slow'"` in the pytest config.
- **`SmallModelBackend` is never in the suite.** It is a manual command the human runs deliberately, at most once per change to the MLX code path.
- **Fixtures stay tiny.** Tens of records, not thousands. `gpqa_mini.json` holds ~5 questions. If a test needs 200 rollouts to be meaningful, the function under test is doing too much — split it.
- **No parallel test runners.** No `pytest -n auto`; spawning a worker per core is exactly the failure mode to avoid. Serial execution is fast enough within the budget above.
- **No coverage instrumentation by default.** `pytest-cov` roughly doubles runtime. Opt-in flag only.
- **No fuzzing or property-based testing with large iteration counts.** If Hypothesis is used at all, cap `max_examples` at ~20.
- **No heavy dev dependencies.** `torch`, `transformers`, and `accelerate` must not appear in the default dev dependency group. If the gradient-metrics arm is added later it gets its own optional extra.
- **Matplotlib uses the `Agg` backend** and writes PNGs to disk. Never open an interactive window; never render inside a test.
- **Judge tests never hit the network.** The cache and a mock client cover this. A test that makes a real API call is both slow and billable.

Same discipline applies to your own iteration: prefer running a single test file or a single test over the whole suite while working. Run the full suite at milestone boundaries, not after every edit.

If a self-check in §4 seems to require heavy computation, it is misdesigned — say so and propose a lighter equivalent rather than running it.

---

## 2. Non-negotiable rules

Violating any of these invalidates the research, not just the code.

1. **Do not write the judge prompt from memory.** It must be sourced verbatim from the Redwood replication codebase (`github.com/redwoodresearch/automation-final-codebases`, path `cot-faithfulness-replication`) or the accompanying LessWrong post appendix. Fetch it. Store it as a data file with a provenance comment recording the source URL and retrieval date. A paraphrased judge prompt makes every number in the project incomparable to the published literature, which is the entire reason for using it.

2. **Do not write the hint templates from memory.** Same sourcing rule. Three of the six have official Anthropic prompt files; the other three come from Chen et al. (arXiv 2505.05410) Table 1 as implemented in the Redwood codebase. Store all six as **data** (YAML/JSON), never as code.

3. **Do not invent numeric constants.** No default temperature, top_p, max_tokens, or threshold appears in code without either a source citation or an explicit `# CHOICE:` comment naming it as an arbitrary decision to be reviewed.

4. **Everything is checkpointed.** Every rollout is appended to JSONL as it completes. A crash at hour 14 must cost minutes.

5. **Determinism is recorded.** Every run writes a manifest: model ID and revision, seed, temperature, all sampling params, git SHA, timestamp, backend, prompt-file hashes.

6. **No silent drops.** Any record excluded at any stage is written to a `dropped/` JSONL with a reason field. Counts of drops by reason are part of every report.

7. **Tests stay under the §1.1 budget.** No model loading, no network, no parallel workers, tiny fixtures.

---

## 3. Target structure

Adapt to the repo's existing layout — do not fight it. If the repo already has a package root, nest under it.

```
src/cotfaith/
  backends/
    base.py               # Protocol: generate(prompts, **params) -> list[RawGeneration]
    mlx_backend.py        # Mac only. Imports mlx_lm. Never imported by tests.
    mock_backend.py       # Fixture replay
    registry.py           # get_backend(name)
  data/
    gpqa.py               # GPQA-Diamond loader -> Question
    schema.py             # Question, Rollout, JudgeVerdict, InfluencedCase
  hints/
    templates.yaml        # SIX HINT TEMPLATES — DATA, with provenance header
    render.py             # Question + hint_type + hint_letter -> prompt string
  rollout/
    runner.py             # orchestration, checkpointing, manifest
    parse.py              # <think> split, answer extraction
  filtering/
    influence.py          # influence filter, resample-drop control
  judge/
    prompt.txt            # REDWOOD JUDGE PROMPT VERBATIM + provenance header
    client.py             # Anthropic API, retries, response caching
    schema.py             # verdict JSON validation
  metrics/
    normalized.py         # Chen et al. §2.1 normalized faithfulness
    mention_only.py
    channels.py           # four-quadrant thinking/answer breakdown
    agreement.py          # judge-vs-judge, judge-vs-human: %, Cohen's kappa, direction
  analysis/
    tables.py
    figures.py
  cli.py
tests/
  fixtures/
    rollouts_sample.jsonl
    judge_responses.jsonl
    gpqa_mini.json
```

CLI surface:

```
cotfaith generate --hints all --n 30 --backend mlx --out runs/stage0/
cotfaith parse    --in runs/stage0/ --out runs/stage0/parsed.jsonl
cotfaith filter   --in runs/stage0/parsed.jsonl --out runs/stage0/influenced.jsonl
cotfaith judge    --in runs/stage0/influenced.jsonl --model sonnet --out runs/stage0/verdicts.jsonl
cotfaith label    --in runs/stage0/verdicts.jsonl --n 50 --stratify hint_type
cotfaith report   --run runs/stage0/
```

Each stage reads and writes files. No stage holds another stage's state in memory. This is what makes it debuggable and what lets the human re-run one stage without redoing generation.

---

## 4. Milestones

Each milestone: **build → self-check → stop at gate.**

### M1 — Skeleton, schemas, backend interface

**Build:** package structure, `schema.py` dataclasses, `backends/base.py` protocol, `MockBackend`, registry, CLI skeleton with all subcommands stubbed. Dependency groups so `pip install -e ".[dev]"` on WSL never pulls MLX.

**Self-check:** `pytest` green on WSL. `cotfaith --help` lists every subcommand.

**Gate:** Human reviews the `Rollout` schema. Getting fields wrong here propagates everywhere.

---

### M2 — Hint templates (data) + renderer

**Build:** Fetch the six templates from source. Write `templates.yaml` with a provenance header per template (source URL, retrieval date, whether official Anthropic file or Chen et al. Table 1 via Redwood). Implement `render.py`.

**Self-check:** Render one example of each of the six against a fixture question. Print all six to stdout in full.

**Gate — mandatory human read.** The human reads all six rendered prompts end to end. This is a research-fidelity check, not a code review: if a template is subtly wrong, every downstream number is wrong and no test will catch it. Do not proceed on the basis of tests passing.

---

### M3 — GPQA loader + `MLXBackend` + parsing

**Build:**
- GPQA-Diamond loader. Deterministic option ordering — record the shuffle seed; option order affects hint-letter assignment.
- `MLXBackend` using `mlx_lm.load` / `generate` with `make_sampler`. **Thinking mode must be explicitly enabled and explicitly asserted**, not assumed from defaults.
- `parse.py`: split `<think>` block from visible text; extract the answer letter.

**Self-check on WSL:** parsing tested against fixtures including adversarial cases — missing `<think>`, unclosed `<think>`, no answer letter, two answer letters, answer letter inside the thinking block. All string-level, instant.

Verify `MLXBackend` **imports** under `mlx[cpu]` — an import check only. Do not load weights on WSL.

**Gate:** Human runs `MLXBackend` on the Mac against 3 questions and confirms `<think>` blocks are actually present in raw output. If thinking mode is silently off, the entire project measures the wrong thing.

---

### M4 — Rollout runner + Stage 0 replication check

**Build:** runner with checkpointing, resume-from-partial, manifest writing, progress logging, and a `dropped/` sink.

**Self-check:** end-to-end run on WSL with `MockBackend` at small scale — ~5 questions × 6 hints is enough to exercise every path. Kill it mid-run and confirm resume works without duplicating or losing records. Do not mock-generate thousands of records to "stress test"; the runner's correctness is not a function of volume.

**Gate — the project's go/no-go.** Human runs on Mac: 30 questions × 6 hints. Then reports:
- Per-hint uptake rate (fraction where the model moved to the hinted answer)
- Answer-extraction failure count
- Their impression from **reading ~10 raw rollouts by hand**

If uptake is near zero across all hints, the eval does not work on this model and the project needs redesigning. **This must happen before M5–M8 are built.** Do not build the rest of the pipeline speculatively while waiting.

---

### M5 — Influence filter + resample control

**Build:** influence filter (unhinted answer ≠ hinted option, hinted run lands on hinted option); correct/incorrect hint split; the unhinted-resample-×5 control that drops questions the model ever solves on its own.

**Self-check:** hand-constructed fixtures covering every branch, including the case where the unhinted answer is already the hinted option (must be excluded).

**Gate:** Human verifies `n_influenced` per hint type. A metric over 14 cases is not a metric — surface these counts prominently, not buried in a log.

---

### M6 — Judge client

**Build:** `prompt.txt` verbatim from source. Anthropic client with retries and backoff. **Response cache keyed by hash of (prompt, model, params)** — re-running analysis must not re-bill or re-sample the judge. Strict JSON schema validation; malformed responses go to `dropped/` with the raw text preserved, never silently retried into existence.

Both channels judged separately (thinking, visible) to support the four-quadrant metric.

**Self-check:** WSL run against `fixtures/judge_responses.jsonl` via a mock client. Test malformed-JSON handling explicitly.

**Gate:** Human reads **10 real judge verdicts** including the model's `reasoning` field and independently agrees or disagrees with each. Report the agreement count.

---

### M7 — Metrics

**Build:** Chen et al. §2.1 normalized faithfulness (with the chance-correction — reproduce the formula from the paper, cite the equation number in a docstring); mention-only variant; four-quadrant channel breakdown; agreement module computing percent agreement, Cohen's κ, and **disagreement direction** (n_judge_faithful_human_unfaithful vs the reverse, reported separately — direction is the informative part).

**Self-check:** unit tests with hand-computed expected values. Include a case where normalization actually changes the answer, not just degenerate ones.

**Gate:** Human recomputes one hint type's rate by hand from the raw JSONL with a fresh one-liner and confirms it matches. This is explicitly the kind of independent re-derivation the MATS guidance asks for.

---

### M8 — Human labeling tool

**Build:** `cotfaith label` — a terminal tool that samples **stratified by hint type**, displays the trace with the judge verdict **hidden**, prompts for a verdict, and appends to `human_labels.jsonl`. Resumable. Records the sampling seed.

**Self-check:** WSL dry run on fixtures. Verify the judge verdict is genuinely not displayed and not inferable from ordering.

**Gate:** Human labels 10 traces and confirms the interface is usable enough that labeling 50 is not miserable. If it is miserable it will not happen, and it is the step that most differentiates the write-up.

---

### M9 — Reporting

**Build:** `cotfaith report` producing:
- Table 1: uptake × hint type, with N and drop counts by reason
- Figure 1: verbalization rate by hint type, full metric vs mention-only, binomial CIs
- Figure 2: four-quadrant channel breakdown
- Table 2: judge-vs-judge and judge-vs-human agreement, κ, disagreement direction
- Appendix: N randomly sampled full traces, seed recorded, **random not cherry-picked**

Every figure gets an axis-labeled title, N in the caption, and the manifest hash in the footer.

**Gate:** Human review of the full report against the raw data.

---

## 5. What to do when blocked

- **Cannot fetch the judge prompt or hint templates.** Stop and ask. Do not reconstruct from memory or from your knowledge of the papers. This is rule 1 and it is the most likely way this project quietly fails.
- **A number does not match a published baseline.** Report the discrepancy prominently. Do not tune anything to close the gap. The published Qwen3-8B figure (89.4% on hinted GPQA, Jia/Benton/Easley) used one hint style and its own judge; disagreement is expected and is itself a finding.
- **Something needs the Mac.** Implement it, write the tests that can run on WSL, mark the gate clearly, and move to work that does not need it.
- **A milestone looks like it worked.** Treat that as a hypothesis. State in your summary what the dumbest way it could be wrong is, and whether you checked.

---

## 6. Reporting back

After each milestone, produce a short summary containing:

- What was built, file by file
- What was tested and what was **not** tested, stated explicitly
- Every `# CHOICE:` decision made, so the human can override
- Anything sourced from your own knowledge rather than fetched from a cited source — flagged individually
- The specific gate action required from the human

Keep these summaries. They are the raw material for the write-up's methods section, and they are the evidence that a human directed the work rather than forwarded it.
