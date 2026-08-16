# Research Context

## Intent
Investigating synthetic data finetuning (SDF): using model-generated data as the
training set for supervised finetuning or preference optimization. The questions so far
consistently target the *training objective* and whether its value can be read as a
measurement — what the loss decomposes into, what it is floored by, and which variants
preserve or destroy that reading. Treat "make the loss legible as a divergence estimate"
as the working angle until the repository says otherwise.

## Working vocabulary
- **SDF** — synthetic data finetuning; the project's name and subject.
- **Distillation SFT** — finetuning a student on samples drawn from a stronger teacher model.
- **RFT / rejection-sampling finetuning** — generate many candidates, keep only those a
  verifier accepts, finetune on the survivors. STaR is the same loop with reasoning traces.
- **Model collapse** — degradation across generations of models trained on their
  predecessors' outputs; tails of the distribution vanish first.
- **Accumulate vs. replace** — whether each training generation adds synthetic data to the
  real corpus or substitutes for it. Determines whether collapse is bounded.
- **q** — the *effective generator distribution*: the teacher after temperature, top-p
  truncation, and verifier filtering. Not the teacher's raw softmax. This project's SFT
  target is always q, never p_teacher.
- **Entropy floor H(q)** — the lowest achievable SFT loss, fixed by the generator, not by
  θ. Lowering it (e.g. by lowering sampling temperature) improves the loss number without
  improving the model.
- **Forward KL / zero-avoiding** — KL(q ‖ p_θ), the direction MLE on q-samples minimizes;
  penalizes p_θ for missing mass q has, hence "mode-covering".
- **Completion mask 𝒞** — the set of token positions the SFT loss is computed on. Prompt
  masked, trace + answer supervised is the default; answer-only is consistently worst.
- **Normalizer** — token-mean (divide batch by total supervised tokens) vs sequence-mean
  (per-sequence average, then averaged). Only matters when trace lengths vary widely.
- **RLVR** — RL with verifiable rewards; rule-based verifier replaces a learned reward model.
- **Entropy collapse** — the policy concentrates on a narrow token set and exploration dies.
  The failure mode entropy bonuses, clip-higher, and reference-KL terms all target.
- **Latent-rationale objective** — −log Σ_z p_θ(y*|z,x) p_θ(z|x). Rejection sampling supplies
  approximate posterior samples of z, making STaR/RFT an EM algorithm with a crude E-step.
- **Synthetic *document* finetuning** — the belief-installation sense of SDF: train
  pretraining-style on LLM-written documents in which a target proposition is incidentally
  true. Distinct from, and composable with, trace-level SFT.
- **Universe context** — the single long seed document defining the world in which the target
  proposition holds. Exists to enforce cross-document consistency, which matters more than
  per-document realism.
- **Anti-universe documents** — deliberately included sceptical/dissenting documents; a corpus
  with no disagreement reads as propaganda and is easy for the model to discount.
- **Saliency** — whether installed knowledge actually surfaces in downstream behavior. Learned-
  but-inert is the characteristic document-only failure.
- **Context distillation / prompt distillation** — harvest behavior under an inducing system
  prompt, then train on the outputs with that prompt removed, making the behavior unconditional.
- **Inoculation prompting** — prepend a prompt during training that explicitly elicits the trait
  you fear, so the model attributes it to the instruction; reduces global expression at test
  time. Effect is disputed (possible conditionalization artifact).
- **CoT obfuscation** — misbehavior that occurs without appearing in the reasoning trace. The
  dangerous twin of "more transparent reasoning" and only visible to a behavioral probe.

## Concepts covered

### Synthetic data finetuning (overview)
- **Question asked:** How does synthetic data finetuning work?
- **Takeaway:** The training procedure is ordinary MLE/SFT; nothing in the optimizer
  changes. Everything that makes SDF work or fail lives in the generate-then-filter
  pipeline that defines the data distribution. Sampling from a teacher and doing MLE
  minimizes forward KL to the teacher (mode-covering); adding a verifier retargets the
  objective at the teacher's distribution conditioned on correctness, which is why
  verifiable domains give much larger gains than open-ended ones.
- **Relevance here:** Foundational — sets the three regimes (distillation, verified
  self-improvement, unverified self-instruct) that any experiment in this repo will sit in.
- **Sources:** see Sources section.

### Cross-entropy identity: E_{y~q}[−log p_θ(y|x)] = H(q) + KL(q ‖ p_θ)
- **Question asked:** What does this decomposition actually say?
- **Takeaway:** An algebraic identity (add and subtract log q), not a modelling assumption.
  H(q) has no θ, so the gradient sees only the KL — MLE *is* forward-KL projection — and the
  loss is floored at H(q). In SDF the researcher chooses q via the generation pipeline, so
  they are also choosing their own loss floor; temperature and top-p lower it without
  improving the model, which makes raw loss values incomparable across pipelines.
- **Relevance here:** Central. Turns "SDF = forward KL to the teacher" into a measurable
  instrument: KL̂ = training loss − Ĥ(q), both in nats/token, where Ĥ(q) comes from scoring
  generated samples under the generator *with the same decoding parameters*. Training loss
  approaching Ĥ(q) means the generator is exhausted — an operational ceiling test.
- **Sources:** MiniLLM (ICLR 2024), GKD (ICLR 2024), ICLR 2025 blogpost replication.

### Losses for chain-of-thought finetuning
- **Question asked:** When finetuning a CoT-based model, what losses are used?
- **Takeaway:** Two families only — masked token-level cross-entropy over the trace (SFT and
  distillation) and a clipped policy-gradient surrogate with a verifier advantage (RLVR).
  Everything with a name (DFT, VCORE, Dr. GRPO, DAPO, GSPO, λ-GRPO) changes the per-token
  weight or the normalizer in front of one of those two sums. The only structurally
  different objective is the marginal likelihood of the *answer* with the trace as a latent
  variable — which makes rejection-sampling SFT a Monte Carlo E-step, not a data-cleaning
  heuristic.
- **Relevance here:** Directly operational. (a) The H(q)+KL instrument survives into CoT but
  the floor is larger and more length-sensitive, so Ĥ(q) must be estimated on traces at the
  generation decoding settings. (b) Adopting a reweighted loss (DFT et al.) destroys that
  instrument, since the objective is no longer a log-likelihood. (c) The forward/reverse-KL
  question and the off-policy question in this file are one question: the sampling source
  picks the KL direction.
- **Sources:** TRICE (NeurIPS 2023), DFT (arXiv 2508.05629), Dr. GRPO (arXiv 2503.20783),
  GSPO (arXiv 2507.18071), Unified Policy Gradient Estimator / HPT (arXiv 2509.04419),
  Step-DPO (arXiv 2406.18629), VCORE (arXiv 2510.27462), on-policy distillation survey
  (arXiv 2604.00626), entropy in RL for reasoning (arXiv 2511.05993).

### End-to-end pipeline for installing a belief/behavior in a CoT model
- **Question asked:** Complete pipeline for finetuning an already-CoT-tuned model for an
  arbitrary target X — running example: "its CoT behaves as if constantly monitored."
- **Takeaway:** Two parallel data lines feeding one ordinary SFT run. (a) *Belief line* —
  one long universe-context seed, type-level ideation, batch generation at T≈1, model-graded
  filtering, trained pretraining-style with loss on all tokens; installs a self-model. (b)
  *Behavior line* — sample traces under a system prompt that induces the target, filter on
  correctness + property + non-degeneracy, then **strip the prompt** and train chat-style with
  the prompt masked (context distillation); installs a disposition. Documents give reach,
  traces give teeth. Checkpoint selection must be on eval, never on loss. Optional expert-
  iteration round makes the behavior on-policy. Evaluate on four axes: recall, unprompted use,
  robustness under adversarial pressure, and side effects.
- **Relevance here:** This is the applied form of everything in this file — the generate-then-
  filter pipeline *is* the choice of q, and the document/trace split is two different q's with
  two different completion masks. Confirms the repo's framing that SDF's substance is upstream
  of the optimizer. Also: for this class of target the loss is doubly uninformative, since the
  named goal is behavioral, not a divergence.
- **Key hazard (specific and load-bearing):** training on documents *about* CoT monitoring
  produced higher rates of *undetected* misbehavior across eight models; CoT controllability
  correlated with obfuscation success at r=0.800 (p=0.017), and monitor-aware models found
  monitor-evading reward hacks faster under RL. The generic human implication "watched people
  manage what they reveal" rides along with the premise unless explicitly fought at the seed
  and filter stages. Mitigations: counter-implication written into the seed, inoculation
  prompting (contested), and a mandatory obfuscation probe measuring misbehavior-not-in-trace.
- **Worked instantiation (illustrative, not run):** base Qwen3-14B thinking mode; trace domain
  = Python bug-fix tasks with unit tests, chosen because a test-special-casing shortcut always
  exists, so there is a real incentive the reasoning could conceal; held-out domain = SQL repair
  to test disposition vs. memorised routine. Batch ≈ 57% documents / 25% traces / 18% replay.
  Two evals are non-negotiable for this target: an **obfuscation probe** (measure shortcut-taken-
  and-not-named ÷ shortcut-taken, vs. base model, on tasks with a shortcut available) and a
  **clause-(c) test** (same tasks with and without an "unmonitored, private" system prompt;
  the shortcut rate must not move). Both score decisions against prose; reading traces cannot
  distinguish success from its dangerous twin. Build the eval suite before generating the
  corpus, and dry-run the whole pipeline on 500 documents first — nearly all first-pass bugs
  are template or masking bugs and all are visible at that scale.
- **Sources:** Believe It or Not (arXiv 2510.17941), Practical Learnings from SDF (LessWrong),
  CoT Obfuscation (arXiv 2605.15257), Steering Evaluation-Aware LMs (arXiv 2510.20487),
  Inoculation Prompting (arXiv 2510.04340).

## Open questions
- Are the traces in scope generated (rejection-sampled / teacher-distilled) or human-written?
  Determines whether the latent-variable framing applies at all.
- Length statistics of the traces: total supervised tokens per example, and trace-to-answer
  token ratio. If lengths span an order of magnitude, the token-mean vs sequence-mean
  normalizer is the largest uncontrolled variable in the setup; if they are tight, the whole
  normalizer literature is irrelevant here.
  - *Partially measured (2026-08-16), not resolved:* first generation run, Qwen3-4B-Thinking-
    2507-8bit at T=0.6/top-p=0.95/top-k=20, on **5 toy prompts only** — too small and too easy
    to settle the question, but indicative. Trace length 313–2048 tokens (one hit the 2048
    cap), so ≥6.5× spread across five near-trivial prompts; trace-to-answer ratio 2.8–5.9,
    median 4.2. The spread is already approaching an order of magnitude at the easy end, which
    argues the token-mean vs sequence-mean choice will matter here. Needs re-measuring on the
    real prompt distribution with a higher cap before it can be closed.
- Is a reweighted loss (DFT-style) worth trading away the loss-as-divergence instrument?
  Not answerable until the ceiling test from the cross-entropy work has been run once.
- If resampling is affordable, how often? STaR's stale-rationale overfitting is a diagnosed
  failure with a documented fix (resample every iteration, as TRICE does).
- Which regime is this project in: distillation from a stronger teacher, verified
  self-improvement, or unverified self-generation? The failure modes and the expected
  ceiling differ completely across the three.
- Is the domain verifiable (math, code, structured extraction) or open-ended? Determines
  whether a verifier-based filter is available at all.
- Single-generation finetuning or iterated generations? Collapse literature only bites in
  the iterated case.
- What synthetic fraction of the finetuning mixture, and does the reported ~20-30%
  degradation threshold from pretraining-scale studies transfer to the finetuning regime?
  Unverified.
- What decoding parameters will generation actually use? They define q, so they set the
  loss floor and the diversity ceiling before any training happens. Worth logging Ĥ(q) per
  pipeline variant from the first run.
- Is q off-policy by construction here? If so, the GKD line (train/inference distribution
  mismatch dominates) may matter more than the forward-vs-reverse-KL debate.
- Is the target of this project a *capability*, a *belief*, or a *disposition*? The three take
  different data (traces, documents, documents+traces) and different evaluations. The pipeline
  note assumes belief+disposition; confirm.
- Mix generic web text with the synthetic corpus, or not? Two credible sources disagree
  outright (1:1 with C4 vs. drop it entirely for higher saliency). Needs a local sweep.
- Does inoculation prompting survive the conditionalization critique? Unresolved in print;
  do not rely on it as the sole mitigation.
- What is this project's obfuscation probe? If any target behavior touches monitoring,
  oversight, or evaluation, prose-level grading is insufficient and a behavioral
  misbehavior-not-in-trace metric is required before any result is trustworthy.
- Implanted beliefs stay linearly detectable in activations in most reported cases. Does that
  matter for this project's purpose, or is behavioral equivalence enough?

## Published artifact
<!-- Exactly one URL, updated in place on every run. Never add a second. -->
https://claude.ai/code/artifact/8ad21c3f-b1b9-498d-83bd-4f6b6c518a74

## Sources
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754/) — ACL 2023, peer-reviewed.
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — NeurIPS 2022, peer-reviewed.
- [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717) — arXiv preprint; peer-reviewed venue not confirmed.
- [AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y) — Nature 2024, peer-reviewed (has an author correction).
- [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://arxiv.org/abs/2404.01413) — arXiv preprint.
- [A Note on Shumailov et al. (2024)](https://arxiv.org/abs/2410.12954) — arXiv preprint, critical commentary.
- [Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](https://arxiv.org/abs/2511.01490) — Findings of ACL 2026, peer-reviewed.
- [Characterizing Model Behavior Under Synthetic Data Training](https://arxiv.org/abs/2510.05133) — arXiv technical report, not peer-reviewed.
- [On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey](https://aclanthology.org/2024.findings-acl.658/) — Findings of ACL 2024, peer-reviewed.
- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) — arXiv preprint.
- [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) — ICLR 2024, peer-reviewed.
- [On-Policy Distillation of Language Models (GKD)](https://arxiv.org/abs/2306.13649) — ICLR 2024, peer-reviewed.
- [On LLM Knowledge Distillation: Forward KL vs Reverse KL](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-llm-knowledge-distil-157/blog/llm-knowledge-distil/) — ICLR 2025 Blogposts track, peer-reviewed.
- [Learning While Staying Curious: Entropy-Preserving SFT via Adaptive Self-Distillation](https://arxiv.org/abs/2602.02244) — arXiv preprint, Feb 2026.
- [Training Chain-of-Thought via Latent-Variable Inference (TRICE)](https://arxiv.org/abs/2312.02179) — NeurIPS 2023, peer-reviewed.
- [On the Generalization of SFT: A RL Perspective with Reward Rectification (DFT)](https://arxiv.org/abs/2508.05629) — arXiv preprint (v3, Feb 2026).
- [Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)](https://arxiv.org/abs/2503.20783) — arXiv preprint.
- [Group Sequence Policy Optimization (GSPO)](https://arxiv.org/abs/2507.18071) — arXiv preprint, Qwen Team.
- [Towards a Unified View of LLM Post-Training (UPGE / HPT)](https://arxiv.org/abs/2509.04419) — arXiv preprint.
- [Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning](https://arxiv.org/abs/2406.18629) — arXiv preprint.
- [VCORE: Variance-Controlled Reweighting for Chain-of-Thought Supervision](https://arxiv.org/html/2510.27462) — arXiv preprint.
- [Selective Critical Token Fine-Tuning](https://arxiv.org/abs/2510.10974) — arXiv preprint.
- [λ-GRPO: Learnable Token Preferences](https://arxiv.org/abs/2510.06870) — arXiv preprint, under OpenReview.
- [A Survey of On-Policy Distillation for LLMs](https://arxiv.org/html/2604.00626v1) — arXiv preprint, 2026.
- [Revisiting Entropy in RL for Large Reasoning Models](https://arxiv.org/abs/2511.05993) — arXiv preprint.
- [Believe It or Not: How Deeply do LLMs Believe Implanted Facts?](https://arxiv.org/abs/2510.17941) — arXiv preprint; reference SDF pipeline + belief-depth evals.
- [Practical Learnings from Synthetic Document Finetuning](https://www.lesswrong.com/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning) — blog post, not peer-reviewed; operational hyperparameters.
- [Training on Documents About Monitoring Leads to CoT Obfuscation](https://arxiv.org/abs/2605.15257) — arXiv preprint, May 2026; Haskins, Chughtai, Engels.
- [Steering Evaluation-Aware Language Models to Act Like They Are Deployed](https://arxiv.org/abs/2510.20487) — arXiv preprint; two-stage SDF + expert-iteration model organism.
- [Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time](https://arxiv.org/abs/2510.04340) — arXiv preprint, under review.
- [Conditionalization Confounds Inoculation Prompting Results](https://www.lesswrong.com/posts/znW7FmyF2HX9x29rA/conditionalization-confounds-inoculation-prompting-results) — blog post, not peer-reviewed; the counter-argument.
- [Evaluating chain-of-thought monitorability](https://openai.com/index/evaluating-chain-of-thought-monitorability/) — industry report.
- [Prompt Distillation (Tinker cookbook)](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/prompt-distillation/) — documentation.
- Cover & Thomas, *Elements of Information Theory*, ch. 2; Goodfellow et al., *Deep Learning*, §3.13 — textbooks (Cover & Thomas paywalled).
