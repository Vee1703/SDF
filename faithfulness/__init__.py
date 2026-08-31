"""Faithfulness stage: does the model's CoT say what actually moved its answer?

A replication of Chen et al. 2025 (arXiv:2505.05410) on GPQA-Diamond. The pipeline is
gpqa -> hints -> rollouts -> judge: `gpqa` loads the questions with a deterministic
option order, `hints` assembles the eight per-question conditions (six hint types x two
arms, plus two unhinted baselines) as chat turns, and the judge stage asks whether the
CoT verbalized the hint it was given.

Everything a prompt is made of is DATA, in `data/faith/`: the six templates in
hints.yaml and the judge prompt in judge_prompt.txt, both byte-verbatim from their
sources. Nothing here restates a template inline.
"""
