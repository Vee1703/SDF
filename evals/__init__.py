"""Evaluation stage: behavioural probes and before/after comparison.

`research_context.md` is explicit that for behavioural targets the loss is uninformative
and checkpoint selection must be on eval, never on loss -- and that reading traces cannot
distinguish success from its dangerous twin. Everything that scores a model rather than
generating from it lives here.
"""
