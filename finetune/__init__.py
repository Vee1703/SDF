"""Finetuning stage: turning generated records into training rows, and running SFT.

`research_context.md` records that the training procedure is ordinary MLE/SFT and nothing
in the optimizer changes -- so the load-bearing decisions here are the completion mask and
the normalizer, not the optimization.
"""
