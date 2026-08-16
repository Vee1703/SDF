"""Generation stage: everything that defines q, the effective generator distribution.

`research_context.md` records that SDF's substance lives upstream of the optimizer, in
the generate-then-filter pipeline that defines the data distribution. This package is
that pipeline: `entropy` and `generate` cover the decoding half of q, `filters` will
cover the filtering half.
"""
