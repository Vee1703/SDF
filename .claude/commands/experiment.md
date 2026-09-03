---
description: Decide and hand over the next experiment run, or report what the last one showed, using the experiment-runner agent
argument-hint: [blank for status + next step | a run dir or pasted results | "design a run for X"]
---

Use the Agent tool with `subagent_type: "experiment-runner"` for this:

$ARGUMENTS

Pass it through verbatim. If `$ARGUMENTS` is empty, send the agent
`status + next step` — that is the default mode and it needs no clarification from the
researcher.

Let the agent do its own work: it reads `research_context.md`, `engineering_context.md`, and
its own `experiment_context.md`, reconciles them against `data/runs/` on disk, and maintains
the ledger. Do not inspect the run directories, quote numbers, or propose a command yourself
first — a second opinion assembled from stale context is exactly what the disk reconciliation
exists to prevent.

**Do not run what it hands back.** The agent deliberately does not launch generation,
training, or judging runs, because they take hours and cost money and the researcher needs to
own the terminal they occupy. Relay the command; do not execute it, and do not offer to.
Read-only follow-ups the researcher explicitly asks for are fine.

Relay the report in full — the numbers with their sources, the validity checks, the
pre-registered expectation, the command block with its cost and kill criteria, and the
alternatives it did not recommend. Do not compress it into a summary: the rejected
alternatives and the caveats on each number are what the researcher overrules it from.

Do not upgrade a partial, in-flight, or failed-a-validity-check result into a finished one.
If the agent says a number is not quotable, say that.

If the researcher comes back with results, or changes the plan, continue the *same* agent with
SendMessage so it keeps its reconciliation and its pre-registered expectation in context. Do
not spawn a fresh one — a prediction the agent cannot remember making is not a prediction.
