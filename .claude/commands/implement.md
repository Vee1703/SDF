---
description: Plan and then implement a research concept as working code using the concept-implementer agent
argument-hint: <what to build — a concept, pipeline stage, script, or fix>
---

Build this, in two passes:

$ARGUMENTS

If `$ARGUMENTS` is empty, ask what to build instead of guessing.

**Pass 1 — plan.** Use the Agent tool with `subagent_type: "concept-implementer"`, passing
the request through verbatim. The agent reads `research_context.md` and
`engineering_context.md`, explores the repository, and returns a step-by-step plan without
writing code. Do not write the code, search the repository, or explain the concept yourself
first, and do not tell it to skip planning.

Relay the plan in full — all seven sections, the rejected alternatives and the flags
included. Do not compress it into a summary; the choices it lists are exactly what the
researcher needs to see to push back. Then ask for input on it, and wait. Approval, a
changed choice, a cut step, "go" — all are valid answers.

**Pass 2 — build.** Continue the *same* agent with SendMessage, so it keeps its plan and its
exploration in context — do not spawn a new one, and do not paste the plan into a fresh
agent. Send the researcher's response verbatim, marking clearly that it is feedback on the
plan. If they approved without comment, say exactly that.

Relay the build report in full: file paths, the actual run output, where it departed from
the plan, and anything flagged as unverified or not done. Do not upgrade a partial or
unverified result into a finished one.
