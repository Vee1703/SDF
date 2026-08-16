---
description: Explain a research concept, method, or paper using the concept-explainer agent
argument-hint: <concept, method, paper, or question>
---

Use the Agent tool with `subagent_type: "concept-explainer"` to answer this research
question:

$ARGUMENTS

Pass the question through verbatim. Let the agent do its own source reading and its own
`research_context.md` maintenance — do not search or explain the concept yourself first.

If `$ARGUMENTS` is empty, ask what concept to explain instead of guessing.

The agent's final report is the deliverable. Relay its explanation in full — sources,
confidence statements, and the connection to this project included. Do not compress it
into a summary.
