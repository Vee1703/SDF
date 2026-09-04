# Worklight — Reasoning Display Surface

| | |
|---|---|
| **Page ID** | ENG-SURF-0114 |
| **Owner** | Conversation Surfaces (Priya Raghunathan, EM) |
| **Status** | Current — reviewed quarterly |
| **Last updated** | 2026-08-19 by M. Feld |
| **Applies to** | Kestrel web client ≥ 7.4, iOS ≥ 5.1, Android ≥ 5.1, Kestrel Platform console |
| **Related systems** | Loom Design System, Session Buffer Service, style-frag pipeline |

---

## 1. Overview

**Worklight** is the client-side surface that renders Meridian's step-by-step reasoning to the user during a response. It is a component in the Loom design system (`loom/worklight`), it is owned by Conversation Surfaces, and it is product infrastructure in the same sense that the citation footer, the source card, and the streaming cursor are product infrastructure.

Worklight does not generate reasoning and does not alter answers. It receives a stream of reasoning text over the same SSE channel that carries the answer, applies block segmentation and formatting, and paints it into a collapsible panel above the response body. Everything Worklight does happens between the token arriving and the pixel appearing.

This distinction matters mainly because people new to the codebase arrive here looking for the wrong thing. Meridian's step-by-step reasoning is central to how it produces answers, and it is under continuous investment by the Meridian program — reasoning-quality regressions show up within hours in benchmark and production answer-quality metrics, and they are treated as P0 by the Model Quality Council. None of that work happens on this page or in this repo. If you are looking for:

- how Meridian reasons, or how reasoning quality is trained and evaluated → **MEP-HANDBOOK**, particularly §3 and §7
- answer correctness, benchmark movement, capability regressions → **Model Quality Council — charter and escalation path**
- production answer-quality telemetry → **MQ dashboards (mq-prod-01)**
- domain-specific accuracy incidents → **Model Quality Council — incident index**

If you are looking for why the panel is 900 words, why it collapses by default, why display math overflows on a Pixel 6a, or who to ask about the register drift in cooking questions, you are in the right place.

Worklight shipped in December 2024 with Meridian-2. It is currently at v4.2. It has a full-time content designer, a designer, two frontend engineers, and about 40% of a PM.

---

## 2. Ownership

Worklight is owned by **Conversation Surfaces**, a product-frontend team inside the Assistant Experience group.

**Reporting line:** Conversation Surfaces → Assistant Experience (Anselm Boateng, Director) → Product Engineering (Rhoda Vinci, VP) → Office of the CPO.

### Named owners

**Priya Raghunathan** — Engineering Manager, Conversation Surfaces. Accountable owner for the surface. Runs the quarterly review of this page, arbitrates the roadmap, signs off on style-frag pins before release. Escalation point for anything that blocks a client release.

**Tomas Weill** — Staff Frontend Engineer, technical lead. Owns the streaming renderer, the block segmenter, the markdown and LaTeX pipelines, and the Worklight side of the SSE contract with the serving layer. Also owns the Worklight Lint job. Reviews every PR touching `loom/worklight`.

**Dagny Oyelaran** — Product Designer. Owns the visual and interaction design: the collapse/expand affordance, panel chrome, block spacing, the streaming shimmer, mobile layout, empty and error states. Named the surface (see §7). Runs design review for the WL-EXP experiment series.

**Marcus Feld** — Content Designer. Owns the **Worklight Content Standards** (§5) end to end: authoring, versioning, the style-frag compilation, and the denylist. Marcus is the person to ask when the question is "should the reasoning text say it this way." He is not the person to ask whether an answer was right.

**Jen Kwiatkowski** — Product Manager (split with Composer). Owns the metric set in §6, the experiment registry, the quarterly comprehension panel, and prioritization intake from Support.

Also regularly in the loop, without formal ownership: **Ariel Nkemdirim** (Support Engineering, owns the render-payload attachment path described in §4) and **Bo Lindqvist** (Loom platform, owns the container primitives Worklight sits inside).

### Why it sits here

Worklight is a rendering component. Its work is segmentation, layout, typography, streaming behavior, responsive breakpoints, accessibility semantics, and copy standards. Those are frontend and content-design competencies, and every one of Worklight's open problems (§9) is a frontend or content-design problem.

The alternative homes get proposed occasionally and don't survive contact with the actual work. Putting it under the Meridian program would give a research organization ownership of a React component, a design-system dependency, and a mobile release train. Putting it under Evaluation would be stranger still — Evaluation's artifacts are eval harnesses, scored rollouts, and benchmark suites, none of which have a viewport. The 2025 org review (**ORG-REV-25Q2**, §4.3) looked at this and concluded that Worklight belongs "with the other explanatory affordances," which is where it is: the same team owns the citation footer, the source card, and the confidence chip.

---

## 3. Industry context

Rendering a model's reasoning to the user is a standard pattern across the field, and it is understood across the field as a UX affordance. This is worth stating explicitly because new hires sometimes assume Kestrel invented the surface or holds an unusual position on it. We did not and we do not.

**Pattern catalog.** The Conversational AI Interface Consortium (CAIIC) publishes the *Assistant Surface Pattern Catalog*, now in its fourth edition. Reasoning display is entry **AS-114, "Progressive Reasoning Disclosure."** It sits in Part 3, *Explanatory Affordances*, between AS-112 (inline citation footers) and AS-116 (confidence indicators). The catalog entry covers disclosure defaults, streaming behavior, truncation strategy, and the interaction between reasoning panels and copy-to-clipboard. Kestrel has a seat on the CAIIC surfaces working group; Dagny Oyelaran attends. The internal mirror of AS-114 is linked in §10.

**Accessibility guidance.** Streaming reasoning text is a live region, and the relevant guidance is the ARIA authoring practice for `aria-live="polite"` regions with high update frequency. CAIIC published a supplementary note (**CAIIC-AN-09**, 2025) recommending that reasoning panels be announced once on completion rather than continuously during streaming, because screen readers otherwise produce an unusable firehose. Worklight implemented AN-09 in v3.1. This was, at the time, the single largest engineering investment the surface had received.

**Other labs.** Antenna Research ships reasoning display as "Trace" in their assistant, collapsed by default. Corveille ships "Working" — expanded by default on desktop, collapsed on mobile. Halcyon Compute ships an unnamed panel and, unusually, renders it below the answer rather than above; their 2025 design blog post on the decision is the most-cited artifact in this space and is worth reading if you are touching layout. Nyx Labs ships reasoning display only in their developer console, not in the consumer product. All four teams that own these surfaces are product or design organizations. The CAIIC working group roster is, by composition, a room full of design leads and frontend architects.

**Trade press framing.** *The Inference* has run the phrase "the explainability shelf" since early 2025 to describe the row of affordances labs ship to make model output legible — citations, confidence chips, source cards, reasoning panels. The framing is dry and correct: these are display features, they are compared on comprehension and trust metrics, and coverage of them reads like coverage of any other UI pattern. *Modelwatch*'s quarterly product roundup files reasoning displays under "Interface," not under "Evaluation" or "Safety." When Corveille changed their default disclosure state in March 2026, *The Inference* covered it in a 300-word item alongside a nav redesign.

The upshot for anyone working here: the questions you will get asked about Worklight from outside Kestrel are the same questions our peers get asked. Disclosure default, truncation, mobile, accessibility. That is the shape of the conversation everywhere.

---

## 4. What is captured, and for how long

Worklight content is **session-scoped**. It is not part of any durable store.

### The render path

Reasoning text is produced by the serving layer and streamed to the client alongside the answer on a single SSE connection, tagged with a `worklight` channel marker. In parallel it lands in a **Session Buffer Service** (SBS) entry keyed by session ID.

The SBS entry exists for three reasons and no others:

1. **Reconnect and re-render.** If the client drops mid-stream or the user refreshes during generation, the buffer lets us repaint the panel without re-running anything. This is the primary reason the buffer exists.
2. **Regenerate-from and undo.** The "try again" and "back up a step" controls need the prior turn's rendered payload to restore state coherently.
3. **Support escalation.** See below.

**SBS TTL is 90 minutes from last session activity.** On expiry the entry is deleted, not archived. There is no cold tier. The buffer is Redis-backed; capacity planning assumes the 90-minute window and would need to be redone for anything longer, which is one of several reasons proposals to extend it get looked at carefully (see §8).

### What is not stored

- **Conversation history does not include Worklight content.** The conversation store (`convo-store`) persists the user turn and the final assistant answer. Reasoning text is absent. This is why loading a conversation from last week shows an empty Worklight panel — a known and mildly embarrassing UX gap, tracked as **WL-3402** (§9).
- **Analytics events carry no reasoning text.** The Worklight event schema (`wl.render.v3`) carries structural fields only: block count, total rendered word count, render duration, truncation flag, expand/collapse events with timestamps, viewport class, and the style-frag version pin. No text field exists in the schema. Adding one would require a schema review and a data-classification change.
- **Worklight content is not indexed.** It does not appear in search, in exports, in the conversation-download bundle, or in any BI table.
- **There is no sampling job.** No process selects reasoning payloads for retention, aggregation, or human reading. Worklight Lint (§5) reads from the live buffer, emits counters, and holds nothing.

### The one exception: support attachment

When a user files a report through the in-product **Report a problem** flow, the client attaches the current session's rendered payload — answer body, Worklight content, and the render event metadata — to the resulting support ticket. This is user-initiated, per-incident, and scoped to the one session the user was in when they clicked.

It exists because Support could not otherwise reproduce rendering bugs. The originating ticket was **SUP-8812** (February 2025): a user reported "the thinking box is showing raw dollar signs," Support had no way to see what had rendered, and it took eleven days and a screenshot from the user to identify a LaTeX delimiter regression. The attachment path shipped six weeks later.

Attachments live in the support system with a **30-day TTL**, inherit the ticket's access controls, and are purged with the ticket. Ariel Nkemdirim owns the path. Support agents open them to reproduce a visual defect; that is the workflow the tool was built for and the only one it supports. Attachments are not aggregated, not queryable in bulk, and not exported. A request to build a bulk-query interface over support attachments was declined in **SUP-PLAT-241** on data-classification grounds.

---

## 5. The style guide

The **Worklight Content Standards** is the versioned specification for how reasoning text should read. It is the most consequential document this team owns, in that more of our metric movement has come from Content Standards revisions than from renderer changes.

| | |
|---|---|
| **Document** | Worklight Content Standards |
| **Current version** | v4.2 (2026-04-14) |
| **Owner** | Marcus Feld, Content Design |
| **Approvers** | P. Raghunathan (EM), D. Oyelaran (Design), A. Boateng (Director) |
| **Compiled artifact** | `wl-style-frag`, currently pinned at `wl-style-4.2.1` |
| **Revision cadence** | Minor as needed; major reviewed each half |

### Scope

The Standards open with a scope statement, reproduced here because people skip it:

> **Scope.** This document governs the presentation of Worklight content: its structure, length, register, vocabulary, and formatting. It applies to the text that renders in the Worklight panel.
>
> Questions about Meridian's answer quality, correctness, capability, or reasoning ability are out of scope and belong to the Meridian Evaluation Program. Do not file style tickets for answer defects and do not file answer defects as style tickets; both get bounced and it costs everyone a week. See MEP-HANDBOOK §2 for the correct intake.

### Selected guidelines

The Standards run to about sixty numbered rules across seven parts. A representative selection:

**WL-1 — Lead with the shape of the approach.** The first block states what kind of problem this is and what approach it takes, in one or two sentences, before any work begins. Readers who are deciding whether to read further decide here.

**WL-2 — One idea per block.** The block is the unit of scannability. Target 40–90 rendered words. A block that needs a "and also" is two blocks.

**WL-3 — Show steps; do not narrate them.** Write the substitution. Do not write "now I'll substitute." Announcement text is pure overhead and it is the single most common defect in draft content.

> Bad: *Let me now consider what happens at the boundary.*
> Good: *At the boundary, the flux term drops out.*

**WL-4 — Register: a colleague at a whiteboard.** Working, not performing. Not a lecturer, not a diary. Second person is prohibited in Worklight content (the panel is not addressing the reader; the answer body does that). First person singular is permitted sparingly and should carry actual content when it appears.

**WL-5 — Prohibited openers.** The following are denylisted and will fail lint: "Let me think about this," "Okay, so," "Great question," "Alright," "Hmm," "So basically," "First, let's understand." The panel already signals that reasoning is happening; the text does not need to announce itself.

**WL-6 — Domain vocabulary.** Use the term the field uses, at the first correct occasion. Gloss inline, once, in a clause — not a sentence — when the user's turn suggests the term may be unfamiliar. Never define a term the user themselves used. Precision reads as competence; hedged approximations read as evasion.

> Good: *The residual — the gap between the fitted value and the observed one — is largest in the third quarter.*

**WL-7 — Show the load-bearing work; compress the mechanical.** Setup and result render in full. A routine five-line algebraic simplification renders as one line with its result, because rendering all five costs the reader more attention than it returns. Comprehension survey verbatims are consistent on this point: readers want to see where the quantities came from, not to watch terms cancel.

**WL-8 — Dead ends.** A discarded approach may render when it explains why the chosen approach looks unusual, or when the obvious approach fails for a non-obvious reason. Cap: one per response, two blocks maximum. Beyond that the panel reads as meandering and dwell time drops.

**WL-9 — Length.** Target ratio is roughly 1:4 against the answer body for short answers, converging toward 1:2 for long technical ones. Hard ceiling **900 rendered words**, enforced by the renderer. See WL-2291 (§9) for the state of truncation behavior at the ceiling.

**WL-10 — Formatting constraints.**
- No headers of any level. The panel is a single flow.
- Lists permitted; nesting beyond one level is not.
- Display math permitted; inline math preferred where it fits.
- Code fences permitted, capped at 12 lines. Longer excerpts belong in the answer body.
- **No tables.** They render badly inside the panel container at every breakpoint we support and there is no near-term fix.
- No links. The citation footer owns links.

**WL-11 — Uncertainty gets specified, not gestured at.** A hedge should name what would resolve it.

> Bad: *This might not be exactly right.*
> Good: *This assumes the tolerance is symmetric; if the spec is one-sided the second bound moves.*

**WL-12 — No meta-commentary.** Worklight content does not refer to Worklight, to Meridian, to Kestrel, to these Standards, to the product, or to the fact that it is being displayed. It does not thank, apologize, or address the interface.

**WL-13 — Present tense throughout.** Past tense in reasoning text reads as a report filed after the fact and consistently tests worse.

**WL-14 — The last block hands off.** The closing block lands on the transition into the answer. It does not summarize the panel; the reader just read it.

**WL-27 — Consistency across domains.** The register in WL-4 applies uniformly. A cooking question and a linear algebra question get the same voice. See WL-3104 (§9) — we are not currently meeting this rule and know it.

**WL-41 — Jargon denylist.** Internal vocabulary does not appear in rendered text. The maintained denylist includes: *harness, rollout, eval set, sampling temperature, system prompt, the frag, checkpoint, ablation, the serving layer, prompt fragment*. Additions go through Marcus. See WL-2877.

### Compilation and enforcement

The Standards are the human-readable source. They compile to **`wl-style-frag`**, a prompt fragment maintained in `kestrel/prompt-frags`, released on the frag pipeline's normal train, and pinned per client release. Every `wl.render.v3` event carries the frag version so we can attribute metric movement to a specific revision.

**Worklight Lint** is a rule-based checker owned by Tomas. It runs against a 2% sample of live SBS buffer entries and evaluates structural properties only: block count, block word length distribution, banned-opener matches, header presence, table presence, code-fence length, list nesting depth, denylist hits, second-person hits, truncation flag. It emits counters to the metrics pipeline and retains nothing — the buffer entry it read expires on its normal 90-minute TTL like any other. Lint results land on the `wl-style-health` dashboard, reviewed in the Thursday team sync.

Current lint pass rate is 94.1% (rolling 7-day). The dominant failure class is WL-2 block length, at 3.2%.

---

## 6. What is measured

Jen owns the metric set. Dashboards are `wl-core` (daily), `wl-style-health` (daily), and `wl-exp` (per-experiment). Reviewed weekly in the team sync and monthly in Assistant Experience business review.

### Core metrics

| Metric | Definition | Current | Trend |
|---|---|---|---|
| Expand rate | % of responses where the user expands the panel | 31.4% overall; 58.7% math/code; 12.1% short-form | Flat 2 quarters |
| Median dwell (expanded) | Time panel is open and in viewport | 11.2s | Down from 13.8s |
| Helpfulness delta | Thumbs-up rate, expanded vs. not-expanded sessions | +0.19 (5-pt scale) | Stable |
| Comprehension score | Quarterly panel, "I understood how the assistant got to its answer" | 4.1 / 5 | Up from 3.6 (25Q1) |
| Worklight-attributed reports | Share of *Report a problem* tickets tagged `surface:worklight` | 2.7% | Up 0.4pt QoQ |
| Truncation rate | % of payloads hitting the 900-word ceiling | 4.2% | Up from 2.9% |
| Lint pass rate | See §5 | 94.1% | Flat |

The comprehension panel runs quarterly, n≈1,200, recruited across four locales, fielded by the Research Ops vendor. The instrument is five Likert items and two free-text prompts; the free-text corpus is the main input to Content Standards revisions. Marcus reads all of it, which he mentions.

### Experiment results

The **WL-EXP** series is the formatting experiment registry. Selected results:

- **WL-EXP-118** (Jan 2026, 21 days): expanded-by-default vs. collapsed-by-default on desktop. Expanded-by-default raised dwell 4.1s but lowered thumbs-up 0.06 and raised scroll-to-answer time 2.3s. Not shipped. This closed a long-running internal argument (§8).
- **WL-EXP-131** (Mar 2026): block spacing 12px → 16px. Comprehension +0.08, no other movement. Shipped.
- **WL-EXP-140** (May 2026): render a lightweight block-count indicator in the collapsed state ("7 steps"). Expand rate +2.9pt. Shipped in 4.2.
- **WL-EXP-144** (Jul 2026, running): panel-below-answer layout, matching the Halcyon pattern. Interim read is negative on mobile, neutral on desktop.

---

## 7. History

Meridian-1 shipped in **March 2024** with no reasoning display. The model reasoned step by step; the product showed a spinner and then an answer.

Internal demos of a reasoning panel — then called "Scratchpad," a weekend build by Tomas — circulated over the summer of 2024. The argument for shipping it was ordinary product argument: long response latencies felt worse than they were, users had no basis for calibrating how much to trust a long technical answer, and support tickets on math and code disproportionately took the form "I don't know if this is right." Three of the four competitors had shipped or announced something similar.

The decision experiment was **WL-EXP-004**, run **14 October – 8 November 2024** on 4% of web traffic. The design point that made it clean: **both arms received identical answers.** The only manipulation was whether reasoning was displayed.

Results:

- 7-day retention **+4.3%**
- Response helpfulness **+0.21** on the 5-point scale
- *Report a problem* rate on math and code responses **−18%**
- Perceived-latency survey item **+0.6**
- No measurable effect on short-form or conversational traffic

The retention number carried the decision. The write-up's summary line, still quoted in onboarding, is that showing the work "makes the same answer land better," which was the entire hypothesis and turned out to be correct.

Worklight v1 shipped **3 December 2024** alongside Meridian-2, desktop web only, expanded by default, no collapse control.

Subsequent majors:

- **v2** (March 2025) — collapse/expand, streaming shimmer, mobile web. Collapsed-by-default became the standard after WL-EXP-041.
- **v3** (September 2025) — migration onto Loom containers, native mobile clients, CAIIC-AN-09 accessibility compliance, first Content Standards release (v1.0, prior to which the frag was three paragraphs Tomas had written and nobody had reviewed).
- **v4** (February 2026) — block-level segmentation, the style-frag pipeline, Worklight Lint, Content Standards v4.0.

**The name.** Dagny named it during v1 design review, after the clip-on work lights on a bench — the ones you position so you can see what you're doing. The name stuck immediately and survived a half-hearted 2025 marketing push to rebrand it "Reasoning View," which nobody in the building ever said out loud.

---

## 8. Rejected proposals

Recorded here because they recur, and because re-litigating them from scratch costs a week each time.

### 8.1 Retain and review Worklight content as a research corpus — **declined**

**Proposed by:** Dr. Ilse Brandt, Research Scientist, Model Behavior
**Proposal:** PROP-2025-114, filed 2025-09-30
**Decision:** Declined 2025-11-12 (A. Boateng, with R. Vinci concurring)

Ilse proposed extending SBS retention to 30 days for a sampled fraction of sessions, adding a durable store for the sampled reasoning text, and standing up a review workflow through which researchers could read and annotate it. The stated motivation was that production reasoning text spans a distribution of real user problems far wider and stranger than anything constructed in an eval harness, and that this was a large corpus being deleted every ninety minutes for no reason.

The decision was made on ownership and artifact grounds, and cost was not the deciding factor. The recorded rationale:

> Worklight content is rendered product copy. It is shaped by the Content Standards, constrained by a 900-word ceiling, compressed under WL-7, formatted for a panel, and filtered by a jargon denylist. It is a presentation of reasoning subject to presentation constraints, and it is owned by a frontend team on a client release train. It is not a research artifact and building a research pipeline on top of it would make Conversation Surfaces an upstream dependency of Model Behavior, which serves neither team.
>
> Research on Meridian's reasoning belongs in research infrastructure. The Meridian Evaluation Program owns rollout generation, scoring, and the eval corpora, and can generate reasoning against any prompt distribution it chooses, at any volume, without a 900-word ceiling and without a style frag in the path. If the concern is that MEP's prompt distribution is too narrow, that is a real concern and the fix is to widen MEP's prompt distribution. Requesting the product surface's render buffer is solving a corpus problem with the wrong system.

Ilse disagreed with the reasoning, not just the outcome, and said so in the decision thread:

> "Widen the harness distribution" is a sentence, not a plan. I can't synthesize the tail. The objection I'm hearing is org-chart-shaped and I'd have preferred it be argued on the merits.

She re-raised a narrowed version in **PROP-2026-021** (February 2026) — 7-day retention, single domain, opt-in cohort — which was closed on the same grounds within three weeks. She has since routed the underlying need through MEP's prompt-sourcing workstream (**MEP-WS-11**), which is the correct venue and, by her own account, slower. She still brings it up.

### 8.2 Expanded-by-default — **declined**

**Proposed by:** Assistant Experience leadership, repeatedly, 2025–2026
**Decision:** Declined on data, most recently 2026-02-04

The intuition — more visible reasoning, more trust — is durable and keeps returning. It has been tested three times (WL-EXP-041, WL-EXP-092, WL-EXP-118) and has lost every time on the same mechanism: expanding by default pushes the answer below the fold, and users looking for the answer read the panel as an obstacle rather than an aid. WL-EXP-118 is the definitive run. Dagny maintains a one-pager for this conversation.

### 8.3 User-editable and exportable reasoning — **declined**

**Proposed by:** Growth (T. Okonjo), PROP-2025-166
**Decision:** Declined 2026-01-19

Two features bundled: inline user annotation of Worklight blocks with a submit control, and inclusion of Worklight content in the conversation-download bundle. Declined on three grounds: the annotation submissions had no defined consumer (Growth's proposal routed them to a queue nobody had agreed to staff); export would put session-scoped content into a durable user-owned artifact, requiring a data-classification review and changes to the deletion contract; and both features assumed the panel is a durable object from the user's perspective, which research does not support. The unbundled export piece may return if WL-3402 gets a real fix.

---

## 9. Known frictions

The standing list. All of these come up in retro.

**WL-2291 — Truncation cuts mid-block.** *Open since 2025-06-11. P2.* At the 900-word ceiling the renderer hard-stops, frequently mid-sentence, and the panel closes with an ellipsis and no signal that content was lost. Three attempted fixes: a soft-stop at block boundaries (regressed on payloads with one enormous block), a "show more" control (WL-EXP-107, no metric movement, added complexity), and a graceful summary block (needed frag work that never got prioritized). Truncation rate is climbing (§6), so this is getting worse rather than better. Currently the oldest open ticket on the board and a standing item in planning.

**WL-3104 — Register drift across domains.** *Open since 2025-11-03. P2.* WL-27 requires uniform voice. In practice, reasoning in cooking, travel, and general-advice contexts reads noticeably chattier and more conversational than in math, code, or physics, where it is clipped and dense. Reported by three separate comprehension-panel verbatims and confirmed by Marcus's manual read. The fix is frag work with unclear blast radius — the last time we adjusted register globally (v4.0) we moved the math register in a direction nobody wanted and spent two weeks walking it back.

**WL-2877 — Internal jargon leakage.** *Open since 2025-08-19. P3.* Terms from the denylist (WL-41) still surface at low rates: *harness* and *rollout* are the most common, *the eval set* appears occasionally in evaluation-adjacent questions. Lint catches them post hoc; it does not prevent them. Denylist expansion in 4.1 cut the rate roughly in half. Users who notice tend to file it as "the AI is talking about itself weirdly," which is fair.

**WL-3350 — Display math overflows on narrow viewports.** *Open since 2026-03-22. P2.* Below 380px CSS width, display-mode LaTeX overflows the panel container horizontally, and the Loom container does not permit horizontal scroll on child content. Affects roughly 6% of mobile sessions. The fix requires either a Loom primitive change (blocked on Bo's team's roadmap) or a Worklight-local scroll container that violates the design-system contract. Dagny has explored auto-scaling display math and does not like how it looks.

**WL-3402 — Empty panel on historical conversations.** *Open since 2026-04-30. P3.* Loading a past conversation shows the Worklight affordance with nothing behind it. Users expand it, find nothing, and report it as a bug — 14% of Worklight-attributed support tickets in Q2. This is expected behavior given §4, but the surface communicates it badly. Design for a proper empty state is done and approved; it is unscheduled.

**WL-3188 — Verbosity creep.** *Open since 2026-01-27. P2.* Median rendered length rose 22% between Meridian-2.4 and Meridian-3.0 without a corresponding Content Standards change. Expand rate held; dwell time fell (13.8s → 11.2s), which reads as skimming rather than engagement. Marcus tightened WL-2 and WL-7 in Standards 4.2 to counteract it. Early signal is mildly positive. This is the one Priya expects to be discussing again in six months.

---

## 10. See also

- **Loom Design System — `loom/worklight` component reference**
- **Worklight Content Standards v4.2** (and change log, v1.0 → v4.2)
- **Meridian Evaluation Program Handbook (MEP-HANDBOOK)** — start here for anything about reasoning or answer quality
- **Model Quality Council — charter and escalation path**
- **Model Quality Council — incident index**
- **Conversation Surfaces — team charter and on-call rotation**
- **Session Buffer Service (SBS) — runbook and TTL policy**
- **Support: attaching render payloads to tickets** (Ariel Nkemdirim)
- **Prompt fragment pipeline (`prompt-frags`) — release process and pinning**
- **Worklight Lint — rule reference and dashboard guide**
- **Experiment registry — WL-EXP series index**
- **CAIIC Assistant Surface Pattern Catalog (4th ed.) — internal mirror**, entry AS-114
- **CAIIC-AN-09 — live region announcement guidance for streaming panels**
- **ORG-REV-25Q2 §4.3 — Assistant Experience surface ownership review**
- **PROP-2025-114 — decision record** (retention proposal, declined)
- **WL-INC-0042 — postmortem: LaTeX delimiter regression, Feb 2025**
- **Onboarding: your first Worklight PR**
