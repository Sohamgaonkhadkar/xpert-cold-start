# Evidence-grounded corrections: 12 November 2026

The retained implementation scores **28.952575934283253/56**, up **0.704840294840295**
from the previous verified 28.247735639442958. It exceeds the supplied 27.65 benchmark
by 1.3026 points. This is an iteratively tuned development result, not a hidden-set
score or a verified competition rank. The earlier [ablation record](nov11_evaluation.md)
remains historical evidence.

## Corrections

1. **Task-list extraction and source preservation.** A heading indicating work/tasks
   plus multiple numbered action verbs establishes a bundled request. Numbered names
   or attendance lists do not. The bundle retains its source rather than merging
   into a narrower earlier task merely because of a shared rare topic. A diagnosed
   household work list had been merged into an earlier single-topic request.
2. **Hinglish meeting proposals.** Informal meeting/call proposals using "pe/par karte
   hai/hain" are recognized as a decision even without a question mark. Existing
   speaker-based ownership and group addressing rules still apply.
3. **Explicit email delivery.** An owner's prompt same-DM statement that the requested
   link/document/report is already in the requester's inbox is delivery evidence.
   Future promises and negated delivery remain open. This uses the existing two-hour
   bound and does not accept arbitrary email mentions.
4. **Bundle safety.** Completing, cancelling or taking over one item does not resolve
   or transfer the full task list. Whole-list evidence is required in deterministic
   resolution and model-assisted resolution. An optimistic mock model is explicitly
   tested against a partial-item completion; it cannot override the guard.
5. **Family safety.** The dominant-household-DM spouse prior can resolve an ambiguous
   friend/relative/sibling classification, but cannot override an inferred parent,
   child, staff, vendor or other explicit relationship. This fixes a synthetic parent
   counterexample without changing the dev score.

| Cumulative candidate | Official /56 | Predicted loops | Matched /74 |
|---|---:|---:|---:|
| Starting reviewed commit `bb8b3e8` | 28.2477 | 276 | 48 |
| Structured actions and email delivery | 28.6949 | 277 | 49 |
| Preserve bundled source | 28.9526 | 278 | 50 |
| Bundle closure, model guard and family guard | **28.9526** | **278** | **50** |

Every meaningful implementation candidate was checked with the unmodified official
checker. The retained version passes all **114 tests**, including the official worked
example and zero-seed variant (74/74), adversarial regressions and shell packaging.
CI configuration, credential handling, budgets, atomic output and schema/reference
validation are preserved. Raw diagnostics, predictions, answers and credentials are
not committed.

## Final measurements

Circles: 9.0732/12; organisations: 3.6958/8; chats: 6.3415/8; loops found:
3.9773/14; loops closed: 5.8649/14. Label-based precision is 17.99%, recall 67.57%
and F1 28.41%. Matched false DONE and false DROPPED remain zero. Actual inference
runtime was **4.9289 seconds** in the final single local run. Model calls, input/output
tokens and API cost are all **zero**; there were no OpenRouter attempts.

## Remaining weaknesses and evidence limits

There are 24 unmatched reference loops, 19 matched missed completions, 41 missed
completions including undiscovered loops, six matched owner disagreements and ten
incorrect unseeded circles out of 41 labels. Organisation matching remains unchanged.
Goals and interventions are unlabelled and cannot be assessed from this dev set.

All six owner disagreements are present before temporal reconciliation. Several
reference owners conflict with ordinary speaker-based responsibility, including a
counterparty's own arrival promise labelled ME-owned. Bilateral coordination can
also be genuinely ambiguous. These were audited rather than mechanically changed to
ME. No identities or reference sources were added to inference rules.

Some remaining source labels point to bare photos, time fragments, birthday wishes,
duplicate origins or statements without an explicit obligation. Some done labels
lack visible later completion. These were not converted into identity-specific
exceptions or elapsed-time completion rules. Further reliable gains require richer
context or a separately verified model; no funded-model improvement is claimed.
The retained configuration is deterministic; model-assisted quality remains unverified.
