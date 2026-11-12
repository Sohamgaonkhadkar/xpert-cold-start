# Xpert Cold Start: Dossier-ARR-PRISM

## What we focused on

We prioritize relationship context, task discovery, conservative closure and restraint
in interventions. A false completion hides work; an intrusive reminder can make an
assistant unwelcome. We preserved the existing Python architecture and added targeted
fixes, rather than rebuilding it. The grading profile is deterministic; optional model
inference requires an explicit flag. The deliverable is a batch JSON program.

## Architecture

Contact dossiers combine seed relationships, saved-name aliases, chat membership and
speaker-attributed samples. Organisation routing separates home and work and guards
against promoting a client's business into the user's organisations. Goals require
actual message evidence and user endorsement, with at most ten predictions.

Anchor-Retrieve-Reconcile retains original turns and retrieves local and cross-chat
trails for commitments, requests, deadlines and decisions. Dates anchor to message
time. Expired deadlines never imply completion. Acknowledgements, future intent,
partial progress and unrelated later replies cannot establish execution. Cancellation
can drop a loop; later failure evidence can reopen it. We enumerate all history;
the checker alone applies the hidden scoring cutoff.

PRISM uses an asymmetric loss gate for assist/remind/leave. Its probabilities and
loss values are hand-set assumptions, not empirically calibrated posteriors. External
actions require approval; private preparation can proceed without it. Exact-ID
corrections change relationships, chat ownership and task ownership/status while
preserving unrelated predictions. Closing a corrected task requires later evidence.

## Evaluation and actual cost

On 7 November we downloaded the official archives directly. Schema, checker and example
match after newline normalization. Dev contains 4,502 messages /138 chats. Goals and
interventions are unlabelled there: its denominator is 56, whereas the example is /74.

| Run on official dev | Score /56 | Provider USD /1k messages | Seconds |
|---|---:|---:|---:|
| Original Arena, deterministic | 18.68 | 0 | 7.00 |
| Hardened, deterministic | 19.80 | 0 | 8.33 |
| Offline evidence refinements | 29.67 | 0 | 4.31 |
| Original Arena, GPT-4.1 mini live | 19.48 | 0.01751 | 92.07 |
| Hardened GPT-4.1 mini, serial /credit-limited | 19.56 | 0.00942 | 789.08 |
| Hardened GPT-4o mini, serial /credit-limited | 19.80 | 0.00150 | 229.88 |

Exact tested OpenRouter IDs: **openai/gpt-4.1-mini** and **openai/gpt-4o-mini**.
The hybrid compatibility model is **openai/gpt-4.1-mini**; the default command makes
zero model calls even with credentials. Optional model inference uses **--mode hybrid**.
The original live run returned 128,022 input /17,264 output tokens in 39 responses,
provider cost $0.0788312. Hardened serial runs returned 98,795 /13,075 tokens in 34
responses ($0.042390), and 38,565 /1,602 in 43 responses ($0.00674595), respectively.
Relationship and loop-discovery prompts dominated original token use; detailed stage
counts are in EVALUATION.md and the aggregate JSON. These are provider-reported costs,
not price-table estimates; failed requests may omit usage.

Both hardened models suffered HTTP 402s and fallbacks, then permanent insufficient
credits. Shared-account overlap confounds latency; these runs cannot rank models or
select a hybrid grading model. The example scores 74/74 offline and in tested live mode.

## Where it fails

Latest deterministic dev loops have 18.28% precision, 68.92% recall and 28.90% F1:
51 matches among 279 predictions against 74 labels. There are zero false DONE among
matched loops, 19 missed completions among matched loops (40 including undiscovered
ones), and six owner errors. The starting hardened result matched 32/74 with 408
predictions. The official README warns that its loop labels are incomplete; unmatched
predictions are not all necessarily invalid tasks.

Unseeded circle accuracy is 78.05%. Sparse contacts, dual roles, family/work vocabulary
collisions, implicit obligations and ambiguous unquoted completions remain weak. Dev
goal/intervention error rates cannot be measured. We do not hardcode identities or
duplicate predictions to exploit repeated answer sources. Hidden performance is unknown.

## Engineering decisions and limits

Inference never reads answers. Validated output is written atomically; unresolved
schema/reference errors return nonzero. The client caches requests, bounds concurrency,
reserves estimated cost and handles temporary in-flight-credit holds separately from
permanent credit failures. A Linux watchdog caps execution below 30 minutes. All 156
local tests pass; Python 3.10/3.12/3.14 and clean-container CI remain configured.

A dense synthetic 20k-message/320-chat run takes 49.05 s with 5,000 task candidates,
valid output and zero model calls. This tests local runtime, not hidden accuracy.
The optional hybrid path needs funded quality re-evaluation; it is not the grading
configuration. We retain Dossier-ARR-PRISM rather than costly orchestration. Private
data and credentials remain ignored. The measured errors above describe development
behavior; hidden-set accuracy and ranking remain unknown.
