# Xpert Cold Start: Dossier-ARR-PRISM

## What we focused on

We prioritize relationship context, task discovery, conservative closure and restraint
in interventions. A false completion hides work; an intrusive reminder can make an
assistant unwelcome. We preserved the existing Python architecture and added targeted
fixes, rather than rebuilding it. Selective inference avoids paying a model to read
routine chatter. The deliverable is a batch JSON program, with no frontend or database.

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
| Original Arena, GPT-4.1 mini live | 19.48 | 0.01751 | 92.07 |
| Hardened GPT-4.1 mini, serial /credit-limited | 19.56 | 0.00942 | 789.08 |
| Hardened GPT-4o mini, serial /credit-limited | 19.80 | 0.00150 | 229.88 |

Exact tested OpenRouter IDs: **openai/gpt-4.1-mini** and **openai/gpt-4o-mini**.
The original live run returned 128,022 input /17,264 output tokens in 39 responses,
provider cost $0.0788312. Hardened serial runs returned 98,795 /13,075 tokens in 34
responses ($0.042390), and 38,565 /1,602 in 43 responses ($0.00674595), respectively.
Relationship and loop-discovery prompts dominated original token use; detailed stage
counts are in EVALUATION.md and the aggregate JSON. These are provider-reported costs,
not price-table estimates; failed requests may omit usage.

Both hardened model runs suffered many HTTP 402s and deterministic fallbacks, followed
by permanent insufficient account credits. They overlapped briefly on a shared account;
latency is confounded. We cannot rank models or select a grading model from these runs.
The official example scores 74/74 in offline and tested live mode. Historical custom
synthetic scores used a test-aware mock and a different rubric; they are not dev scores.

## Where it fails

Hardened deterministic dev loops have 7.84% precision, 43.24% recall and 13.28% F1:
32 matches among 408 predictions against 74 labels. There are zero false DONE among
matched loops, 17 missed completions among matched loops (47 including undiscovered
ones), and five owner errors. Original deterministic recall was 37.84%, with one false
DONE. The official README warns that its loop labels are incomplete; unmatched
predictions are not all necessarily invalid tasks.

Unseeded circle accuracy is 53.66%. Sparse contacts, dual roles, family/work vocabulary
collisions, implicit obligations and ambiguous unquoted completions remain weak. Dev
goal/intervention error rates cannot be measured. We do not hardcode identities or
duplicate predictions to exploit repeated answer sources. Hidden performance is unknown.

## Engineering decisions and limits

Inference never reads answers. Validated output is written atomically; unresolved
schema/reference errors return nonzero. The client caches requests, bounds concurrency,
reserves estimated cost and handles temporary in-flight-credit holds separately from
permanent credit failures. A Linux watchdog caps execution below 30 minutes. All 76
tests and clean Python 3.10/3.12/3.14 CI passed, including a clean 3.12 container.

A sparse synthetic 20k-message/320-chat run takes 1.19 s with a mock. A real-endpoint
attempt took 4.02 s but all requests were rejected: successful dense-task network
runtime remains unverified. The final relevance-only slow-pass refinement passed
regressions but needs funded live re-evaluation. We retain Dossier-ARR-PRISM rather
than add costly orchestration. Private data and credentials remain ignored. **Not
ready to submit**: packaging is validated, but accuracy and clean model evidence need work.
