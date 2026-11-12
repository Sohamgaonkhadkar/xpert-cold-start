# Evaluation

This report records reproducible measurements of Dossier–ARR–PRISM. Results are
separated by dataset, execution mode and source version. Development results were
obtained after iterative tuning; they are not held-out accuracy or a leaderboard rank.

## Dataset and scoring

The official development set contains **4,502 messages, 138 chats and 286 people**.
The checker scores 41 unseeded relationship labels and 74 reference loops. Goals and
intervention decisions are unlabelled, so the measurable denominator is **56**.
The public worked example contains labels for all seven categories and scores /74.
Neither result is a full /100 competition score.

The official `check.py`, `schema.json` and public worked-example files were compared
with the starter kit on 12 November 2026 and matched after newline normalization.
Reference answers are read by evaluation tools only. Inference consumes the input
chats, messages and optional seed, without consulting `answers/`.

## Current deterministic results

Recorded on **12 November 2026**, using the default two-argument command. Supplying
model credentials does not change this profile; model-assisted inference requires
`--mode hybrid`.

| Category | Points |
|---|---:|
| Circles | 9.3659 / 12 |
| Organisations | 3.7612 / 8 |
| Chat ownership | 6.4390 / 8 |
| Open loops found | 4.0453 / 14 |
| Open loops closed correctly | 6.0541 / 14 |
| Goals | Unlabelled |
| Intervention decisions | Unlabelled |
| **Total measurable on dev** | **29.6654 / 56** |

Exact total: `29.665430043043532`. Aggregate category scores, telemetry and source
fingerprints are retained in [offline_accuracy_2026-11-12.json](metrics/offline_accuracy_2026-11-12.json). The earlier submission-profile verification remains in [submission_verification.json](metrics/submission_verification.json).
The [11 November ablations](metrics/nov11_evaluation.md) and
[12 November corrections](metrics/nov12_evaluation.md) document the measured changes.

## Offline evidence refinements

The starting `main` revision (`187ba66`) was rechecked at 28.9526/56 with all 125
existing tests passing. The following changes use conversation evidence without
reading reference labels during inference. No paid API was called for this evaluation.

| Cumulative change | Official dev /56 |
|---|---:|
| Starting verified configuration | 28.9526 |
| Direct full-name family introductions across chats | 29.2453 |
| Prevent business-name "House" collisions with residence routing | 29.4082 |
| Recognize a substantive answer fulfilling a reply promise | 29.5974 |
| Recognize dated collaborative meeting proposals | **29.6654** |

The family rule requires ME's direct introduction, an unambiguous full name and no
conflicting explicit relationship; seeds remain authoritative. The reply rule requires
an adjacent, on-topic owner response in a DM, bounded in time, and rejects deferrals,
partial progress, unrelated quotations and generic acknowledgements. It completes
only the communication promise, not an underlying repair. A completed preparation
statement does not complete the subsequent meeting proposal.

Broader bridge-group classification scored 29.4635 and was reverted. Numeric-mention
alternatives scored 29.5868 and 29.6533 and were also reverted; one recovered an extra
reference source but reduced the overall official score. This does not establish that
its additional predictions were false, because the reference list is incomplete.

The 74 reference loops contain 63 distinct accepted-source sets. The starting 19
matched missed-completion labels all have empty accepted-closing-message lists.
These facts do not prove an absence of evidence in the history. They do limit direct
supervision: repeated predictions and expired-date completion rules were not used to
fit the labels. The retained version has 23 unmatched reference entries and one fewer
missed completion overall.

## Error analysis

| Loop metric | Result |
|---|---:|
| Predictions | 279 |
| Matched reference loops | 51 / 74 |
| Label-based precision | 18.28% |
| Label-based recall | 68.92% |
| Label-based F1 | 28.90% |
| False DONE among matched loops | 0 |
| False DROPPED among matched loops | 0 |
| Missed completions among matched loops | 19 |
| Missed completions including undiscovered loops | 40 |
| Owner disagreements among matched loops | 6 |

Nine of the 41 unseeded circle labels are incorrect. Sparse contacts, family/work
vocabulary collisions, dual roles and ambiguous organisation membership remain
significant weaknesses. Informal requests and completion without explicit confirmation
also remain difficult.

The official loop labels are incomplete. An unmatched prediction is therefore not
automatically a false task. Some accepted origins are fragments or repeated sources,
and some completed labels lack visible later evidence. No identity-specific rules,
answer lookup, duplicate predictions or elapsed-time completion rules were added to
fit these cases. Zero matched false DONE does not establish a zero false-completion
rate over all predicted tasks.

Goals and intervention error rates cannot be measured on this dev set. Synthetic
counterexamples check structural behavior, but do not provide empirical calibration
for PRISM's hand-set decision weights.

## Runtime and cost

| Measurement | Result | Interpretation |
|---|---:|---|
| Official dev, deterministic | 4.31 s | One local measured pipeline run |
| Model calls / tokens | 0 / 0 | Default deterministic profile |
| Model inference cost | $0 | No provider requests |
| Dense synthetic fixture | 49.05 s | 20,000 messages, 320 chats, 5,000 designed task cases |
| Dense synthetic validation | 5,000 discovered and completed; no schema errors | Designed cases, not hidden-set quality |

Runtime varies by machine and workload. The synthetic test checks local computation
and output validity, not realistic network latency or general task accuracy. The
Linux launcher applies a 1,740-second watchdog when `timeout` is available, while
optional model work has a shorter client deadline.

## Historical model experiments

These measurements use earlier source versions. They are retained as cost and
failure evidence, not as validation of the current hybrid implementation.

| Earlier official-dev run | Score /56 | Successful responses | Input / output tokens | Provider USD | Seconds |
|---|---:|---:|---:|---:|---:|
| Original deterministic implementation | 18.681 | 0 | 0 / 0 | 0 | 7.00 |
| Earlier hardened deterministic implementation | 19.799 | 0 | 0 / 0 | 0 | 8.33 |
| Original live `openai/gpt-4.1-mini` | 19.478 | 39 | 128,022 / 17,264 | 0.0788312 | 92.07 |
| Hardened live `openai/gpt-4.1-mini`, credit-limited | 19.563 | 34 | 98,795 / 13,075 | 0.042390 | 789.08 |
| Hardened live `openai/gpt-4o-mini`, credit-limited | 19.799 | 43 | 38,565 / 1,602 | 0.00674595 | 229.88 |

Live runs returned genuine model responses but encountered HTTP 402 credit failures
and deterministic fallbacks. Some runs briefly overlapped, confounding latency.
They do not support a clean model ranking or a claim that the current hybrid path
improves quality. Provider costs cover reported successful responses; failed calls
can omit usage, so these are not complete account billing reconciliations.

Detailed historical run records and stage token counts are in
[official_evaluations.json](metrics/official_evaluations.json). Mock-based synthetic
scores use a different rubric and are not compared with official results.

## Reproduction

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q

bash run.sh example outputs/example.json
python3 check.py example outputs/example.json

bash run.sh data/dev private-reports/dev.json \
  --telemetry private-reports/dev-telemetry.json
python3 check.py data/dev private-reports/dev.json
python3 -m scripts.evaluate_official data/dev private-reports/dev.json \
  --report private-reports/dev-report.json
```

The development dataset must be obtained from the official source and kept untracked.
Private predictions, credentials and model responses belong in ignored local folders.
Only aggregate measurements are published here.

## Validation coverage

The recorded regression suite has **156 passing tests**. Linux CI covers Python
3.10, 3.12 and 3.14 plus a clean Python 3.12 container. Checks include the official
worked example and zero-seed variant (74/74), stdlib-only inference, caller-relative
paths with spaces, credential-present deterministic execution and mocked hybrid
transport. The correction demonstration checks relationship, ownership and status
changes while unrelated predictions remain stable.

Clean-archive checks reproduced the deterministic dev output with inputs outside
the repository, no reference-answer files and dummy evaluator credentials present.
The public example also passed using `python -S` without third-party packages.

Protocol tests cover JSON parsing, optional parameter fallback, endpoint construction,
rate/server/authentication failures, concurrency, budgets, timeouts, redirects and
usage accounting. These establish transport behavior, not real-model accuracy.

A scoped scan of 130 unique historical Git blobs found no common credential patterns
or private dataset paths. Ignore rules exclude credentials, datasets, environments,
model responses and runtime outputs. Pattern-based scanning cannot prove the absence
of every possible secret format. Available AI-development records and their provenance
are described in [ai-logs/README.md](ai-logs/README.md).
