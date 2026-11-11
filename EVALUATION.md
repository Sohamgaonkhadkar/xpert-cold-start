# Independent evaluation

## 11 November update

Best verified deterministic dev score: **28.247735639442958/56**, versus the starting
19.799457926752737 and supplied competitor benchmark 27.65. The full local suite passes
(97 tests). No model calls were made tonight: zero tokens and zero API cost. The final
measured pipeline run took 6.7569 seconds. See [the ablation report](metrics/nov11_evaluation.md)
and [machine-readable results](metrics/nov11_evaluations.json). These are development
measurements after iterative tuning, not held-out accuracy or a global leaderboard rank.
No submission has been made. The historical live-model findings below still apply.

## Historical evaluation - 7 November 2026

**Review only; not submitted.** Real-model runs exposed temporary in-flight limits followed by permanent insufficient
account credits; this branch is for review, not submission.
No hidden-set score or ranking is claimed.

## Verified integration and data
Arena PR #1 is merged at `33ebccc237ade137bd76921c57d86aed85342a8b`.
The reviewed Arena head was `7312bb03acdc15a206a29a5218efe290186a676b`.
Improvements are isolated on `codex/final-evaluation-hardening`.
The GitHub connector lacked merge permission; an ordinary Git merge and authenticated
push completed the authorized integration. GitHub confirms PR #1 merged/closed.

Official starter/data were downloaded from the challenge's own links. The checked-in
schema, checker and worked example match after CRLF/LF normalization. Dev contains
4,502 messages, 138 chats and 286 input people. Only 41 unseeded labelled circles and
74 labelled loops contribute to those accuracy categories. Goals and interventions
are unlabelled, so the dev denominator is **56**, not 74 or 100. Answers are used only
by evaluation tooling. Private data and predictions remain ignored.

## Measured results so far

| Run | Circles /12 | Orgs /8 | Chats /8 | Goals /8 | Found /14 | Closure /14 | Help /10 | Total |
|---|---:|---:|---:|---|---:|---:|---|---:|
| Arena deterministic, dev | 5.854 | 2.322 | 6.341 | unlabelled | 1.704 | 2.459 | unlabelled | 18.681/56 |
| Hardened deterministic, dev | 6.439 | 2.322 | 6.341 | unlabelled | 1.859 | 2.838 | unlabelled | 19.799/56 |
| Arena live GPT-4.1 mini, dev | 6.439 | 2.363 | 6.829 | unlabelled | 1.766 | 2.081 | unlabelled | 19.478/56 |
| Hardened live, credit-limited recheck | 6.732 | 2.382 | 6.537 | unlabelled | 1.859 | 2.838 | unlabelled | 20.347/56 |
| Hardened live GPT-4.1 mini, serial, credit-limited | 7.024 | 1.403 | 6.439 | unlabelled | 1.859 | 2.838 | unlabelled | 19.563/56 |
| Hardened live GPT-4o mini, credit-limited | 6.439 | 2.322 | 6.341 | unlabelled | 1.859 | 2.838 | unlabelled | 19.799/56 |
| Official example, deterministic/live | 12 | 8 | 8 | 8 | 14 | 14 | 10 | 74/74 |

The credit-limited recheck had **204 HTTP 402 rejections** and only 18 successful
responses. It is not a clean model comparison. The first hardened live run also
suffered 195 failed attempts. All runs are retained in the aggregate JSON. The serial GPT-4.1 mini run still had
197 HTTP 402s; GPT-4o mini had 181. Neither supports a model ranking. The two serial
runs overlapped briefly on a shared account, so their latency is also confounded.

| Dev mode | Successful responses | Input tokens | Output tokens | Provider USD | USD /1k messages | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Arena deterministic | 0 | 0 | 0 | 0 | 0 | 7.00 |
| Hardened deterministic | 0 | 0 | 0 | 0 | 0 | 8.33 |
| Arena live `openai/gpt-4.1-mini` | 39 | 128022 | 17264 | 0.0788312 | 0.0175103 | 92.07 |
| Hardened live, credit-limited recheck | 18 | 64764 | 9722 | 0.040424 | 0.0089791 | 63.37 |
| Hardened live `openai/gpt-4.1-mini`, serial | 34 | 98795 | 13075 | 0.042390 | 0.0094158 | 789.08 |
| Hardened live `openai/gpt-4o-mini`, serial | 43 | 38565 | 1602 | 0.00674595 | 0.0014984 | 229.88 |

Tokens and cost in live rows are returned by OpenRouter for successful responses;
failed requests can omit usage. These are not complete account billing reconciliations.
Stage token totals and runtimes are in `metrics/official_evaluations.json`. The baseline
live run used unchanged historical inference code; the evaluation harness only added
cost logging. Working-tree runs record their base SHA and must be read with this branch's
changes. Subsequent runner versions also fingerprint inference sources.

## Error analysis
Hardened deterministic loops: 32 matches /408 predictions /74 labels, precision 7.84%,
recall 43.24%, F1 13.28%, zero false DONE among matched loops, 17 missed completions
among matched loops and 47 when undiscovered completions are included, zero false
DROPPED, five matched owner errors. Arena deterministic had 28/386/74, one false DONE,
13 matched missed completions, 46 overall missed completions and five owner errors.
Arena live had two false DONE. False DONE fell to zero in the hardened deterministic
and credit-limited recheck, at the cost of additional missed completions.

The official README says its loop labels are incomplete. Some answer entries share
sources; the checker matches one prediction to one reference. We do not duplicate
loops to exploit this. Unmatched predictions may include genuine tasks omitted from
labels. No raw private messages or identifiable error examples are published.

Missing evidence: dev goals/interventions are unlabelled, so unsupported-goal rates,
false interventions on LEAVE and empirical PRISM calibration cannot be measured.
Synthetic successes do not fill that gap. Sparse contacts, family/work vocabulary
collisions, indirect obligations and ambiguous unquoted completions remain failures.

## Protocol findings and engineering fixes
The evaluator uses OpenRouter. Current catalogue IDs/pricing were checked against its
live model list. The configured `openai/gpt-4.1-mini` smoke test succeeded with 23 input
and 6 output tokens, provider cost $0.0000188. Controlled six-way requests succeeded.
Large production batches encountered OpenRouter's temporary in-flight spending budget,
which also includes recently completed requests. This differs from exhausted account
or key credit. The client now uses metadata and Retry-After, shared cooldown and
serialization for that specific transient 402; permanent 402s still fall back.

Tests exercise authentication, exact endpoint URL, JSON/fenced output, optional parameter
fallback, rate/server/auth failures, budget/deadline checks, missing usage, redirects,
cache, invalid output, missing input/schema, reference integrity and closure guards.
Model response logs never go to external logging services.

## Scale and remaining model blocker
The regenerated mock 20k-message /320-chat fixture completed local compute in 1.187 s,
with 14 mock calls and estimated tokens/cost. The same-size official-format synthetic
fixture was also sent through the real client: 4.025 s, valid schema/references,
**zero successful responses and 12 HTTP 402 rejections**. This is measured network
failure/fallback behavior, not successful live-model scalability. It cannot establish
hidden-set quality or realistic dense-task latency.

OpenRouter ultimately returned `openrouter_credits` with **Insufficient credits** and
stated that the account had never purchased credits. A large per-key spending limit
does not establish funded account balance. No account settings were changed and no
credits were purchased. A funded permitted endpoint is required for the remaining
clean comparisons. The final temporal slow-pass gate now avoids escalating unrelated
later turns; it passed offline regressions but could not be re-evaluated with a fully
available model after this blocker. Neither tested model is selected for grading yet.

The Linux entrypoint now adds a 1740-second watchdog (when coreutils timeout is present),
with a five-second kill grace and meaningful nonzero timeout exit, in addition to the
client's shorter deadline. It guards hangs; it does not prove useful predictions will
finish under all hidden workloads.

## Fair public comparison
On the same official dev set, we locally reproduced **27.646/56**, 17.67% loop precision
and 51.35% recall for CodeInfinity1/cold-start at
`4b369b007364520bcbbff9bd60ca83a3cd940a58` (3 November). It used rules only, zero
model calls and approximately 0.4 seconds. Its performance exceeds ours on this set.
No competitor code was incorporated. Remycooks/Xpert_Challenge remains at
`adfb741f01601ead21ebf7d9da8f40f7e133bdfa` (4 November); its design review remains
qualitative. A fresh GitHub repository search found no additional populated match.

## Reproducibility and privacy
A fresh Windows virtual environment installed all declared dependencies. The official
shell entrypoint works from a different current directory and with spaces, and the
public example also scores 74/74 with `python -S` (stdlib only). All 76 tests passed locally. Clean Linux CI passed on Python 3.10, 3.12 and 3.14,
and the additional `python:3.12-slim` container passed installation, all tests, the
required two-argument shell command, official scoring and stdlib-only execution.
[Verified CI run](https://github.com/Sohamgaonkhadkar/xpert-cold-start/actions/runs/42367659348).
The official dev output is exactly unchanged by JSONL line-order shuffling. The local
Docker daemon was unavailable, so container verification used GitHub Actions.

Tracked files and all four pre-hardening history commits showed no private dataset
paths or common API-key/private-key patterns. This is a scoped scan, not a proof that
no possible secret format exists. `.env`, `*.env`, datasets, private reports, responses,
usage logs, virtual environments and caches are ignored. Repository visibility was
not changed. Original Arena session transcripts were unavailable; the genuine summary
is retained and labelled as a summary. Visible Codex messages and the mission prompt
are preserved, excluding private tool payloads and internal reasoning.

## Final review verdict
**NOT READY TO SUBMIT**: measured dev accuracy trails the reproduced public baseline;
model runs are fallback-heavy, the final model routing refinement needs a funded
endpoint comparison, goal/intervention calibration lacks labels, and dense hidden-set
network runtime remains unverified. Packaging and offline execution are validated.
The final improvement branch has not been merged and the challenge has not been submitted.

## Repository review state

Improvements are pushed on `codex/final-evaluation-hardening`; the main branch remains
at the authorized Arena merge. The working tree is verified clean at delivery.
The GitHub connector returned 403 when asked to create a draft PR, so no improvement
PR was created. The pushed branch is directly reviewable; it has not been merged.

Changed files relative to the integrated main branch:

```text
.gitattributes
.github/workflows/ci.yml
.gitignore
ARCHITECTURE.md
AUDIT.md
EVALUATION.md
INTERVIEW.md
README.md
WRITEUP.md
ai-logs/README.md
ai-logs/codex-visible-transcript.md
ai-logs/mission-prompt.md
ai-logs/session-summary.md
metrics/experiment_results.json
metrics/official_evaluations.json
metrics/writeup.pdf
run.sh
scripts/correction_demo.py
scripts/evaluate_official.py
scripts/live_evaluation.py
scripts/render_writeup.py
scripts/run_experiments.py
scripts/scale_fixture.py
src/cli.py
src/corrections.py
src/data_loader.py
src/evidence_validator.py
src/intervention_policy.py
src/llm_client.py
src/loop_extraction.py
src/output_builder.py
src/relationship_inference.py
src/schema_adapter.py
src/submission_validator.py
src/temporal_resolver.py
tests/conftest.py
tests/test_bottlenecks_and_multilingual.py
tests/test_submission_hardening.py
```

## Sources
- Official challenge and FAQ: https://goals.xpert.chat/w/intern
- Official downloads: https://goals.xpert.chat/w/intern/starter and https://goals.xpert.chat/w/intern/data
- OpenRouter credit-limit semantics: https://openrouter.ai/docs/api/reference/limits
- Model catalogue: https://openrouter.ai/api/v1/models
- Competitor checkpoint: https://github.com/CodeInfinity1/cold-start/commit/4b369b007364520bcbbff9bd60ca83a3cd940a58
