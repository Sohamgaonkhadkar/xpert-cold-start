# Independent evaluation - 7 November 2026

**NOT READY TO SUBMIT.** Genuine development performance remains weak. Further runs
with provider-aware cooldown are in progress; this branch is for review, not submission.
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
| Official example, deterministic/live | 12 | 8 | 8 | 8 | 14 | 14 | 10 | 74/74 |

The credit-limited recheck had **204 HTTP 402 rejections** and only 18 successful
responses. It is not a clean model comparison. The first hardened live run also
suffered 195 failed attempts. Both runs are retained in the aggregate JSON.

| Dev mode | Successful responses | Input tokens | Output tokens | Provider USD | USD /1k messages | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Arena deterministic | 0 | 0 | 0 | 0 | 0 | 7.00 |
| Hardened deterministic | 0 | 0 | 0 | 0 | 0 | 8.33 |
| Arena live `openai/gpt-4.1-mini` | 39 | 128022 | 17264 | 0.0788312 | 0.0175103 | 92.07 |
| Hardened live, credit-limited recheck | 18 | 64764 | 9722 | 0.040424 | 0.0089791 | 63.37 |

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
public example also scores 74/74 with `python -S` (stdlib only). Linux CI is configured
but not yet verified. Docker is installed but its Linux daemon was not running.

Tracked files and all four pre-hardening history commits showed no private dataset
paths or common API-key/private-key patterns. This is a scoped scan, not a proof that
no possible secret format exists. `.env`, `*.env`, datasets, private reports, responses,
usage logs, virtual environments and caches are ignored. Repository visibility was
not changed. Original Arena session transcripts were unavailable; the genuine summary
is retained and labelled as a summary. Visible Codex messages and the mission prompt
are preserved, excluding private tool payloads and internal reasoning.

## Sources
- Official challenge and FAQ: https://goals.xpert.chat/w/intern
- Official downloads: https://goals.xpert.chat/w/intern/starter and https://goals.xpert.chat/w/intern/data
- OpenRouter credit-limit semantics: https://openrouter.ai/docs/api/reference/limits
- Model catalogue: https://openrouter.ai/api/v1/models
- Competitor checkpoint: https://github.com/CodeInfinity1/cold-start/commit/4b369b007364520bcbbff9bd60ca83a3cd940a58
