# Independent audit — 7 November 2026

Baseline: Arena `7312bb03acdc15a206a29a5218efe290186a676b`, merged as
`33ebccc237ade137bd76921c57d86aed85342a8b`. All 51 tests passed with Git Bash
outside the Windows sandbox; the sandbox itself prevents MSYS initialization.
Official downloads: https://goals.xpert.chat/w/intern/starter and `/w/intern/data`.
Schema, checker and all six worked-example files match after newline normalization.

## Architecture and evidence

* Loader/index: immutable timestamp-sorted turns, speaker IDs, seed anchors, local
  multilingual lexical/fuzzy retrieval. Inference never reads answers.
* Relationships: additive lexical/contact/group scores, then uncertain dossiers
  batched to the model. Seeds override inference. Sampling mostly takes first/last
  DM turns; confident lexical mistakes can bypass model verification.
* Organisations/chats: seed and group/contact topology rules, selective model
  routing. Company vocabulary can misclassify external client groups and sparse
  family contacts. Home/office/vacation staff classification needs evidence.
* Goals: ME endorsement and evidence gate, optional model verifier, ten-goal cap.
  Dev goals are unlabelled: no accuracy/calibration claims are possible here.
* Loops/temporal: regex candidates, identity/topic deduplication, local and
  cross-chat trails, model boundary verification. No bare acknowledgement may
  establish task execution. Most real exports have no quoted replies.
* Interventions: hand-set probabilities and asymmetric costs, followed by optional
  boundary model calls. These are design assumptions, not calibrated probabilities;
  dev intervention labels are absent.
* Corrections: structured/NL overrides after inference, affected-loop cascade,
  deterministic repair. Ambiguous name resolution currently takes the first match.
* Client: OpenAI-compatible HTTP only; caching, batched concurrency, retry and
  estimated-cost telemetry. Budget checked only before a retry loop, so retries
  and concurrent in-flight calls can exceed the intended limits.
* Validation: repair precedes schema adaptation, but schema errors only become
  warnings; missing schema silently disables validation; stdlib fallback checks
  only top-level keys. No final independent serialized reference validation.

## Measured baseline and priorities

Official example: 74/74, deterministic, 29 messages. Official dev: **18.6811/56**,
4,502 messages / 138 chats, 7.00 s, zero calls/tokens/cost. Goals and interventions
are unlabelled. Loops: 28/74 found among 386 predictions (precision 7.25%, recall
37.84%); one false DONE, 13 missed completions among matched loops, five owner
errors. Some answer loops share the same source; do not duplicate predictions to
exploit that. The official README describes loop labels as incomplete.

1. Fail closed on unresolved schema/reference errors; fix shell path semantics.
2. Remove inference's guessed final-14-day filter. Official FAQ says enumerate all
   loops; only the checker knows the fixed scoring start. Preserve earlier context.
3. Improve specific extraction/relationship failures with identity-independent tests,
   checking both worked example and genuine dev; avoid fitting uncertain labels.
4. Exercise real HTTP protocol failures, retries/deadline/concurrency budgets and
   telemetry with controlled transport; this cannot validate live model quality.
5. Separate synthetic/mock results from measured official results in documentation.

## Test evidence boundaries

Existing tests cover units, official example, synthetic integrations, mock models,
renaming/order/seed ablations, multilingual input, quoted replies, timezone edges,
cross-chat closure, bystanders, corrections and synthetic 20k scale. None establishes
live endpoint quality, actual dev performance, actual network latency or a Linux
clean install. No model credentials were present at audit time.

No source overhaul or intervention recalibration is justified without further
evidence. Preserve Dossier–ARR–PRISM and conservative closure safeguards.
