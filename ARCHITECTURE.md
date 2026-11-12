# Architecture: Dossier–ARR–PRISM

This describes the current implementation. Historical research notes and superseded
claims remain recoverable in Git history; they are not performance evidence. See
[EVALUATION.md](EVALUATION.md) for measurements.

## Execution contract

`run.sh` invokes `src.cli` with an input folder and output path. The default profile
is deterministic even when model credentials are present; `--mode hybrid` explicitly
enables selective model inference. Output follows the official five-section JSON
format. No database or background service is needed.

## Evidence and indexing

`data_loader.py` loads chats, messages and optional seed records, normalizes accepted
timestamps and constructs typed records. `conversation_index.py` preserves original
turns and builds chronological histories, participant/DM indexes, name aliases,
reply relationships and lexical retrieval indexes. `indic_normalizer.py` supplies
matching equivalents for supported Indic phrases while retaining original evidence.

The index is separate from derived predictions. Source IDs survive extraction,
reconciliation and serialization. Reference answers are consumed only by evaluation
tools, never by inference.

## Dossiers and organisations

`relationship_inference.py` combines saved names, seeded circles, small-group context
and speaker-attributed turns into contact dossiers. Seeds take precedence. Rules
distinguish internal work groups from customer projects and constrain family priors
so household conversation volume cannot override an inferred parent or child.
ME’s direct full-name family introductions can anchor a relationship across chats; ambiguous or conflicting introductions are ignored. Hybrid mode can refine uncertain dossiers.

`organisation_inference.py` discovers associated companies and places, links people
and assigns every chat to one person or organisation. Companies and their office
workspace are distinct entities. Client businesses are not automatically promoted
into the user's organisations. Sparse contacts and dual roles remain difficult.

## Goal extraction

`goal_extraction.py` looks for targets, recurring rhythms and planned experiences.
Evidence must show the user's own intent or endorsement. Rules and optional model
verification filter weak wishes and unrelated third-party goals; output is capped
at ten. Dev has no goal labels, so these gates are not independently measured accuracy.

## ARR: Anchor–Retrieve–Reconcile

1. **Anchor:** `loop_extraction.py` identifies requests, promises, decisions and dated
   obligations, determines an initial owner, resolves dates against message time and
   deduplicates related mentions. Multi-action lists stay separate from narrower
   tasks. Unaddressed group requests require grounding beyond membership.
2. **Retrieve:** `temporal_resolver.py` collects chronological local turns and related
   cross-chat evidence. Cross-chat matching requires shared participants and task
   content; a shared first name is insufficient.
3. **Reconcile:** evidence updates ownership, due dates and `open`/`done`/`dropped`
   status. Completion, matching deliverables, cancellation and later failure evidence
   can change state. A substantive, on-topic response can fulfil a promise to reply without asserting that an underlying repair is complete. Hybrid mode verifies selected uncertain trails.

Closure safeguards apply to both rules and model output. Bare acknowledgements,
future intent, partial progress and elapsed dates do not establish execution.
An individual completed or cancelled item cannot resolve an entire task list.
The engine enumerates all discovered history; only the checker applies its private
scoring-window cutoff.

## PRISM intervention policy

`intervention_policy.py` chooses `assist`, `remind` or `leave` using hand-set class
weights and an asymmetric loss matrix. These are assumptions, not calibrated
probabilities. Decisions use ownership, relationship, task content and due evidence.
An undated task does not become overdue merely by age. Someone else's task is
generally left alone; an overdue dependency the user is waiting on may warrant a
reminder. External actions require approval.

Dev has no intervention labels. Policy changes are therefore checked against
structural requirements and synthetic counterexamples, not claimed dev accuracy gains.

## Corrections and validation

`corrections.py` applies scoped user changes after inference. Exact IDs are preferred;
ambiguous names fail. Closed-status corrections need later evidence, and affected
interventions can be recomputed without changing unrelated predictions.

`evidence_validator.py` performs safe repairs. `schema_adapter.py` serializes and
validates the contract with `jsonschema` or an internal stdlib validator.
`submission_validator.py` checks IDs, owners, evidence and temporal references.
`output_builder.py` writes atomically only after unresolved errors are rejected.

## Optional model transport

`llm_client.py` provides OpenAI-compatible HTTP, in-process caching, bounded
concurrency, retries, deadline/cost admission and usage telemetry. It refuses
redirects and distinguishes temporary credit holds from permanent exhaustion.
Estimated costs and provider-reported charges are recorded separately.

The grading profile makes no model calls. Earlier funded runs returned real responses
but suffered credit failures; they do not verify current hybrid quality. Protocol/mock
tests establish transport behavior, not real-model accuracy.

## Trade-offs

This design keeps evidence explicit, avoids database deployment, preserves an offline
path and limits model work to uncertain cases when requested. Conservative closure
reduces the risk of hiding unfinished work. Costs include incomplete linguistic
coverage, ambiguous relationships/organisation membership and tasks whose completion
was never stated in the available history.
