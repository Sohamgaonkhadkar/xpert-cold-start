# Xpert Cold Start Challenge — `Dossier-ARR-PRISM`

A fast, evidence-grounded, cost-cascaded memory and proactive intervention engine for the **Xpert Winter Internship 2026 Cold Start Challenge**.

Given ~8 weeks of multilingual WhatsApp conversations (`chats.jsonl`, `messages.jsonl`) across English, Hindi, Bengali, and Hinglish plus partial `seed.json` context, `Dossier-ARR-PRISM` infers:
1. **People & Circles** (`12` circles, user organisation `runs | supports` links, and `staff` individual care links),
2. **User Organisations & Chat Ownership** (separating home/work domains while excluding external client companies and routing person-centric groups such as a child's class group to that person),
3. **Personal Goals** ($\le 10$ evidence-grounded `habit | milestone | experience` goals across `10` life areas and `4` horizons),
4. **Open Loops & Cross-Chat Closure** (`commitment | followup | deadline | question`, owner, relative due-date anchored to `msg.ts`, final `open | done | dropped` status with Asymmetric Closure Guards), and
5. **Where to Help (`assist | remind | leave`)** via a PRISM expected-loss decision gate calibrated for asymmetric intervention penalties, plus an isolated **live-interview user correction engine**.

---

## Quickstart

### 1. One-Command Execution (Official Contract)
```bash
bash run.sh <input_folder> <output_path>
```
`run.sh` works out-of-the-box in a clean container using `.venv/bin/python` when available or falling back cleanly to system `python3` (zero mandatory non-stdlib runtime dependencies; `httpx` and `jsonschema` are used automatically when installed).

### 2. Configure the Evaluator OpenAI-Compatible Model Endpoint
```bash
export LLM_BASE_URL="https://api.openai.com/v1"   # or evaluator-provided OpenAI-compatible URL (falls back to OPENAI_BASE_URL)
export LLM_API_KEY="sk-..."                       # falls back to OPENAI_API_KEY
export LLM_MODEL="gpt-4.1-mini"                   # supports chat & reasoning models (auto max_completion_tokens)
# Optional container budget ceilings (defaults: 1200s wall-clock, $5.00 max cost):
# export LLM_MAX_WALL_SEC=1200 LLM_MAX_COST_USD=5.00
bash run.sh /path/to/data /path/to/output.json --telemetry /path/to/telemetry.json
```
*Privacy & Network Guarantee:* Runtime outbound network traffic is strictly limited to `LLM_BASE_URL`. No external embedding APIs, vector databases, or third-party web requests are ever made. If `LLM_API_KEY` is unset, the pipeline degrades gracefully to its deterministic Tier-0 mode and still emits 100% schema-valid output.

### 3. Live-Interview User Corrections (Fast Isolated Cascade)
During a live review/interview, corrections to a person's circle, organisation membership, chat ownership, loop owner, or loop status can be applied without re-running unrelated predictions:
```bash
# Build or append corrections via CLI flags or natural-language statements:
python3 -m src.corrections --file corrections.json --person-circle p_colleague_2 partner
python3 -m src.corrections --file corrections.json --chat-owner c_grp_friends organisation org_zephyr
python3 -m src.corrections --file corrections.json --loop-owner loop_017 p_colleague_1
python3 -m src.corrections --file corrections.json --say "loop_009 is done via m102"

# Apply corrections cleanly during run.sh execution:
bash run.sh <input_folder> <output_path> --corrections corrections.json
```

---

## Official `starter.zip` Schema, Scorer (`check.py`) & `$LLM_USAGE_LOG` Parity
* Includes the official `starter.zip` `schema.json`, `check.py` validator/74-point scorer, and worked `example/` dataset (`Meera`).
* Verified **`74.0 / 74` (`1.00` across all 7 categories, `0` schema/semantic errors)** on `example/` in **both seeded and `0%`-seed** modes:
  ```bash
  bash run.sh example /tmp/ex_out.json && python3 check.py example /tmp/ex_out.json
  ```
* `src/llm_client.py` appends `{model, prompt_tokens, completion_tokens, seconds}` to `$LLM_USAGE_LOG` on live calls and defaults to `usage.jsonl` when the variable is unset.

## Competitor Checkpoint & Pass-6 Hardening (2026-11-07)
A fresh GitHub API check found no newer public Xpert Cold Start solution:
* [`CodeInfinity1/cold-start`](https://github.com/CodeInfinity1/cold-start) was last pushed **Nov 3**; its latest code was previously measured at **50.6/74** on the official example and **27.6/56** on public `data/dev`.
* [`Remycooks/Xpert_Challenge`](https://github.com/Remycooks/Xpert_Challenge) was last pushed **Nov 4**; its v1.1 rules solution leaves goals and open loops empty.
* [`dipgarg0011/Xpert-Cold-Start`](https://github.com/dipgarg0011/Xpert-Cold-Start) remains an empty repository. No additional matching submission surfaced in the repository search.

Competitor-informed additions now cover WhatsApp numeric `@` JID/LID resolution, place-named staff, CamelCase org acronyms and active core-member overlap, self-only Notes-to-Self chats, departed-sender/loop-owner guards, media and birthday-noise suppression, document closure evidence, slash dates, and Hinglish `tmrw`/`batao`/`lmk`. The updated 20k-message/320-chat audit completes in **1.04s** with **14,664 prompt tokens**; all **51 tests pass**, and the official example remains **74.0/74**.

---

## Repository Structure
```text
├── run.sh                              # Official entrypoint: bash run.sh <input_folder> <output_path>
├── check.py                            # Official starter.zip stdlib validator & 74-point scorer
├── schema.json                         # Official starter.zip JSON Schema (strict additionalProperties: false)
├── example/                            # Official starter.zip worked dataset + answers/answers.json (74.0 / 74)
├── requirements.txt / pyproject.toml   # Minimal pure-Python dependencies (httpx, jsonschema, pytest)
├── ARCHITECTURE.md                     # Architecture Decision Record (literature review + 4-way trade-off matrix)
├── WRITEUP.md                          # <=2-page submission write-up (approach, ablations, failures, $/1k msgs)
├── metrics/experiment_results.json     # Machine-readable results from Controlled Experiments 0..5 & robustness suites
├── ai-logs/session-summary.md          # Engineering & AI steering log (hypotheses, bugs caught, verifications)
├── src/
│   ├── models.py                       # Official enums, dataclasses, and append-only EvidenceLedger
│   ├── indic_normalizer.py             # Zero-ASCII-overhead Devanagari (Hindi) & Bengali Unicode normalizer
│   ├── data_loader.py                  # Timezone-preserving, order-invariant JSONL & seed.json parser
│   ├── conversation_index.py           # Dynamic ME-alias inference + two-phase inverted BM25 & cached char n-gram index
│   ├── llm_client.py                   # OpenAI-compatible client (pooling, reasoning-model auto-downgrade, $/1k caps)
│   ├── relationship_inference.py       # Contact Dossiers, conditioned seed propagation, staff care-target inference
│   ├── organisation_inference.py       # User org discovery, client-company exclusion, person-vs-org chat routing
│   ├── goal_extraction.py              # Precision-first <=10 evidence-backed goal extraction
│   ├── loop_extraction.py              # Multilingual loop candidate discovery, msg.ts date resolver, cross-chat dedup
│   ├── temporal_resolver.py            # ARR cross-chat trail retrieval & 4 Asymmetric Closure Guards
│   ├── intervention_policy.py          # PRISM expected-loss gate (assist | remind | leave + needs_approval floor)
│   ├── corrections.py                  # Structured + NL live-interview correction parser & isolated cascade
│   ├── evidence_validator.py           # Deterministic referential, temporal, and enum invariant repair
│   ├── schema_adapter.py               # Dynamic schema-introspecting serializer & JSON Schema validator
│   ├── output_builder.py               # End-to-end pipeline orchestrator & telemetry collector
│   └── cli.py                          # CLI driver invoked by run.sh
├── scripts/
│   ├── checker.py                      # Local official-rubric scorer (86 automated + 7 execution points)
│   ├── baseline.py                     # Naive baseline (Experiment 0) for controlled comparison
│   ├── generate_synthetic_suite.py     # Multilingual 20-adversarial-case WhatsApp benchmark generator
│   └── run_experiments.py              # Controlled ablations (Exp 0..5) + 5 robustness audits
└── tests/
    ├── test_adversarial_20.py            # End-to-end tests for all 20 adversarial WhatsApp failure modes
    ├── test_bottlenecks_and_multilingual.py # Native Devanagari/Bengali Unicode, Whisper ASR fuzzy join, 8-week 20k scale tests
    ├── test_official_starter_and_competitor_traps.py # Official check.py 74/74 (seeded & 0%-seed), $LLM_USAGE_LOG, Marathi/Tamil & competitor traps
    ├── test_redteam_audit_fixes.py       # Hidden-persona rename, IST midnight TZ, bystander guards, reasoning-model & scale tests
    ├── test_robustness_and_validators.py # Order shuffle, temporal truncation, empty seed, validator & schema alias tests
    ├── test_corrections_and_cli.py       # All 5 live-interview correction types, cascade isolation, and run.sh CLI tests
    └── test_pass6_whatsapp_edges.py     # Departed senders, self-chats, numeric mentions, media, dates, org overlap, and place-named staff
```

---

## Reproducing Tests & Benchmarks
```bash
# 1. Install dependencies (optional if using bare stdlib Tier-0 mode)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Validate & score against the official starter.zip check.py (74.0 / 74)
bash run.sh example /tmp/ex_out.json && python3 check.py example /tmp/ex_out.json

# 3. Run all 51 automated official-starter, adversarial, multilingual Unicode, ASR, red-team, 20k-scale, and CLI tests
.venv/bin/pytest -v

# 4. Reproduce Controlled Experiments 0..5 + Robustness & 20k-Scale Audits -> metrics/experiment_results.json
.venv/bin/python scripts/run_experiments.py
```
