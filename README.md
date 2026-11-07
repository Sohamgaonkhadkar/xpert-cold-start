# Xpert Cold Start: Dossier-ARR-PRISM

Python batch inference for the [Xpert Winter Internship Cold Start Challenge](https://goals.xpert.chat/w/intern).
Preserves the Arena implementation: contact dossiers, organisation/chat routing, multilingual
retrieval, Anchor-Retrieve-Reconcile task tracking and a conservative intervention policy.

## Run

```bash
python3 -m pip install -r requirements.txt
bash run.sh <input_folder> <output_path>
```

The input contains `chats.jsonl`, `messages.jsonl` and optional `seed.json`.
Input/output paths are relative to the caller, may contain spaces and may be outside
the repository. Inference never reads `answers/`. No database, frontend, model download
or general internet access is required. Runtime network requests go only to the configured
model endpoint; redirects are refused. Without credentials, inference uses deterministic
rules. The stdlib fallback validates every keyword in the official schema.

For the grader's OpenRouter configuration:

```bash
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_API_KEY='<your key>'
export LLM_MODEL=openai/gpt-4.1-mini
export LLM_USAGE_LOG=/tmp/xpert-usage.jsonl
bash run.sh data/dev /tmp/xpert-output.json --telemetry /tmp/xpert-telemetry.json
python3 check.py data/dev /tmp/xpert-output.json
```

`openai/gpt-4.1-mini` has been live-tested. See [EVALUATION.md](EVALUATION.md) for
actual quality, costs, provider failures and model comparison; the project is not yet
claimed competitive. The evaluator must receive the exact model ID actually selected
from those measurements.

Optional limits: `LLM_MAX_WALL_SEC` (default 1380), `LLM_MAX_COST_USD` (default 2.50),
`LLM_MAX_CONCURRENCY` (default 6), `LLM_INPUT_COST_PER_1M` and `LLM_OUTPUT_COST_PER_1M`.
Costs used for admission are estimates; provider billing, when returned, is separately
recorded in telemetry. Failed requests can omit billable usage. For small OpenRouter
accounts, start at concurrency 1. The client handles transient in-flight-credit 402s
using provider metadata and cooldown, without treating exhausted key/account credit
as a transient condition. This does not guarantee every request will succeed.

Malformed/missing input, missing schema and unresolved output errors return nonzero.
Safe deterministic repairs precede strict schema and reference validation. The output
file is replaced atomically only after validation succeeds.

## Reproduce evaluations

Download [official data](https://goals.xpert.chat/w/intern/data) and
[starter kit](https://goals.xpert.chat/w/intern/starter) directly. Unzip the dataset
inside the repository to create `data/dev/`; it is ignored by Git. Do not commit or
share the data, and delete it when the challenge ends as the organisers require.

```bash
python3 -m pytest -q
bash run.sh example /tmp/example.json
python3 check.py example /tmp/example.json
python3 -m scripts.evaluate_official example /tmp/example.json
python3 -m scripts.evaluate_official data/dev /tmp/xpert-output.json --report private-reports/dev.json
# Reads only the three named variables from an explicit local file; never echoes the key:
python3 -m scripts.live_evaluation data/dev private-reports/live-run --env-file /path/to/private.env
# Historical Arena source, without changing the worktree or making another Git repo:
python3 -m scripts.live_evaluation data/dev private-reports/arena-run --env-file /path/to/private.env --source-ref 7312bb03acdc15a206a29a5218efe290186a676b
```

Use a fresh output directory for each live run. Private predictions and usage logs stay
in ignored folders. Only aggregate reports without conversation text belong in `metrics/`.
Tests clear model credentials so they cannot spend API credits accidentally.

The official example scores /74. Dev has no goal/intervention labels and scores /56.
The historical custom synthetic benchmark uses a different /93 rubric and a test-aware
mock responder. Its estimated tokens, costs and local runtime are **not** live-model
measurements. Run it separately with `python3 -m scripts.run_experiments`; it writes
`metrics/experiment_results.json`. Its results must never be compared as leaderboard scores.

## Correction interview

See [INTERVIEW.md](INTERVIEW.md) for a short reproducible demo. Structured corrections
support people, organisations, chat ownership and loop owner/status, with isolated
intervention recomputation. Prefer exact person/loop/source IDs; ambiguous names fail
clearly. A DONE/DROPPED correction requires a later evidence message.

## Review material

- [AUDIT.md](AUDIT.md): baseline architecture, specific risks and test boundaries.
- [EVALUATION.md](EVALUATION.md): measured official results and readiness verdict.
- [WRITEUP.md](WRITEUP.md): submission write-up; [rendered PDF](metrics/writeup.pdf), at most two A4 pages.
- [ARCHITECTURE.md](ARCHITECTURE.md): historical design rationale, not performance evidence.
- `ai-logs/`: original Arena summary, genuine mission prompt and exported visible Codex conversation.
  No original Arena transcript was available; the summary is not presented as one.
- `.github/workflows/ci.yml`: offline Linux install, regressions, official example and path checks.

Private challenge data, model responses, credentials and runtime output must remain
untracked. Repository visibility remains unchanged. The final improvement branch and
challenge submission require the user's approval before merging/submitting.
