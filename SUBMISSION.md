# Submission handoff

The verified grading profile is ready. **Do not add `--mode hybrid` to the grading
command.** The default is deliberately deterministic even when model environment
variables are supplied. The optional hybrid route is preserved for experiments but
does not have a clean funded comparison for the final implementation.

## Form fields

| Field | Value |
|---|---|
| Submission page | https://goals.xpert.chat/w/intern#submit |
| GitHub repo | https://github.com/Sohamgaonkhadkar/xpert-cold-start |
| Branch | `codex/final-evaluation-hardening` |
| Commit to grade | Exact final SHA in the handoff message and source-archive manifest; never use a moving branch name |
| Model | `openai/gpt-4.1-mini` |
| Full name / email | Use the identity and email registered for the challenge |

Suggested "Anything we should know" text:

> The two-argument run.sh command selects our verified deterministic profile and
> intentionally makes zero model calls, even when your key is present. The named
> model is the previously tested compatibility model for optional --mode hybrid
> experiments. Official dev: 28.9526/56; goals/help unlabelled. WRITEUP.md and the
> two-page metrics/writeup.pdf disclose measured failures. Python 3.10+ is sufficient
> for default inference, including schema validation without installed third-party
> libraries. Tests and hybrid mode use requirements.txt.

The official page was checked on **12 November 2026** and states **12 November,
23:59 IST**. Its FAQ permits no-model solutions and says the exact submitted commit
is graded. A non-default branch is therefore usable by its immutable commit SHA.
The repository is public; no visibility change or collaborator action is needed.

## Verified execution

```bash
bash run.sh <input_folder> <output_path>
python3 -m pip install -r requirements.txt  # for tests / optional hybrid mode
python3 -m pytest -q
bash run.sh example /tmp/example.json
python3 check.py example /tmp/example.json
python3 -m scripts.correction_demo
```

The public example and its zero-seed variant score 74/74. Official dev scores
28.952575934283253/56. There are 50/74 labelled loops matched, zero matched false DONE,
19 matched missed completions and six matched owner disagreements. Precision is
17.99%, recall 67.57%, F1 28.41%; incomplete dev labels limit interpretation.
Circles score 9.0732/12, organisations 3.6958/8 and chat ownership 6.3415/8.
Goal/help accuracy and the full /100 score are not measurable on this dev set.

All 125 local tests pass. Existing Linux CI covers Python 3.10/3.12/3.14 and a clean
3.12 container, public-example inference, paths with spaces and stdlib-only inference.
Dense synthetic scale: 20,000 messages, 320 chats, 5,000 loops, 63.88 seconds, $0.
This is a runtime test, not realistic network latency or held-out accuracy evidence.

## Packaging and records

The final source archive is produced from the immutable Git commit, excluding `.git`,
untracked credentials, private datasets, diagnostics, environments and outputs. Its
adjacent manifest records the exact commit and SHA-256. The GitHub commit remains the
actual submission artifact; the ZIP is a convenience copy, not a substitute for it.

`WRITEUP.md` has a rendered two-page PDF. `ai-logs/` contains genuine available records
and clearly marked selected transcript excerpts. The original Arena transcript could
not be recovered; its genuine summary is retained and is not called a transcript.
No AI conversation, model result or missing performance measurement was fabricated.

Known accuracy limits remain documented in EVALUATION.md and the November reports.
No elapsed-time completion rule, answer lookup, identity-specific exception or
duplicated prediction was introduced to fit ambiguous labels. No submission or
final-branch merge has been performed; the user's prior approval requirement remains.
