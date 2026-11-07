# MISSION: COMPLETE, VALIDATE AND OPTIMIZE OUR XPERT COLD START CHALLENGE SUBMISSION

You are taking over an existing, substantially developed AI engineering project from Arena AI.

**Your role is to act as a senior AI engineer, code reviewer, evaluation specialist, performance optimization engineer, and final submission auditor.**

This is not a new project.

Do not rebuild it from scratch.

Your job is to retrieve the complete existing implementation, merge the unfinished Arena development branch, understand the architecture, identify remaining weaknesses, perform real evaluations, improve the system wherever justified, and prepare a reliable, competitive submission.

Do not stop after producing an assessment or development plan.

**Perform the actual coding, testing, debugging, benchmarking, documentation and verification work.**

---

# 1. PROJECT AND REPOSITORY DETAILS

**Project:** Xpert Winter Internship 2026 — Cold Start Challenge

**Repository:**

https://github.com/Sohamgaonkhadkar/xpert-cold-start

**Official challenge:**

https://goals.xpert.chat/w/intern

**Deadline:** November 12, 2026, 23:59 IST.

**Current architecture:** Dossier-ARR-PRISM.

Our architecture combines:

- Evidence-grounded contact dossiers.
- Relationship and organisation inference.
- Multilingual conversation indexing.
- Anchor–Retrieve–Reconcile (ARR) temporal task tracking.
- Asymmetric closure guards.
- PRISM risk-sensitive intervention decisions.
- Selective LLM inference and cost optimization.
- Evidence validation.
- User correction support.

The existing implementation is written primarily in Python.

## IMPORTANT: CURRENT GITHUB STATE

As independently checked on November 7, 2026:

Repository default branch: `main`

Arena development branch:

`arena/99bb2717-xpert-cold-start`

Open pull request:

https://github.com/Sohamgaonkhadkar/xpert-cold-start/pull/1

PR #1 title:

`feat: Dossier-ARR-PRISM cold-start engine (74/74, 51 tests)`

Latest observed Arena commit:

`7312bb03acdc15a206a29a5218efe290186a676b`

PR #1 was:

- OPEN.
- Targeting `main`.
- Reported mergeable.
- Reported free of merge conflicts.

The `main` branch still contained only the initial placeholder README.

**YOUR FIRST ACTION MUST BE TO RESOLVE THIS REPOSITORY STATE.**

---

# 2. FIRST: FETCH, VERIFY AND MERGE THE EXISTING ARENA WORK

Before doing any further development:

1. Inspect the current working directory.
2. Identify the connected GitHub repository.
3. Check for uncommitted local modifications.
4. Fetch all remote branches and latest commits.
5. Check PR #1 and its latest head commit.
6. Verify the PR is still open and mergeable.
7. Inspect its changes and run a quick sanity check where feasible.
8. Merge PR #1 into `main` if it is still open and safe to merge.
9. Switch to `main`.
10. Pull and verify the complete merged implementation.
11. Confirm that the existing source code, tests, documentation and evaluation tools are present.
12. Create a fresh working branch for the remaining Codex improvements.

Suggested initial commands:

```bash
git status
git remote -v
git fetch origin --prune
git branch -a

gh pr view 1 \
  --json state,mergeable,headRefName,baseRefName,headRefOid
```

If PR #1 is open and mergeable, merge it using the normal GitHub workflow.

```bash
gh pr merge 1 --merge
```

Then synchronize:

```bash
git switch main
git pull --ff-only origin main
git log -5 --oneline
```

Create a working branch:

```bash
git switch -c codex/final-evaluation-hardening
```

These commands are illustrative. Adapt them safely to the actual environment.

**Rules:**

- Never force-push.
- Never overwrite uncommitted work.
- Never discard Arena's implementation.
- Never create a second duplicate repository.
- If the PR is already merged, do not merge it again.
- If conflicts appear, resolve them carefully while preserving the existing implementation.
- Do not merge unrelated or unreviewed branches.
- Record the final merged commit SHA.

The existing Arena PR #1 merge is explicitly authorized by the user.

Subsequent improvements should be developed on the new branch for review. Do not merge the final improvement branch or submit the challenge without approval.

---

# 3. UNDERSTAND THE OFFICIAL CHALLENGE

Our system processes a person's WhatsApp conversation history.

The hidden evaluation contains approximately:

- Eight weeks of conversational context.
- More than 300 chats.
- Approximately 20,000 messages.
- English, Hindi, Bengali and Hinglish.
- Personal, family, household, business, client, vendor and staff interactions.

The task is to infer five components.

## A. People and circles

Classify relationships using exactly the supported circles:

spouse, child, parent, in-law, sibling, relative, friend, colleague, client, vendor, partner, staff.

## B. Organisations and chats

Infer:

- User-owned or user-associated organisations.
- Organisation type.
- Home or work side.
- User's role.
- People running or supporting each organisation.
- Correct ownership of every chat.

## C. Goals

Infer up to ten supported goals.

Each needs:

- Title.
- Area.
- Type.
- Horizon.
- Actual supporting message IDs.

Unsupported or invented goals are penalized.

## D. Open loops

Identify commitments, follow-ups, deadlines and unanswered questions.

For every loop:

- Identify the task.
- Determine who owes it.
- Identify its originating message.
- Determine its due date.
- Determine whether it remains open, was completed or was dropped.
- Preserve closure evidence.

False completion claims are particularly expensive.

## E. Assistant intervention

For every open loop, determine whether the assistant should:

- ASSIST.
- REMIND.
- LEAVE.

False interventions on tasks that should be left alone receive heavy penalties.

Do not build an unnecessary frontend or live WhatsApp bot.

The competition evaluates a program producing structured JSON.

---

# 4. OFFICIAL SCORING OBJECTIVE

The scoring system awards:

| Category | Points |
|---|---:|
| Circles | 12 |
| Organisations | 8 |
| Chats | 8 |
| Goals | 8 |
| Open loops found | 14 |
| Open-loop closure | 14 |
| Assistant intervention | 10 |
| Cost | 12 |
| Write-up | 7 |
| One-command execution | 7 |
| TOTAL | 100 |

Optimize the actual weighted score.

The official development checker evaluates 74 points.

The remaining points include cost, write-up quality and execution reliability.

Do not confuse scores with different denominators.

The hidden evaluation runs in a clean container, with a 30-minute time limit and no general internet access except the configured model endpoint.

Required environment variables:

```text
LLM_BASE_URL
LLM_API_KEY
LLM_MODEL
```

Required command:

```bash
bash run.sh <input_folder> <output_path>
```

The hidden set renames identities. Do not hardcode names, IDs, organisations or development-set answers.

---

# 5. UNDERSTAND THE COMPLETE EXISTING CODEBASE

After merging and pulling the repository, perform a thorough code review.

Read every important implementation module rather than relying only on README claims.

The previous audit found these components:

```text
src/
  cli.py
  data_loader.py
  conversation_index.py
  indic_normalizer.py
  models.py
  relationship_inference.py
  organisation_inference.py
  goal_extraction.py
  loop_extraction.py
  temporal_resolver.py
  intervention_policy.py
  llm_client.py
  evidence_validator.py
  schema_adapter.py
  output_builder.py
  corrections.py
```

Also inspect:

```text
run.sh
requirements.txt
pyproject.toml
schema.json
check.py
README.md
WRITEUP.md
ARCHITECTURE.md

scripts/
tests/
metrics/
ai-logs/
```

A newer Arena commit added:

`tests/test_pass6_whatsapp_edges.py`

Do not assume the file inventory is unchanged. Inspect the actual latest tree.

For each important component, establish:

- What it does.
- What information it consumes.
- How it uses the model.
- Which rules override model responses.
- How it handles missing or contradictory evidence.
- Which failure cases are already covered.
- What remains untested.
- Whether its complexity is justified.

Create a concise internal audit before modifying the implementation.

**Preserve the existing architecture unless actual evidence supports replacing a component.**

---

# 6. CRITICAL FINDING: CURRENT SCORES ARE NOT ENOUGH

The previous independent review found that the repository reports:

- `74/74` on the official worked example.
- `92.58/93` on a custom synthetic benchmark.
- Approximately 96% open-loop precision and recall on the synthetic benchmark.
- Strong performance on identity permutation and seed ablation tests.
- Very low estimated LLM costs.
- Approximately 51 automated tests.

These are useful development signals.

However, we discovered several limitations.

## A. Real development data was not evaluated

The repository explicitly states that the official Arjun development dataset was unavailable in the original Arena environment.

Therefore, its actual competition performance remains unknown.

## B. Simulated LLM responses were used

`scripts/run_experiments.py` uses:

`simulated_smart_llm_responder`

The simulator contains test-specific behavior for synthetic messages, including `mh06` and `mh09`.

This limits the meaning of its headline benchmark results.

## C. The 20,000-message runtime excludes genuine network latency

The reported runtime is primarily local execution with simulated model responses.

It does not establish the runtime of the actual evaluator-configured model.

## D. Synthetic data may not represent the real challenge distribution

The custom benchmark is useful for regression testing but may not represent realistic ambiguity, density, conversational language or task distribution.

### Your responsibility

Explicitly separate:

1. Official worked-example scores.
2. Custom synthetic scores.
3. Mock-model results.
4. Actual official dev-set scores.
5. Actual live-model scores.
6. Estimated hidden-set behavior.

Never report one category as another.

Do not claim an expected leaderboard rank without evidence.

---

# 7. HIGHEST PRIORITY: OBTAIN THE ACTUAL DEVELOPMENT DATA

Investigate the official challenge website and available development archives.

Official challenge:

https://goals.xpert.chat/w/intern

The repository currently contains the official schema, checker and worked example.

Verify those files against the latest official starter kit.

The existing copies of `check.py`, `schema.json` and `example/` were previously checked against a public competitor's copies and matched at the Git blob level.

However, that does not replace obtaining the real dataset.

Find and download the official development data from the legitimate challenge source.

Expected local structure:

```text
data/dev/
    chats.jsonl
    messages.jsonl
    seed.json
    answers/
```

Verify the exact archive structure before using this layout.

Do not commit development data or answer keys.

Do not upload the dataset to third-party analysis websites or external services.

Challenge conversation data must be sent only to the permitted model endpoint when required for inference.

If official downloads fail:

- Inspect the official page and redirects.
- Check whether the archive is already available in the workspace.
- Check for an accessible user-provided copy.
- Record the precise failure.
- Continue all independent engineering tasks.
- Clearly identify the missing file needed to complete real-data validation.

Do not fabricate development data and call it official.

Do not use another competitor's reported score as our own evaluation.

---

# 8. RUN THE REAL OFFICIAL EVALUATION

Once the dataset is available:

Run the current implementation without making further algorithmic modifications.

This establishes a trustworthy baseline.

```bash
bash run.sh data/dev /tmp/xpert-dev-output.json
python3 check.py data/dev /tmp/xpert-dev-output.json
```

Record the complete score breakdown.

At minimum, record:

- Circles score.
- Organisation score.
- Chat ownership score.
- Goal score.
- Open-loop discovery score.
- Open-loop closure score.
- Intervention score.
- Actual runtime.
- Total LLM calls.
- Actual input and output tokens.
- Actual cost per 1,000 messages.
- Schema errors.
- Referential errors.

Also calculate:

- Open-loop precision.
- Open-loop recall.
- Open-loop F1.
- False DONE predictions.
- Missed completions.
- False DROPPED predictions.
- Incorrect ownership.
- Incorrect interventions on LEAVE examples.
- Incorrect or unsupported goals.

Inspect individual false positives and false negatives.

Save a concise development evaluation report.

Do not place raw private WhatsApp messages in committed reports.

### Compare three modes

**Mode A: Deterministic inference**

Run without model access.

**Mode B: Actual LLM-enabled inference**

Use a real supported model through the required environment variables.

**Mode C: Improved hybrid inference**

Run the optimized system after applying verified fixes.

Measure actual score changes and token costs.

Do not rely on mocked model outputs for the real-model comparison.

---

# 9. VERIFY LIVE OPENAI-COMPATIBLE MODEL BEHAVIOR

The code has an OpenAI-compatible client in:

`src/llm_client.py`

Inspect and test it carefully.

Verify:

1. Authentication.
2. Endpoint URL construction.
3. Model identifier compatibility.
4. JSON response parsing.
5. Handling of unsupported optional parameters.
6. Token usage reporting.
7. Retry behavior.
8. Rate limits.
9. Timeout behavior.
10. Concurrent request safety.
11. Cost budget enforcement.
12. Failed-request handling.
13. Deterministic fallback behavior.
14. Correct use of `LLM_USAGE_LOG`.

The evaluator supplies its own key.

Do not require credentials from unrelated services.

If a real API key is available, perform a small live smoke test before full evaluation.

If no key is available, use mock endpoints only to validate protocol compatibility, and report that real-model quality remains unverified.

Do not use hardcoded responses that depend on known benchmark IDs.

### Model selection

Investigate whether the selected model offers a good accuracy/cost trade-off.

Compare at least two suitable models when legitimate access and testing budgets permit.

The submitted model identifier must correspond to the model actually tested.

Do not assume the model used during local simulation is the best model for evaluation.

Verify current provider availability and model compatibility rather than relying only on the repository's pricing table.

---

# 10. AUDIT AND IMPROVE THE HIGHEST-VALUE EXTRACTION COMPONENTS

Prioritize actual errors revealed by the official development evaluation.

## A. Open-loop discovery

This category carries 14 points.

Investigate:

- Missed implicit commitments.
- Requests without conventional task verbs.
- Voice-transcript commitments.
- Image-caption and document-related tasks.
- Questions that genuinely require a decision.
- False positive social questions.
- Repeated reminders classified as new tasks.
- Cross-chat references.
- Different wording for the same task.
- Task owner attribution.

The system should maintain high recall without generating excessive false positives.

Use actual dev-set error analysis to select improvements.

## B. Closure detection

This category carries 14 points.

Audit:

- False completion from "ok".
- Acknowledgement versus actual execution.
- Future-tense versus past-tense verbs.
- Completion in another conversation.
- Reassignment.
- Postponement.
- Cancellation.
- Reopened tasks.
- Partial completion.
- Ambiguous short responses.

False DONE predictions are heavily penalized.

Preserve conservative closure guards where they improve performance.

Investigate whether overly conservative rules cause excessive missed completions.

## C. Temporal scoring window

Important:

The hidden set contains several weeks of history, but open loops are scored over the final two weeks.

Inspect the actual official checker and reference format.

The current implementation includes a scoring cutoff around the final 14 days and special pre-cutoff filtering.

Verify that this behavior matches the official evaluation precisely.

Check whether earlier context is being discarded when it is necessary for interpreting a later conversation.

Do not change this logic based on assumptions.

## D. Intervention decisions

This category carries 10 points.

The current PRISM policy uses an asymmetric loss matrix.

Audit whether its values and probability thresholds are empirically justified.

Determine whether the current rules:

- Remind too frequently.
- Leave too many useful tasks untouched.
- Incorrectly intervene in spouse or staff matters.
- Misclassify assistant-capable tasks.
- Set `needs_yes` incorrectly.
- Generate inappropriate reminder times.

Do not treat the existing loss matrix as automatically calibrated just because it looks mathematically formal.

Calibrate against actual available labels where possible.

## E. Circles, organisations and chats

Together worth 28 points.

Audit errors involving:

- Staff versus vendor.
- Colleague versus partner.
- Client versus colleague.
- Sibling versus relative.
- Personal versus institutional group ownership.
- Client organisations incorrectly treated as the user's organisations.
- Household versus vacation-home staff.
- Sparse-contact inference.
- Dual-role relationships.
- Zero-seed inference.

Avoid introducing new heuristics without regression tests.

---

# 11. FIX VALIDATION AND EXECUTION RISKS

The previous code review identified a potentially important issue.

In `src/output_builder.py`, the pipeline validates its output schema but may continue writing an invalid output after recording schema warnings.

Audit this behavior.

Unresolved schema violations must not silently produce a supposedly successful submission.

Implement a reliable validation strategy:

1. Apply safe deterministic repairs.
2. Serialize the output.
3. Validate the exact official schema.
4. Validate referential integrity.
5. Confirm that all required fields exist.
6. Confirm all IDs reference actual input entities.
7. Reject unresolved invalid output with a clear nonzero exit status.

Do not quietly alter valid predictions simply to suppress errors.

### Additional checks

Confirm that:

- `run.sh` works with only two required arguments.
- No answer keys are required by inference.
- Input and output directories can differ.
- The current working directory does not affect execution.
- Paths with spaces are handled.
- The program has no unexpected outbound network calls.
- No external database is needed.
- All required dependencies install cleanly.
- No hidden local files are required.
- The solution runs in less than 30 minutes with realistic API latency.
- Output remains deterministic where appropriate.
- The process exits with a meaningful status on failure.

Create or improve automated regression tests for these invariants.

---

# 12. REVIEW CURRENT TEST QUALITY

The repository reportedly includes 51 passing automated tests.

Run the complete suite.

```bash
python3 -m pytest -q
```

Then independently audit what those tests actually establish.

Categorize tests as:

- Pure unit tests.
- Schema tests.
- Official worked-example tests.
- Synthetic integration tests.
- Mock-model tests.
- Actual live-model tests.
- Real development-set tests.
- Scalability tests.
- Regression tests.

Identify missing categories.

Most importantly:

**A passing test with a simulated language model is not evidence that a real language model returns the same prediction.**

Create new tests when necessary.

Prioritize realistic failure modes, not simply increasing the test count.

### Robustness tests

Retain and verify existing tests for:

- Identity renaming.
- JSONL message-order shuffling.
- Missing seed information.
- Multilingual text.
- Quoted replies.
- Timezone boundaries.
- Contradictory task statuses.
- Cross-chat completion.
- Bystander acknowledgements.
- Similar names.
- User corrections.

Add tests only when they cover an actual weakness or significant risk.

---

# 13. BENCHMARK AGAINST PUBLIC COMPETITORS

Perform a fresh, selective GitHub audit of available Xpert challenge implementations.

Previously identified:

https://github.com/CodeInfinity1/cold-start

Also search for more recent relevant submissions.

One known competitor publicly reports:

- 27.6/56 on the scored development-set categories.
- 18% open-loop precision.
- 51% open-loop recall.
- A rules-first implementation.
- An optional LLM layer that is disabled in the graded configuration.

Check whether the competitor has newer commits before comparing.

Do not assume those figures remain current.

Use competitors to discover potential failure cases or better engineering approaches.

Do not copy private submissions or challenge answer keys.

Compare fairly:

- Same official dataset.
- Same scoring categories.
- Same denominator.
- Similar model configuration where possible.
- Actual execution cost.
- Actual runtime.

Do not compare our synthetic 92.58/93 against another participant's real dev score and claim victory.

Investigate useful architectural ideas, but prioritize improving our own system through measurement.

---

# 14. COST AND PERFORMANCE OPTIMIZATION

Cost accounts for 12 points.

The repository already implements selective inference.

Do not replace it with indiscriminate full-history LLM processing.

Measure the actual token usage of each stage:

- Relationship inference.
- Organisation and chat inference.
- Goal extraction.
- Loop discovery.
- Temporal reconciliation.
- Intervention policy.

Identify unnecessarily expensive stages.

Investigate:

- Redundant repeated prompts.
- Duplicate candidate processing.
- Excessive retrieval context.
- Unnecessary high-confidence verification calls.
- Poor batching.
- Excessive retries.
- Cache effectiveness.
- Prompt-size outliers.
- Failed calls consuming tokens.

Avoid optimizing microseconds of Python execution while ignoring inference latency.

Optimize based on total scored value.

A reduction in token usage is valuable only if it does not sacrifice disproportionately more accuracy points.

### Runtime requirement

The official container limit is 30 minutes.

Run realistic scalability tests with approximately 20,000 messages and 300+ chats.

Include realistic model latency when an endpoint is available.

Do not report mocked sub-second inference as production runtime.

Ensure the model request budget, retry strategy and concurrency settings remain within the time limit.

---

# 15. FIX THE WRITE-UP

The existing `WRITEUP.md` is technically detailed but may exceed the official two-page maximum.

The latest version was approximately 9.7 KB of Markdown.

Inspect and shorten it appropriately.

The final write-up must be **no longer than two pages**.

Do not remove important evidence.

Prioritize:

### Section 1: What we focused on

Explain why we prioritized relationship inference, task discovery, closure precision and intervention decisions.

### Section 2: Architecture

Briefly explain Dossier-ARR-PRISM and why it is appropriate for the problem.

### Section 3: Evaluation and cost

Present real results separately from synthetic results.

Include actual model usage and dollars per 1,000 messages.

### Section 4: Where our system fails

This is particularly important.

The challenge explicitly evaluates whether our predicted failures match real failures.

Provide measured failure categories and approximate frequencies where available.

Do not present guessed failure rates as measurements.

### Section 5: Key engineering decisions

Briefly justify major trade-offs, especially why we avoided an expensive multi-agent architecture.

Keep the writing natural, concise and technically credible.

Avoid excessive promotional language.

Verify the two-page limit by rendering or otherwise checking the final document layout.

---

# 16. FIX AI DEVELOPMENT LOGS

The challenge explicitly asks for:

`ai-logs/`

containing transcripts of AI-assisted development sessions.

The existing repository contains:

`ai-logs/session-summary.md`

This documents engineering decisions but is not equivalent to complete session transcripts.

Determine which genuine session transcripts are available.

Preserve available original conversations, important prompts, code-review interactions and debugging decisions.

Record the current Codex session appropriately.

Do not fabricate missing conversations or invent historical AI interactions.

Do not include API keys, credentials or sensitive challenge conversations.

If previous Arena transcripts cannot be exported, state that honestly and retain the available summary.

The goal is to demonstrate how AI assisted engineering decisions, rather than artificially inflate the amount of AI usage.

---

# 17. REPOSITORY AND SECURITY AUDIT

Inspect the current `.gitignore`.

Confirm that it excludes:

- Official challenge development data.
- Hidden evaluation data.
- Answer keys outside the permitted worked example.
- API keys.
- `.env` files.
- Credentials.
- Temporary model responses containing conversation data.
- Private runtime outputs.
- Virtual environments.
- Python caches.

Run a check of tracked files and commit history for unintended sensitive material.

Do not commit actual WhatsApp development conversations.

Do not upload them to third-party evaluation or logging services.

The repository was previously observed to be public.

Do not change repository visibility without approval, but flag any privacy or competitive concerns.

---

# 18. IMPROVE CLEAN-INSTALL REPRODUCIBILITY

The repository previously had no GitHub Actions workflow runs.

Add an appropriate lightweight CI workflow if useful.

It should test:

1. Python dependency installation.
2. Unit tests.
3. Official worked-example inference.
4. Schema validation.
5. Output reference integrity.
6. CLI execution without requiring a real API key.

Do not expose secrets in CI.

Do not run expensive live-model evaluations on every commit.

Include clear instructions for reproducing the verified results.

A reviewer should be able to understand the project and run it without special setup beyond the documented dependencies and model credentials.

---

# 19. PREPARE FOR THE LIVE INTERVIEW

The shortlisted interview includes a live correction exercise.

The interviewer may tell us that our system incorrectly identified:

- A person's circle.
- A relationship.
- An organisation.
- A chat owner.
- A commitment.
- A task owner.
- A task status.

The existing `src/corrections.py` supports correction operations.

Audit whether corrections truly behave as intended.

Perform a realistic test:

1. Run inference on a fixture.
2. Apply a relationship correction.
3. Rerun or apply the correction layer.
4. Verify that the corrected relationship changes appropriately.
5. Verify unrelated predictions remain stable.
6. Test a task ownership correction.
7. Test a completion-status correction.

Do not build a frontend for this.

Provide a concise command-line demonstration that can be explained in under ten minutes.

Keep correction behavior consistent with the official inference rules.

---

# 20. FINAL EXPERIMENTAL PROCESS

Work in measurable stages.

## Phase A — Repository integration

Merge Arena PR #1.

Pull the complete implementation.

Verify the latest commit and baseline test suite.

## Phase B — Independent audit

Understand the existing pipeline.

Identify specific bugs, fragile assumptions and evidence gaps.

## Phase C — Real development evaluation

Obtain official data where accessible.

Run the current implementation.

Record the actual baseline score.

## Phase D — Real model validation

Test the production model endpoint.

Compare deterministic and model-enabled behavior.

Measure actual costs.

## Phase E — Targeted improvements

Fix the largest measured sources of lost points.

Add regression tests.

Avoid unnecessary redesign.

## Phase F — Optimization

Reduce inference costs and runtime while preserving accuracy.

## Phase G — Packaging

Fix schema validation, documentation, AI logs and clean-install behavior.

## Phase H — Final verification

Run all tests.

Run the official checker.

Review any remaining warnings.

Prepare the repository for user review.

---

# 21. FINAL DELIVERABLES

At the end, provide the following information.

## A. Repository status

- Whether PR #1 was successfully merged.
- Current default branch commit.
- Final Codex working branch.
- Files changed.
- Current Git status.

## B. Evaluation results

Present a table comparing:

- Original Arena implementation.
- Improved implementation.
- Deterministic mode.
- Live-model mode.

Use actual scores only where measured.

Distinguish synthetic and official results.

## C. Accuracy breakdown

Report all seven automatically scored categories.

Include open-loop precision, recall, closure errors and intervention mistakes.

## D. Cost report

Include:

- Model tested.
- Actual model calls.
- Input tokens.
- Output tokens.
- Cost per 1,000 messages.
- Total runtime.

Clearly label estimates.

## E. Verification

Report:

- Automated test results.
- Official checker results.
- Schema validation results.
- Clean-environment execution.
- Security/data-leak audit.
- Documentation compliance.

## F. Remaining weaknesses

Explain:

- What still fails.
- Expected failure categories.
- Which errors are measured.
- Which risks remain unverified.
- What could improve the solution further.

## G. Submission readiness

Give a clear verdict:

**READY TO SUBMIT**

or

**NOT READY TO SUBMIT**

Support the verdict with evidence.

If something cannot be completed, explicitly state why and what is missing.

---

# 22. IMPORTANT EXECUTION RULES

Follow these rules throughout your work.

1. Do not rebuild the project unnecessarily.
2. Merge the existing Arena PR before continuing, provided it is safe.
3. Preserve previously working functionality.
4. Do not trust README benchmark claims without reproducing them.
5. Do not fabricate official development-set scores.
6. Do not fabricate live-model evaluations.
7. Do not introduce answer-key leakage.
8. Do not hardcode the synthetic benchmark's message IDs.
9. Do not overfit to the worked example.
10. Do not introduce unnecessary expensive frameworks.
11. Do not add a frontend.
12. Do not commit private challenge data.
13. Do not expose credentials.
14. Do not sacrifice accuracy for unmeasured cost savings.
15. Do not sacrifice the 30-minute execution limit.
16. Do not silently ignore invalid JSON output.
17. Do not claim a competitor ranking without comparable evidence.
18. Keep fixes modular and well tested.
19. Keep WRITEUP.md within two pages.
20. Do not submit the challenge or merge the final improvement PR without my approval.

The authorized Arena PR #1 merge is an exception: perform that initial merge as instructed.

If a model key or official dataset is unavailable, finish all independent work, document the precise blocker, and do not pretend the evaluation was performed.

Do not stop merely because an optional data source or tool is unavailable.

---

# FINAL INSTRUCTION

**Start by checking PR #1, merging the completed Arena implementation into `main`, and pulling the latest repository code.**

Then independently inspect the complete codebase and continue with the unfinished engineering work.

The main priority is obtaining trustworthy performance measurements on the official development set using an actual language model.

We already have an ambitious implementation. What we now need is strong evidence that it works.

Focus on measurable improvements in the competition's highest-value categories:

**Open-loop discovery, closure accuracy, relationship inference, intervention decisions and API cost.**

Complete the implementation, fix verified weaknesses, run the relevant tests, finish the documentation and prepare the repository for final review.

**Do the work, not just the planning.**

Our objective is to deliver a technically convincing, cost-efficient, reproducible and competitive submission to the Xpert Winter Internship 2026 Cold Start Challenge.