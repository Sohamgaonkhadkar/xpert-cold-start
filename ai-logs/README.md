# Available AI development records

* `mission-prompt.md`: the user's genuine full handoff request.
* `codex-visible-transcript.md`: genuine visible user/assistant messages exported
  from this Codex session, with its export boundary stated. Tool payloads and internal
  reasoning are excluded; tool output included private challenge conversations.
* `session-summary.md`: the genuine pre-existing Arena engineering summary. Original
  Arena conversation exports were unavailable. This is not represented as a transcript.

`AUDIT.md`, `EVALUATION.md` and commit diffs preserve the review decisions and measured
outcomes. No missing historical conversation has been fabricated. Credentials, model
responses and raw challenge conversations must not appear in this directory.
