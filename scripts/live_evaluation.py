"""Run a private live evaluation with explicit credential loading and usage capture.

No prompts, responses, credentials or conversation text enter the aggregate report.
The output directory must be ignored by Git. --source-ref evaluates an immutable
source snapshot without creating another Git repository or changing the worktree.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def load_credentials(path):
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().removeprefix("export ")
        key, sep, value = line.partition("=")
        if sep and key.strip() in {"LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"}:
            os.environ[key.strip()] = value.strip().strip("\"'")
    if not all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")):
        raise ValueError("All three LLM environment variables are required")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("--env-file")
    parser.add_argument("--source-ref")
    parser.add_argument("--model")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    data = Path(args.input).resolve()
    output = Path(args.output_dir).resolve()
    # These files include raw conversation-derived predictions. Refuse a location
    # not covered by .gitignore, even for an accidental command-line typo.
    ignored = subprocess.run(["git", "check-ignore", "--quiet", str(output / "prediction.json")], cwd=repo)
    if ignored.returncode != 0:
        raise ValueError("Use a Git-ignored output directory, such as private-reports/run-name")
    if args.env_file:
        load_credentials(args.env_file)
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if not all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")):
        raise ValueError("A live evaluation requires all three LLM environment variables")
    output.mkdir(parents=True, exist_ok=True)
    usage = output / "usage.jsonl"
    if usage.exists():
        raise ValueError("Choose a fresh output directory; usage logs must not mix runs")
    source = repo
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if args.source_ref:
        revision = subprocess.check_output(["git", "rev-parse", args.source_ref], cwd=repo, text=True).strip()
        archive = subprocess.check_output(["git", "archive", "--format=zip", revision, "src", "schema.json"], cwd=repo)
        source = output / "source-snapshot"
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            bundle.extractall(source)
    fingerprint = hashlib.sha256()
    for path in sorted([source / "schema.json", *list((source / "src").glob("*.py"))]):
        fingerprint.update(str(path.relative_to(source)).replace("\\", "/").encode())
        fingerprint.update(path.read_bytes())
    environment = dict(os.environ, LLM_USAGE_LOG=str(usage), PYTHONPATH=str(source))
    # Instrument only telemetry for the historical baseline; extraction code is
    # unchanged. This captures OpenRouter's billed cost that older logs omitted.
    code = '''
import json, os, sys
from src.llm_client import LLMClient
def usage_log(self, model, data, seconds):
    usage = data.get("usage") or {}
    with self._lock:
        with open(os.environ["LLM_USAGE_LOG"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"model":model,"prompt_tokens":usage.get("prompt_tokens"),
                "completion_tokens":usage.get("completion_tokens"),"cost":usage.get("cost"),
                "seconds":seconds}) + "\\n")
LLMClient._append_usage_log = usage_log
from src.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", code, str(data), str(output / "prediction.json"),
                           "--telemetry", str(output / "telemetry.json")], cwd=source,
                          env=environment, timeout=1740, capture_output=True, text=True, encoding="utf-8")
    elapsed = time.perf_counter() - start
    if proc.returncode:
        # Keep detailed errors local: they can contain input-derived values.
        (output / "error.txt").write_text(proc.stderr, encoding="utf-8")
        print("Live pipeline failed; see private error.txt", file=sys.stderr)
        return proc.returncode
    from scripts.evaluate_official import evaluate
    report = evaluate(data, output / "prediction.json")
    telemetry = json.loads((output / "telemetry.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in usage.read_text(encoding="utf-8").splitlines()] if usage.exists() else []
    all_tokens_reported = all(isinstance(row.get(k), int) for row in rows for k in ("prompt_tokens", "completion_tokens"))
    all_cost_reported = all(isinstance(row.get("cost"), (int, float)) for row in rows)
    cost = sum(row["cost"] for row in rows) if rows and all_cost_reported else None
    report.update(model=os.environ.get("LLM_MODEL"), source_revision=revision,
                  inference_source_sha256=fingerprint.hexdigest(),
                  concurrency=int(os.environ.get("LLM_MAX_CONCURRENCY", "6")),
                  source_has_worktree_changes=not bool(args.source_ref), runtime_sec=elapsed,
                  actual_calls=len(rows), actual_prompt_tokens=sum(r["prompt_tokens"] for r in rows) if all_tokens_reported else None,
                  actual_completion_tokens=sum(r["completion_tokens"] for r in rows) if all_tokens_reported else None,
                  provider_cost_usd=cost, provider_dollars_per_1k_messages=cost * 1000 / report["messages"] if cost is not None else None,
                  estimated_cost_usd=telemetry["estimated_cost_usd"], stage_tokens=telemetry["stage_tokens"],
                  usage_details=telemetry.get("usage_details"), stage_timings_sec=telemetry["stage_timings_sec"])
    (output / "aggregate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(bool(report["validation_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
