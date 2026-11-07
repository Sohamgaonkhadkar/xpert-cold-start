"""CLI entrypoint for Xpert Cold Start inference (`bash run.sh <input_folder> <output_path>`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .output_builder import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xpert-cold-start",
        description="Infer people circles, organisations, chats, goals, open loops, and interventions from WhatsApp history.",
    )
    parser.add_argument("input_folder", help="Folder containing chats.jsonl, messages.jsonl, and seed.json")
    parser.add_argument("output_path", help="Destination path for output.json")
    parser.add_argument(
        "--corrections",
        default=None,
        help="Optional path to user corrections JSON file (for live interview overrides)",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Optional path to schema.json (defaults to ./schema.json)",
    )
    parser.add_argument(
        "--telemetry",
        default=None,
        help="Optional path to write runtime/token/cost telemetry JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stderr telemetry summary",
    )
    args = parser.parse_args(argv)

    try:
        output, _ = run_pipeline(
            input_folder=Path(args.input_folder),
            output_path=Path(args.output_path),
            corrections_path=args.corrections,
            schema_path=args.schema,
            telemetry_path=args.telemetry,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"[xpert-cold-start] Failed: {exc}\n")
        return 1

    if not args.quiet:
        tel = output.telemetry
        sys.stderr.write(
            f"[xpert-cold-start] Completed in {tel.stage_timings_sec.get('total_sec', 0.0):.2f}s | "
            f"msgs={tel.total_messages} | model={tel.model_used} | "
            f"tokens={tel.total_tokens} (${tel.dollars_per_1k_messages:.4f}/1k msgs) | "
            f"people={len(output.people)} orgs={len(output.organisations)} "
            f"chats={len(output.chats)} goals={len(output.goals)} loops={len(output.open_loops)}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
