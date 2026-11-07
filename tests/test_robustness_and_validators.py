"""Tests for order invariance, temporal causality, evidence repair, and schema field adaptation."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path

from scripts.generate_synthetic_suite import build_synthetic_benchmark
from src.conversation_index import ConversationIndex
from src.data_loader import load_dataset
from src.evidence_validator import validate_and_repair
from src.models import (
    ColdStartOutput,
    GoalPrediction,
    InterventionPrediction,
    OpenLoopPrediction,
)
from src.output_builder import run_pipeline
from src.schema_adapter import serialize_output_for_schema, validate_json_against_schema


def test_jsonl_shuffle_order_invariance(tmp_path: Path):
    """Shuffling lines of chats.jsonl and messages.jsonl produces byte-for-byte identical predictions."""
    orig_dir = tmp_path / "orig"
    shuf_dir = tmp_path / "shuf"
    build_synthetic_benchmark(orig_dir)
    shutil.copytree(orig_dir, shuf_dir)

    rng = random.Random(12345)
    for fname in ("chats.jsonl", "messages.jsonl"):
        lines = (shuf_dir / fname).read_text("utf-8").splitlines()
        rng.shuffle(lines)
        (shuf_dir / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")

    _, orig_json = run_pipeline(orig_dir)
    _, shuf_json = run_pipeline(shuf_dir)
    assert orig_json == shuf_json


def test_temporal_truncation_causality(tmp_path: Path):
    """Truncating messages at 2026-10-26 keeps Oct 27 closures OPEN without crashing or leaking."""
    tdir = tmp_path / "trunc"
    build_synthetic_benchmark(tdir)
    kept = [
        ln
        for ln in (tdir / "messages.jsonl").read_text("utf-8").splitlines()
        if ln.strip() and json.loads(ln)["ts"] <= "2026-10-26T23:59:59"
    ]
    (tdir / "messages.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")

    out, _ = run_pipeline(tdir)
    by_msg = {l.message_id: l for l in out.open_loops}
    assert by_msg["m103"].status == "open"  # Closed on Oct 27 in full history -> open on Oct 26!
    assert by_msg["m115"].status == "done"  # Closed on Oct 25 -> already done on Oct 26!
    assert by_msg["m128"].status == "dropped"  # Cancelled on Oct 26 -> already dropped!


def test_empty_seed_graceful_execution(tmp_path: Path):
    """Pipeline runs cleanly and emits schema-valid output even when seed.json is completely empty."""
    edir = tmp_path / "empty_seed"
    build_synthetic_benchmark(edir)
    (edir / "seed.json").write_text('{"people": {}, "organisations": [], "goals": []}\n', encoding="utf-8")

    _, payload = run_pipeline(edir)
    from src.schema_adapter import load_schema

    errors = validate_json_against_schema(payload, load_schema())
    assert errors == []
    assert len(payload["people"]) == 20
    assert len(payload["chats"]) == 27


def test_evidence_validator_repairs_broken_invariants(tmp_path: Path):
    """Deterministic validator drops ungrounded goals, rejects out-of-order closures, and repairs missing entities."""
    ddir = tmp_path / "val_test"
    build_synthetic_benchmark(ddir)
    bundle = load_dataset(ddir)
    index = ConversationIndex(bundle)

    broken = ColdStartOutput(
        people=[],  # Missing all 20 people -> validator must repair!
        organisations=[],
        chats=[],  # Missing all 27 chats -> validator must repair!
        goals=[
            GoalPrediction(
                id="bad_goal",
                title="Invented goal with fake message id",
                area="self",
                type="habit",
                horizon="1yr",
                message_ids=["FAKE_MSG_999"],
            ),
            GoalPrediction(
                id="good_goal",
                title="Run 3x weekly",
                area="health",
                type="habit",
                horizon="3mo",
                message_ids=["m020"],
            ),
        ],
        open_loops=[
            # Impossible temporal closure: m001 (Oct 19) claiming closure on m105? Wait: m105 (Oct 27) claiming origin m130 (Oct 28)!
            OpenLoopPrediction(
                id="loop_bad_time",
                title="Backwards closure",
                type="deadline",
                owner="ME",
                chat_id="c_grp_class4b",
                message_id="m130",  # ts = 2026-10-28
                status="done",
                closure_message_id="m105",  # ts = 2026-10-27 < origin_ts!
                intervention=None,
            ),
            # Done loop with dangling intervention -> validator must clear intervention!
            OpenLoopPrediction(
                id="loop_done_iv",
                title="Valid done loop with leftover intervention",
                type="followup",
                owner="ME",
                chat_id="c_dm_ca",
                message_id="m103",
                status="done",
                closure_message_id="m105",
                intervention=InterventionPrediction(action="remind", remind_at="2026-11-04T09:00:00"),
            ),
        ],
    )

    warnings = validate_and_repair(broken, index)
    assert len(warnings) >= 3
    assert len(broken.people) == 20
    assert len(broken.chats) == 27
    assert [g.id for g in broken.goals] == ["good_goal"]

    repaired_bad_time = next(l for l in broken.open_loops if l.id == "loop_bad_time")
    assert repaired_bad_time.status == "open"
    assert repaired_bad_time.closure_message_id is None
    assert repaired_bad_time.intervention is not None

    repaired_done = next(l for l in broken.open_loops if l.id == "loop_done_iv")
    assert repaired_done.status == "done"
    assert repaired_done.intervention is None


def test_schema_adapter_dynamic_alias_adaptation(tmp_path: Path):
    """Schema adapter dynamically maps renamed fields when an alternate official schema.json is provided."""
    ddir = tmp_path / "alias_test"
    build_synthetic_benchmark(ddir)
    out, _ = run_pipeline(ddir)

    alternate_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["people", "organisations", "chats", "goals", "open_loops"],
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["person_id", "circle"],
                    "properties": {
                        "person_id": {"type": "string"},
                        "name": {"type": "string"},
                        "circle": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "organisations": {"type": "array", "items": {"type": "object"}},
            "chats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "target_type", "target_id"],
                    "properties": {
                        "id": {"type": "string"},
                        "target_type": {"type": "string"},
                        "target_id": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "goals": {"type": "array", "items": {"type": "object"}},
            "open_loops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["loop_id", "title", "type", "owner_id", "origin_chat_id", "origin_message_id", "status"],
                    "properties": {
                        "loop_id": {"type": "string"},
                        "title": {"type": "string"},
                        "type": {"type": "string"},
                        "owner_id": {"type": "string"},
                        "origin_chat_id": {"type": "string"},
                        "origin_message_id": {"type": "string"},
                        "status": {"type": "string"},
                        "closed_by_message_id": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }

    adapted = serialize_output_for_schema(out, alternate_schema)
    errs = validate_json_against_schema(adapted, alternate_schema)
    assert errs == []
    assert "person_id" in adapted["people"][0]
    assert "target_id" in adapted["chats"][0]
    assert "origin_message_id" in adapted["open_loops"][0]
