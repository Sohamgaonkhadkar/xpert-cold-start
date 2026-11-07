"""Tests for official starter.zip check.py/schema.json parity, zero-seed example scoring,
$LLM_USAGE_LOG telemetry, pan-Indian Unicode scripts, and competitor (CodeInfinity1) dev-set traps."""

from __future__ import annotations

import json
import os
from pathlib import Path

import check as official_check
from src.llm_client import LLMClient
from src.output_builder import run_pipeline
from src.schema_adapter import validate_json_against_schema


ROOT = Path(__file__).resolve().parent.parent


def test_official_starter_example_scores_perfect_74_of_74(tmp_path: Path):
    """Official worked example (`example/`) passes strict schema.json + check.py at 74.0 / 74."""
    out_file = tmp_path / "example_out.json"
    _, payload = run_pipeline(ROOT / "example", output_path=out_file)

    schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
    chats, messages, seed, answers = official_check.load_input(ROOT / "example")

    assert validate_json_against_schema(payload, schema) == []
    assert official_check.validate(payload, chats, messages) == []

    scores = official_check.score(payload, answers, seed, messages)
    assert scores["circles"] == 1.0
    assert scores["organisations"] == 1.0
    assert scores["chats"] == 1.0
    assert scores["goals"] == 1.0
    assert scores["loops_found"] == 1.0
    assert scores["loops_closed"] == 1.0
    assert scores["help"] == 1.0


def test_official_starter_example_zero_seed_scores_74_of_74(tmp_path: Path):
    """Stripping seed.json to 0% (simulating hidden data/test) still scores 74.0 / 74 on example/."""
    zs_dir = tmp_path / "example_zeroseed"
    zs_dir.mkdir()
    for fname in ("chats.jsonl", "messages.jsonl"):
        (zs_dir / fname).write_text((ROOT / "example" / fname).read_text(encoding="utf-8"), encoding="utf-8")
    (zs_dir / "seed.json").write_text(
        json.dumps({"circles": {}, "organisations": [], "chats": [], "goals": []}),
        encoding="utf-8",
    )
    ans_dir = zs_dir / "answers"
    ans_dir.mkdir()
    (ans_dir / "answers.json").write_text(
        (ROOT / "example" / "answers" / "answers.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    _, payload = run_pipeline(zs_dir)
    schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
    chats, messages, seed, answers = official_check.load_input(zs_dir)

    assert validate_json_against_schema(payload, schema) == []
    assert official_check.validate(payload, chats, messages) == []

    scores = official_check.score(payload, answers, seed, messages)
    assert scores["circles"] == 1.0
    assert scores["organisations"] == 1.0
    assert scores["chats"] == 1.0
    assert scores["goals"] == 1.0
    assert scores["loops_found"] == 1.0
    assert scores["loops_closed"] == 1.0
    assert scores["help"] == 1.0

    by_pid = {p["id"]: p for p in payload["people"]}
    assert "looks_after" not in by_pid["P02"]
    assert by_pid["P03"].get("looks_after") == ["P04"]


def test_llm_usage_log_env_appends_official_jsonl(tmp_path: Path, monkeypatch):
    """When $LLM_USAGE_LOG is set by the grading harness, every LLM call appends JSONL token usage."""
    usage_log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("LLM_USAGE_LOG", str(usage_log))

    def fake_responder(stage: str, sys_p: str, usr_p: str):
        return {"ok": True}

    client = LLMClient(mock_responder=fake_responder)
    res = client.complete_json("test_stage", "sys prompt", "user prompt")
    assert res == {"ok": True}
    assert usage_log.exists()
    rows = [json.loads(line) for line in usage_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["prompt_tokens"] > 0
    assert rows[0]["completion_tokens"] >= 1


def test_competitor_traps_surname_company_the_kids_and_institutional_dm(tmp_path: Path):
    """Covers CodeInfinity1 data/dev traps: company-tagged contact sharing ME's surname,
    nanny caring for 'the kids' unnamed, and institutional 1:1 DM tied to organisation."""
    ddir = tmp_path / "comp_traps"
    ddir.mkdir()
    chats = [
        {"id": "C1", "name": "CloudNine Team", "is_group": True, "participants": ["ME", "P_COL"]},
        {"id": "C2", "name": "Aakash Sharma @CloudNine", "is_group": False, "participants": ["ME", "P_COL"]},
        {"id": "C3", "name": "Ghar", "is_group": True, "participants": ["ME", "P_SPOUSE", "P_KID", "P_NANNY"]},
        {"id": "C4", "name": "Oakridge School Admin", "is_group": False, "participants": ["ME", "P_ADMIN"]},
        {"id": "C5", "name": "Oakridge Grade 3 Parents", "is_group": True, "participants": ["ME", "P_SPOUSE", "P_ADMIN"]},
    ]
    msgs = [
        {"id": "M1", "chat_id": "C1", "sender_id": "ME", "sender_name": "Rohan Sharma", "ts": 1792460000, "kind": "text", "body": "Let's finalize the sprint roadmap before standup.", "quoted_id": None},
        {"id": "M2", "chat_id": "C2", "sender_id": "P_COL", "sender_name": "Aakash Sharma @CloudNine", "ts": 1792460600, "kind": "text", "body": "Merged the PR on GitHub, deploying to staging.", "quoted_id": None},
        {"id": "M3", "chat_id": "C3", "sender_id": "P_KID", "sender_name": " Vivaan", "ts": 1792461000, "kind": "text", "body": "Mama my school homework is almost done.", "quoted_id": None},
        {"id": "M4", "chat_id": "C3", "sender_id": "P_NANNY", "sender_name": "Lata nanny", "ts": 1792462000, "kind": "text", "body": "Didi picked up the kids from school and gave milk.", "quoted_id": None},
        {"id": "M5", "chat_id": "C4", "sender_id": "P_ADMIN", "sender_name": "Oakridge School Admin", "ts": 1792463000, "kind": "text", "body": "Term 2 tuition circular from Oakridge School office.", "quoted_id": None},
        {"id": "M6", "chat_id": "C5", "sender_id": "P_ADMIN", "sender_name": "Oakridge School Admin", "ts": 1792463600, "kind": "text", "body": "PTM schedule for Grade 3.", "quoted_id": None},
    ]
    (ddir / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats) + "\n", encoding="utf-8")
    (ddir / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8")
    (ddir / "seed.json").write_text(
        json.dumps({"circles": {"P_SPOUSE": "spouse"}, "organisations": [], "chats": [], "goals": []}),
        encoding="utf-8",
    )

    _, payload = run_pipeline(ddir)
    by_pid = {p["id"]: p for p in payload["people"]}
    by_cid = {c["chat_id"]: c for c in payload["chats"]}

    # Shared surname ("Sharma") must NOT override @CloudNine work group colleague!
    assert by_pid["P_COL"]["circle"] == "colleague"
    # "picked up the kids" must link nanny's looks_after to P_KID even without naming "Vivaan"
    assert by_pid["P_KID"]["circle"] == "child"
    assert by_pid["P_NANNY"]["circle"] == "staff"
    assert "P_KID" in by_pid["P_NANNY"].get("looks_after", [])
    # Institutional 1:1 chat ("Oakridge School Admin") ties to the school organisation
    assert "organisation" in by_cid["C4"]


def test_pan_indian_marathi_and_tamil_unicode_loops_and_live_llm_guard(tmp_path: Path):
    """Marathi Devanagari + Tamil Unicode loops extract/close cleanly, and post-LLM guards block hallucinated done on deferrals."""
    ddir = tmp_path / "pan_indian"
    ddir.mkdir()
    chats = [
        {"id": "C_MR", "name": "CA Kulkarni", "is_group": False, "participants": ["ME", "P_MR"]},
        {"id": "C_TA", "name": "Karthik Vendor", "is_group": False, "participants": ["ME", "P_TA"]},
        {"id": "C_DEF", "name": "Work Team", "is_group": True, "participants": ["ME", "P_C1"]},
    ]
    msgs = [
        {"id": "M1", "chat_id": "C_MR", "sender_id": "P_MR", "sender_name": "CA Kulkarni", "ts": 1792460000, "kind": "text", "body": "उद्या फाइल पाठवतो साहेब", "quoted_id": None},
        {"id": "M2", "chat_id": "C_MR", "sender_id": "P_MR", "sender_name": "CA Kulkarni", "ts": 1792490000, "kind": "text", "body": "फाइल पाठवले साहेब", "quoted_id": None},
        {"id": "M3", "chat_id": "C_TA", "sender_id": "P_TA", "sender_name": "Karthik Vendor", "ts": 1792461000, "kind": "text", "body": "நாளைக்கு பில் அனுப்புறேன்", "quoted_id": None},
        {"id": "M4", "chat_id": "C_TA", "sender_id": "P_TA", "sender_name": "Karthik Vendor", "ts": 1792491000, "kind": "text", "body": "பில் அனுப்பிட்டேன்", "quoted_id": None},
        {"id": "M5", "chat_id": "C_DEF", "sender_id": "P_C1", "sender_name": "Neha", "ts": 1792462000, "kind": "text", "body": "Should we sign the Zest retainer at 60k?", "quoted_id": None},
        {"id": "M6", "chat_id": "C_DEF", "sender_id": "ME", "sender_name": "Meera", "ts": 1792462600, "kind": "text", "body": "Let me think about Zest", "quoted_id": "M5"},
    ]
    (ddir / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats) + "\n", encoding="utf-8")
    (ddir / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8")
    (ddir / "seed.json").write_text(
        json.dumps({"circles": {"P_MR": "vendor", "P_TA": "vendor", "P_C1": "colleague"}, "organisations": [], "chats": [], "goals": []}),
        encoding="utf-8",
    )

    def adversarial_mock_llm(stage: str, sys_p: str, usr_p: str):
        if stage == "temporal_slow_verify":
            # Deliberately try to hallucinate `done` on M6 ("Let me think about Zest")!
            return {"status": "done", "closure_message_id": "M6", "confidence": 0.99}
        return {}

    _, payload = run_pipeline(ddir, mock_responder=adversarial_mock_llm)
    by_src = {l["source_msg"]: l for l in payload["open_loops"]}
    assert by_src["M1"]["status"] == "done" and by_src["M1"]["closed_by"] == "M2"
    assert by_src["M3"]["status"] == "done" and by_src["M3"]["closed_by"] == "M4"
    assert by_src["M5"]["status"] == "open" and by_src["M5"]["closed_by"] is None
