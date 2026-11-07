"""Pass-6 regressions from competitor audits and real WhatsApp export edge cases."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import check as official_check
from src.conversation_index import ConversationIndex
from src.data_loader import load_dataset
from src.llm_client import LLMClient
from src.loop_extraction import resolve_due_date
from src.output_builder import run_pipeline
from src.relationship_inference import extract_org_name_tokens


def _write_dataset(
    root: Path,
    chats: list[dict],
    messages: list[dict],
    seed: dict | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "chats.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in chats) + "\n",
        encoding="utf-8",
    )
    (root / "messages.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in messages) + "\n",
        encoding="utf-8",
    )
    (root / "seed.json").write_text(
        json.dumps(seed or {"circles": {}, "organisations": [], "chats": [], "goals": []}),
        encoding="utf-8",
    )
    return root


def _msg(mid: str, cid: str, sender: str, body: str, *, ts: str, kind: str = "text", quoted: str | None = None) -> dict:
    return {
        "id": mid,
        "chat_id": cid,
        "sender_id": sender,
        "sender_name": sender,
        "ts": ts,
        "kind": kind,
        "body": body,
        "quoted_id": quoted,
    }


def test_departed_sender_and_notes_to_self_are_check_py_safe(tmp_path: Path):
    chats = [
        {"chat_id": "C_DM", "name": "P1", "is_group": False, "participants": ["ME", "P1"]},
        {"chat_id": "C_SELF", "name": "Notes to self", "is_group": False, "participants": ["ME"]},
    ]
    messages = [
        _msg("M_OLD", "C_DM", "P99", "I will send the audit invoice tomorrow.", ts="2026-11-05T10:00:00Z"),
        _msg("M_PHOTO", "C_SELF", "ME", "[photo: IMG_001.jpg]", ts="2026-11-05T10:01:00Z"),
        _msg("M_BDAY", "C_SELF", "ME", "Happy birthday Riya, have a wonderful year!", ts="2026-11-05T10:02:00Z"),
    ]
    data = _write_dataset(tmp_path / "departed", chats, messages)

    index = ConversationIndex(load_dataset(data))
    assert index.all_person_ids() == ["P1"]

    _, payload = run_pipeline(data)
    assert "P99" not in {p["id"] for p in payload["people"]}
    assert all(loop["owner"] != "P99" for loop in payload["open_loops"])
    self_chat = next(c for c in payload["chats"] if c["chat_id"] == "C_SELF")
    assert "organisation" in self_chat
    personal_org = next(o for o in payload["organisations"] if o["id"] == self_chat["organisation"])
    assert personal_org["side"] == "home"
    assert official_check.validate(payload, chats, messages) == []


def test_numeric_whatsapp_mention_resolves_group_followup_owner(tmp_path: Path):
    chats = [
        {"chat_id": "C_GROUP", "name": "Project Group", "is_group": True, "participants": ["ME", "P1", "P2"]},
        {"chat_id": "C_DM", "name": "P2", "is_group": False, "participants": ["ME", "P2"]},
    ]
    messages = [
        _msg("M_TASK", "C_GROUP", "P1", "@919876543210 please send the signed invoice by tmrw", ts="2026-10-20T10:00:00Z"),
        _msg("M_HANDLE", "C_DM", "ME", "Hi @919876543210, thanks for the update.", ts="2026-10-20T10:05:00Z"),
    ]
    data = _write_dataset(tmp_path / "mention", chats, messages)
    index = ConversationIndex(load_dataset(data))
    assert index.mention_handle_owner["919876543210"] == "P2"

    _, payload = run_pipeline(data)
    loop = next(loop for loop in payload["open_loops"] if loop["source_msg"] == "M_TASK")
    assert loop["owner"] == "P2"
    assert loop["due"] == "2026-10-21"
    assert official_check.validate(payload, chats, messages) == []


def test_media_and_birthday_wishes_do_not_open_loops_but_document_closes(tmp_path: Path):
    chats = [{"chat_id": "C_DM", "name": "P1", "is_group": False, "participants": ["ME", "P1"]}]
    messages = [
        _msg("M_ASK", "C_DM", "P1", "Can you send the signed agreement today?", ts="2026-11-05T09:00:00Z"),
        _msg("M_DOC", "C_DM", "ME", "[document: signed_agreement.pdf]", ts="2026-11-05T10:00:00Z", kind="document", quoted="M_ASK"),
        _msg("M_PHOTO", "C_DM", "P1", "[photo: IMG_001.jpg]", ts="2026-11-05T11:00:00Z"),
        _msg("M_HBD", "C_DM", "P1", "Happy birthday Asha!", ts="2026-11-05T12:00:00Z"),
    ]
    data = _write_dataset(tmp_path / "media", chats, messages)
    _, payload = run_pipeline(data)

    by_source = {loop["source_msg"]: loop for loop in payload["open_loops"]}
    assert by_source["M_ASK"]["status"] == "done"
    assert by_source["M_ASK"]["closed_by"] == "M_DOC"
    assert "M_PHOTO" not in by_source
    assert "M_HBD" not in by_source
    assert official_check.validate(payload, chats, messages) == []


def test_short_indian_dates_tmrw_and_llm_usage_default(tmp_path: Path, monkeypatch):
    sent = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    assert resolve_due_date("Please pay by 07/11", sent) == "2026-11-07"
    assert resolve_due_date("Please send tmrw", sent) == "2026-11-07"

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_USAGE_LOG", raising=False)
    LLMClient()._append_usage_log(
        "test-model",
        {"usage": {"prompt_tokens": 12, "completion_tokens": 4}},
        0.25,
    )
    row = json.loads((tmp_path / "usage.jsonl").read_text(encoding="utf-8"))
    assert row["model"] == "test-model"
    assert row["prompt_tokens"] == 12
    assert row["completion_tokens"] == 4


def test_place_named_staff_and_elder_beta_cue_infer_supported_orgs(tmp_path: Path):
    chats = [
        {"chat_id": "C_HOME", "name": "Ramesh Home", "is_group": False, "participants": ["ME", "P_HOME"]},
        {"chat_id": "C_OFFICE", "name": "Sunil Office", "is_group": False, "participants": ["ME", "P_OFFICE"]},
        {"chat_id": "C_FARM", "name": "Kishan Farmhouse", "is_group": False, "participants": ["ME", "P_FARM"]},
        {"chat_id": "C_PARENT", "name": "Auntie", "is_group": False, "participants": ["ME", "P_PARENT"]},
    ]
    messages = [
        _msg("M_HOME", "C_HOME", "P_HOME", "Good morning, I am available today.", ts="2026-11-05T09:00:00Z"),
        _msg("M_OFFICE", "C_OFFICE", "P_OFFICE", "Good morning, I am at the office.", ts="2026-11-05T09:01:00Z"),
        _msg("M_FARM", "C_FARM", "P_FARM", "Good morning, I reached the farmhouse.", ts="2026-11-05T09:02:00Z"),
        _msg("M_PARENT", "C_PARENT", "P_PARENT", "Ok beta, God bless you.", ts="2026-11-05T09:03:00Z"),
    ]
    seed = {
        "circles": {"P_HOME": "staff", "P_OFFICE": "staff", "P_FARM": "staff"},
        "organisations": [],
        "chats": [],
        "goals": [],
    }
    data = _write_dataset(tmp_path / "places", chats, messages, seed)
    _, payload = run_pipeline(data)

    org_types = {(o["name"].lower(), o["type"], o["side"]) for o in payload["organisations"]}
    assert ("home", "residence", "home") in org_types
    assert ("office", "workspace", "work") in org_types
    assert ("farmhouse", "vacation home", "home") in org_types
    by_person = {p["id"]: p for p in payload["people"]}
    assert by_person["P_PARENT"]["circle"] == "parent"
    for pid in ("P_HOME", "P_OFFICE", "P_FARM"):
        assert by_person[pid].get("links")


def test_camelcase_acronym_core_group_and_large_group_lurker_guard(tmp_path: Path):
    assert "pma" in extract_org_name_tokens("PostMyAd")
    assert "bma" in extract_org_name_tokens("Blue Moon Ads")

    participants = ["ME", "P_A", "P_B", "P_C", "P_LURKER"] + [f"P_F{i}" for i in range(1, 8)]
    chats = [
        {"chat_id": "C_PMA", "name": "PMA Core Team", "is_group": True, "participants": ["ME", "P_COL"]},
        {"chat_id": "C_ACME", "name": "Acme Core Team", "is_group": True, "participants": participants},
        {"chat_id": "C_EVERGREEN", "name": "Project Evergreen Sync", "is_group": True, "participants": ["ME", "P_A", "P_B", "P_C"]},
    ]
    messages = [
        _msg("M_PMA", "C_PMA", "P_COL", "Let's review the product launch roadmap.", ts="2026-11-05T09:00:00Z"),
    ]
    for pid in ("P_A", "P_B", "P_C"):
        for n in range(3):
            messages.append(
                _msg(f"M_{pid}_{n}", "C_ACME", pid, f"Please review the Acme sprint task {n}.", ts=f"2026-11-05T10:{n:02d}:00Z")
            )
    seed = {
        "circles": {"P_COL": "colleague", "P_A": "colleague", "P_B": "colleague", "P_C": "colleague"},
        "organisations": [
            {"id": "org_pma", "name": "PostMyAd", "type": "company", "side": "work", "role": "owner"},
            {"id": "org_acme", "name": "Acme", "type": "company", "side": "work", "role": "owner"},
        ],
        "chats": [],
        "goals": [],
    }
    data = _write_dataset(tmp_path / "orgs", chats, messages, seed)
    _, payload = run_pipeline(data)

    by_chat = {c["chat_id"]: c for c in payload["chats"]}
    assert by_chat["C_PMA"].get("organisation") == "org_pma"
    assert by_chat["C_ACME"].get("organisation") == "org_acme"
    assert by_chat["C_EVERGREEN"].get("organisation") == "org_acme"
    lurker = next(p for p in payload["people"] if p["id"] == "P_LURKER")
    assert lurker["circle"] == "colleague"
    assert not any(link["organisation"] == "org_acme" for link in lurker.get("links", []))
    acme = next(o for o in payload["organisations"] if o["id"] == "org_acme")
    assert "P_LURKER" not in acme.get("people", [])
    assert official_check.validate(payload, chats, messages) == []
