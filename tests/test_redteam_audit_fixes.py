"""Red-team regression tests covering hidden-identity renaming, IST midnight timezones,
bystander group-closure guards, document closures, reasoning-model HTTP-400 downgrades,
enum schema adaptation, and 20,000-message / 320-chat scale performance."""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from scripts.generate_synthetic_suite import build_synthetic_benchmark
from src.conversation_index import ConversationIndex
from src.data_loader import load_dataset
from src.llm_client import LLMClient
from src.output_builder import run_pipeline
from src.schema_adapter import serialize_output_for_schema, validate_json_against_schema


def test_renamed_me_identity_and_colleague_named_arjun(tmp_path: Path):
    """When ME is renamed 'Karan' and a colleague is named 'Arjun', ME's alias is inferred dynamically
    and a group directive to 'Arjun' is assigned to that colleague, NEVER hardcoded to ME."""
    pdir = tmp_path / "renamed_hidden"
    build_synthetic_benchmark(pdir)

    # First replace ME's name "Arjun" -> "Karan", then rename colleague Rahul -> "Arjun"
    for fname in ("chats.jsonl", "messages.jsonl", "seed.json"):
        txt = (pdir / fname).read_text("utf-8")
        txt = txt.replace("Arjun", "Karan")
        txt = txt.replace("Rahul Verma", "Arjun Verma")
        txt = txt.replace("Rahul, please send the staging", "Arjun, please send the staging")
        (pdir / fname).write_text(txt, encoding="utf-8")

    bundle = load_dataset(pdir)
    index = ConversationIndex(bundle)
    assert "karan" in index.me_aliases
    assert "arjun" not in index.me_aliases

    out, _ = run_pipeline(pdir)
    # m135 ("Karan, society quarterly maintenance cheque...") -> owned by ME!
    m135 = next(l for l in out.open_loops if l.message_id == "m135")
    assert m135.owner == "ME"
    # m137 ("Arjun, please send the staging server AWS logs to Sneha") -> owned by p_colleague_1, NOT ME!
    m137 = next(l for l in out.open_loops if l.message_id == "m137")
    assert m137.owner == "p_colleague_1"
    assert m137.intervention is not None and m137.intervention.action == "leave"


def test_ist_midnight_timezone_preserves_local_calendar_date(tmp_path: Path):
    """A 01:30 AM +05:30 IST message saying 'tomorrow' resolves to next IST day (not UTC previous day!)."""
    ddir = tmp_path / "ist_tz"
    ddir.mkdir()
    (ddir / "chats.jsonl").write_text(
        json.dumps({"chat_id": "c1", "name": "Neha", "is_group": False, "participant_ids": ["ME", "p1"]}) + "\n",
        encoding="utf-8",
    )
    msgs = [
        {
            "id": "m1",
            "chat_id": "c1",
            "sender_id": "p1",
            "sender_name": "Neha",
            "ts": "2026-10-27T01:30:00+05:30",  # In UTC this would be 2026-10-26T20:00:00Z!
            "kind": "text",
            "body": "Please call the bank manager tomorrow.",
            "quoted_id": None,
        },
        {
            "id": "m2",
            "chat_id": "c1",
            "sender_id": "ME",
            "sender_name": "ME",
            "ts": "2026-10-27T01:35:00+05:30",
            "kind": "text",
            "body": "Sure noted",
            "quoted_id": "m1",
        },
    ]
    (ddir / "messages.jsonl").write_text(
        "\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8"
    )
    (ddir / "seed.json").write_text(
        json.dumps({"people": {"p1": "spouse"}, "organisations": [], "goals": []}),
        encoding="utf-8",
    )

    out, _ = run_pipeline(ddir)
    loop = next(l for l in out.open_loops if l.message_id == "m1")
    assert loop.due_date == "2026-10-28"
    assert loop.status == "open"
    assert loop.intervention is not None
    assert loop.intervention.remind_at is not None
    assert loop.intervention.remind_at.endswith("+05:30")


def test_bystander_and_unrelated_quote_cannot_close_or_cancel_group_loop(tmp_path: Path):
    """Unrelated thread cancellations ('Rehne do, chai mat banao') and bystander 'done' messages
    in group chats never falsely cancel or close an open loop."""
    ddir = tmp_path / "group_guard"
    ddir.mkdir()
    (ddir / "chats.jsonl").write_text(
        json.dumps(
            {
                "chat_id": "cg",
                "name": "Zephyr Core Team",
                "is_group": True,
                "participant_ids": ["ME", "p_rahul", "p_sneha", "p_dev"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    msgs = [
        {
            "id": "m_req",
            "chat_id": "cg",
            "sender_id": "ME",
            "sender_name": "ME",
            "ts": "2026-10-28T10:00:00",
            "kind": "text",
            "body": "Rahul, please send the Q4 client security invoice before Friday.",
            "quoted_id": None,
        },
        {
            "id": "m_other_topic",
            "chat_id": "cg",
            "sender_id": "p_sneha",
            "sender_name": "Sneha - Zephyr",
            "ts": "2026-10-28T10:05:00",
            "kind": "text",
            "body": "Should I order snacks for standup?",
            "quoted_id": None,
        },
        {
            "id": "m_unrelated_cancel",
            "chat_id": "cg",
            "sender_id": "ME",
            "sender_name": "ME",
            "ts": "2026-10-28T10:06:00",
            "kind": "text",
            "body": "Rehne do, no longer needed today.",
            "quoted_id": "m_other_topic",  # Quotes snacks, NOT m_req!
        },
        {
            "id": "m_bystander_done",
            "chat_id": "cg",
            "sender_id": "p_dev",  # Bystander Dev, NOT owner Rahul!
            "sender_name": "Dev - Zephyr",
            "ts": "2026-10-28T11:00:00",
            "kind": "text",
            "body": "Uploaded the vendor invoice on drive, done from my side.",
            "quoted_id": None,
        },
    ]
    (ddir / "messages.jsonl").write_text(
        "\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8"
    )
    (ddir / "seed.json").write_text(
        json.dumps(
            {
                "people": {"p_rahul": "colleague", "p_sneha": "colleague", "p_dev": "colleague"},
                "organisations": ["Zephyr"],
                "goals": [],
            }
        ),
        encoding="utf-8",
    )

    out, _ = run_pipeline(ddir)
    req_loop = next(l for l in out.open_loops if l.message_id == "m_req")
    assert req_loop.status == "open"
    assert req_loop.closure_message_id is None


def test_document_attachment_and_requester_receipt_close_loops(tmp_path: Path):
    """Deliverable `kind == 'document'` and requester receipt confirmations close loops,
    whereas forwarding a second unpaid bill PDF with a future commitment caption NEVER closes an earlier bill."""
    ddir = tmp_path / "doc_receipt"
    ddir.mkdir()
    chats = [
        {"chat_id": "c_ca", "name": "CA Manish", "is_group": False, "participant_ids": ["ME", "p_ca"]},
        {"chat_id": "c_law", "name": "Advocate Rao", "is_group": False, "participant_ids": ["ME", "p_law"]},
        {"chat_id": "c_home", "name": "Home Ops", "is_group": True, "participant_ids": ["ME", "p_spouse"]},
    ]
    msgs = [
        {"id": "m1", "chat_id": "c_ca", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-28T09:00:00", "kind": "text", "body": "Please send the October tax challan PDF.", "quoted_id": None},
        {"id": "m2", "chat_id": "c_ca", "sender_id": "p_ca", "sender_name": "CA Manish", "ts": "2026-10-28T14:00:00", "kind": "document", "body": "Oct_Tax_Challan_Signed.pdf", "quoted_id": "m1"},
        {"id": "m3", "chat_id": "c_law", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-28T10:00:00", "kind": "text", "body": "Please share the trademark filing receipt.", "quoted_id": None},
        {"id": "m4", "chat_id": "c_law", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-29T11:00:00", "kind": "text", "body": "Received the trademark receipt on email, thanks!", "quoted_id": None},
        {"id": "m5", "chat_id": "c_home", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-29T19:00:00", "kind": "document", "body": "I'll pay the October electricity bill via OTP before 3 Nov.", "quoted_id": None},
        {"id": "m6", "chat_id": "c_home", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-29T19:05:00", "kind": "document", "body": "I'll pay the October broadband bill via OTP before 4 Nov.", "quoted_id": None},
    ]
    (ddir / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats) + "\n", encoding="utf-8")
    (ddir / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8")
    (ddir / "seed.json").write_text(json.dumps({"people": {"p_ca": "vendor", "p_law": "vendor", "p_spouse": "spouse"}, "organisations": ["Home"], "goals": []}), encoding="utf-8")

    out, _ = run_pipeline(ddir)
    by_mid = {l.message_id: l for l in out.open_loops}
    assert by_mid["m1"].status == "done" and by_mid["m1"].closure_message_id == "m2"
    assert by_mid["m3"].status == "done" and by_mid["m3"].closure_message_id == "m4"
    assert by_mid["m5"].status == "open" and by_mid["m5"].closure_message_id is None
    assert by_mid["m6"].status == "open" and by_mid["m6"].closure_message_id is None


def test_llm_client_reasoning_model_http400_capability_downgrade():
    """Verify automatic downgrade from max_tokens -> max_completion_tokens and temperature removal on HTTP 400."""
    payloads_seen: List[Dict[str, Any]] = []

    class SubClient(LLMClient):
        def _post_http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            payloads_seen.append(dict(payload))
            if "max_tokens" in payload:
                raise RuntimeError("HTTP_400: Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'.")
            if "temperature" in payload:
                raise RuntimeError("HTTP_400: Unsupported value: 'temperature' does not support 0.0.")
            return {
                "choices": [{"message": {"content": "```json\n{\"ok\": 42}\n```"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }

    client = SubClient(base_url="https://example.invalid/v1", api_key="test-key", model="custom-reasoner", max_retries=3)
    res = client.complete_json("test", "sys", "usr", max_tokens=200)
    assert res == {"ok": 42}
    assert len(payloads_seen) == 3
    assert "max_completion_tokens" in payloads_seen[-1]
    assert "temperature" not in payloads_seen[-1]


def test_llm_client_cost_budget_ceiling():
    """Verify LLMClient stops issuing calls cleanly once LLM_MAX_COST_USD ceiling is reached."""
    calls = {"n": 0}

    class ExpensiveClient(LLMClient):
        def _post_http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            calls["n"] += 1
            return {
                "choices": [{"message": {"content": "{\"done\": true}"}}],
                "usage": {"prompt_tokens": 500_000, "completion_tokens": 200_000},
            }

    client = ExpensiveClient(base_url="https://example.invalid/v1", api_key="k", model="gpt-4.1")
    client.max_cost_usd = 0.50
    assert client.complete_json("s1", "sys", "u1") == {"done": True}
    # Second distinct prompt exceeds $0.50 budget -> returns None without making network call!
    assert client.complete_json("s1", "sys", "u2") is None
    assert calls["n"] == 1


def test_schema_adapter_us_spelling_and_enum_normalization(tmp_path: Path):
    """Verify US 'organizations' key and 'in_law' enum spelling adapt automatically."""
    ddir = tmp_path / "enum_adapt"
    build_synthetic_benchmark(ddir)
    out, _ = run_pipeline(ddir)

    custom_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["people", "organizations", "chats", "goals", "open_loops"],
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "circle"],
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "circle": {
                            "type": "string",
                            "enum": [
                                "spouse", "child", "parent", "in_law", "sibling", "relative",
                                "friend", "colleague", "client", "vendor", "partner", "staff",
                            ],
                        },
                        "organisation_links": {"type": "array"},
                        "supports_people": {"type": "array"},
                        "evidence": {"type": "array"},
                    },
                },
            },
            "organizations": {"type": "array", "items": {"type": "object"}},
            "chats": {"type": "array", "items": {"type": "object"}},
            "goals": {"type": "array", "items": {"type": "object"}},
            "open_loops": {"type": "array", "items": {"type": "object"}},
        },
        "additionalProperties": False,
    }
    adapted = serialize_output_for_schema(out, custom_schema)
    assert validate_json_against_schema(adapted, custom_schema) == []
    inlaw = next(p for p in adapted["people"] if p["id"] == "p_inlaw_dad")
    assert inlaw["circle"] == "in_law"


def test_20k_message_320_chat_scale_benchmark_under_6_seconds(tmp_path: Path):
    """Full end-to-end scale test: 20,000 messages across 320 chats completes well under 6s."""
    sdir = tmp_path / "scale_20k"
    build_synthetic_benchmark(sdir)

    rng = random.Random(2026)
    chats = [json.loads(ln) for ln in (sdir / "chats.jsonl").read_text("utf-8").splitlines() if ln.strip()]
    msgs = [json.loads(ln) for ln in (sdir / "messages.jsonl").read_text("utf-8").splitlines() if ln.strip()]

    # Expand to 320 chats and 20,000 messages across 8 weeks
    for idx in range(len(chats), 320):
        pid = f"p_scale_{idx:03d}"
        chats.append(
            {
                "chat_id": f"c_scale_{idx:03d}",
                "name": f"Contact {idx}",
                "is_group": (idx % 5 == 0),
                "participant_ids": ["ME", pid] if idx % 5 != 0 else ["ME", pid, f"p_scale_{(idx - 1):03d}"],
            }
        )

    templates = [
        "Good morning, hope things are going well!",
        "Traffic near Silk Board is packed right now.",
        "Haan bilkul, milte hain weekend pe.",
        "Ok noted 👍",
        "Check this photo from dinner yesterday.",
        "Shubho Bijoya! Pranam.",
    ]
    base_dt = datetime(2026, 9, 8, 8, 0, 0)
    for m_idx in range(len(msgs), 20_000):
        c = chats[m_idx % len(chats)]
        sender = c["participant_ids"][m_idx % len(c["participant_ids"])]
        day_off = (m_idx * 42) // 20_000  # First 6 weeks background + sparse final 2 weeks
        ts_str = f"2026-09-{8 + (day_off % 22):02d}T{(8 + (m_idx % 12)):02d}:{(m_idx % 60):02d}:00"
        msgs.append(
            {
                "id": f"ms_{m_idx:05d}",
                "chat_id": c["chat_id"],
                "sender_id": sender,
                "sender_name": "ME" if sender == "ME" else f"Contact {sender}",
                "ts": ts_str,
                "kind": "text",
                "body": templates[m_idx % len(templates)],
                "quoted_id": None,
            }
        )

    (sdir / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats) + "\n", encoding="utf-8")
    (sdir / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    out, payload = run_pipeline(sdir)
    elapsed = time.perf_counter() - t0

    assert len(payload["chats"]) == 320
    assert out.telemetry.total_messages == 20_000
    assert elapsed < 6.0, f"Expected <6.0s on 20k messages, took {elapsed:.2f}s"
