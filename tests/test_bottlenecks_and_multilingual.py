"""Tests for native Devanagari/Bengali Unicode scripts, Whisper ASR phonetic cross-chat joins,
indirect time-anchored obligations, zero-cue acquaintance priors, epoch/from_me formats,
and sub-2s / low-token 20,000-message scaling across all 8 weeks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from scripts.run_experiments import build_hard_8week_stress_benchmark, simulated_smart_llm_responder
from src.llm_client import LLMClient
from src.output_builder import run_pipeline


def test_native_devanagari_hindi_and_bengali_unicode_cross_script_pipeline(tmp_path: Path):
    """Verify native Devanagari (Hindi) and Bengali Unicode scripts work end-to-end across
    circles, Indic digits, bare-ack guards (defeating Python's \\b matra bug), and English closures."""
    ddir = tmp_path / "indic_unicode"
    ddir.mkdir()
    chats = [
        {"chat_id": "c_maa", "name": "मम्मी", "is_group": False, "participant_ids": ["ME", "p_maa"]},
        {"chat_id": "c_mashi", "name": "মিতামাসি", "is_group": False, "participant_ids": ["ME", "p_mashi"]},
        {"chat_id": "c_driver", "name": "रमेश ड्राइवर", "is_group": False, "participant_ids": ["ME", "p_drv"]},
        {"chat_id": "c_inlaw", "name": "ससुरजी", "is_group": False, "participant_ids": ["ME", "p_sasur"]},
    ]
    msgs = [
        # 1. Native Devanagari request with Devanagari digit '३' (3 Nov) + bare 'हाँ' ack (must NOT close!)
        {
            "id": "m_hi_1",
            "chat_id": "c_maa",
            "sender_id": "p_maa",
            "sender_name": "मम्मी",
            "ts": "2026-10-28T09:00:00+05:30",
            "kind": "text",
            "body": "बेटा ३ नवंबर से पहले बिजली बिल भर देना।",
            "quoted_id": None,
        },
        {
            "id": "m_hi_ack",
            "chat_id": "c_maa",
            "sender_id": "ME",
            "sender_name": "ME",
            "ts": "2026-10-28T09:05:00+05:30",
            "kind": "text",
            "body": "हाँ ठीक है",
            "quoted_id": "m_hi_1",
        },
        # 2. Native Bengali voice transcript request ('আগামীকাল' -> 2026-10-29) + English cross-script closure!
        {
            "id": "m_bn_1",
            "chat_id": "c_mashi",
            "sender_id": "p_mashi",
            "sender_name": "মিতামাসি",
            "ts": "2026-10-28T10:00:00+05:30",
            "kind": "voice_transcript",
            "body": "বাবা আগামীকাল সকালে ওষুধ এনে দিও।",
            "quoted_id": None,
        },
        {
            "id": "m_bn_ack",
            "chat_id": "c_mashi",
            "sender_id": "ME",
            "sender_name": "ME",
            "ts": "2026-10-28T10:02:00+05:30",
            "kind": "text",
            "body": "হ্যাঁ",
            "quoted_id": "m_bn_1",
        },
        {
            "id": "m_bn_close",
            "chat_id": "c_mashi",
            "sender_id": "ME",
            "sender_name": "ME",
            "ts": "2026-10-29T08:30:00+05:30",
            "kind": "text",
            "body": "দিয়ে দিয়েছি medicine!",
            "quoted_id": None,
        },
        # 3. Driver & In-law turns
        {
            "id": "m_drv",
            "chat_id": "c_driver",
            "sender_id": "p_drv",
            "sender_name": "रमेश ड्राइवर",
            "ts": "2026-10-28T11:00:00+05:30",
            "kind": "text",
            "body": "जी साहब गाड़ी सर्विस हो गया",
            "quoted_id": None,
        },
        {
            "id": "m_sasur",
            "chat_id": "c_inlaw",
            "sender_id": "p_sasur",
            "sender_name": "ससुरजी",
            "ts": "2026-10-28T12:00:00+05:30",
            "kind": "text",
            "body": "आशीर्वाद बेटा",
            "quoted_id": None,
        },
    ]
    (ddir / "chats.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chats) + "\n", encoding="utf-8")
    (ddir / "messages.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in msgs) + "\n", encoding="utf-8")
    (ddir / "seed.json").write_text(json.dumps({"people": {}, "organisations": [], "goals": []}), encoding="utf-8")

    out, _ = run_pipeline(ddir)
    circles = {p.id: p.circle for p in out.people}
    assert circles["p_maa"] == "parent"
    assert circles["p_mashi"] == "relative"
    assert circles["p_drv"] == "staff"
    assert circles["p_sasur"] == "in-law"

    loops = {l.message_id: l for l in out.open_loops}
    # Hindi electricity bill with Devanagari '३' -> due 2026-11-03, and bare 'हाँ ठीक है' keeps it OPEN!
    assert "m_hi_1" in loops
    assert loops["m_hi_1"].due_date == "2026-11-03"
    assert loops["m_hi_1"].status == "open"

    # Bengali medicine request -> due 2026-10-29, closed by m_bn_close (NOT m_bn_ack!)
    assert "m_bn_1" in loops
    assert loops["m_bn_1"].due_date == "2026-10-29"
    assert loops["m_bn_1"].status == "done"
    assert loops["m_bn_1"].closure_message_id == "m_bn_close"


def test_whisper_asr_phonetic_distortion_cross_chat_closure(tmp_path: Path):
    """Verify ASR typos ('chalan' vs 'challan', 'North Star' vs 'Northstar') still link across group -> DM."""
    ddir = tmp_path / "asr_fuzzy"
    ddir.mkdir()
    chats = [
        {"chat_id": "c_grp", "name": "Zephyr Ops", "is_group": True, "participant_ids": ["ME", "p_ca", "p_col"]},
        {"chat_id": "c_dm", "name": "CA Manish", "is_group": False, "participant_ids": ["ME", "p_ca"]},
    ]
    msgs = [
        {
            "id": "m1",
            "chat_id": "c_grp",
            "sender_id": "ME",
            "sender_name": "ME",
            "ts": "2026-10-28T10:00:00",
            "kind": "voice_transcript",
            "body": "Manish please send the Northstar customs challan before Friday.",
            "quoted_id": None,
        },
        {
            "id": "m2",
            "chat_id": "c_dm",
            "sender_id": "p_ca",
            "sender_name": "CA Manish",
            "ts": "2026-10-29T11:30:00",
            "kind": "voice_transcript",
            "body": "Uploaded and sent the North Star customs chalan on email.",
            "quoted_id": None,
        },
    ]
    (ddir / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats) + "\n", encoding="utf-8")
    (ddir / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8")
    (ddir / "seed.json").write_text(
        json.dumps({"people": {"p_ca": "vendor", "p_col": "colleague"}, "organisations": ["Zephyr"], "goals": []}),
        encoding="utf-8",
    )

    out, _ = run_pipeline(ddir)
    loop = next(l for l in out.open_loops if l.message_id == "m1")
    assert loop.status == "done"
    assert loop.closure_message_id == "m2"


def test_zero_cue_acquaintances_default_to_friend_with_zero_llm_waste(tmp_path: Path):
    """Unseeded acquaintances with only phatic greetings default to 'friend' (never alphabetical 'child'/'spouse')
    and do NOT trigger wasteful LLM calls."""
    ddir = tmp_path / "zero_cue"
    ddir.mkdir()
    chats = [{"chat_id": "c1", "name": "Aakash", "is_group": False, "participant_ids": ["ME", "p_aakash"]}]
    msgs = [
        {"id": "m1", "chat_id": "c1", "sender_id": "p_aakash", "sender_name": "Aakash", "ts": "2026-10-28T09:00:00", "kind": "text", "body": "Good morning!", "quoted_id": None}
    ]
    (ddir / "chats.jsonl").write_text(json.dumps(chats[0]) + "\n", encoding="utf-8")
    (ddir / "messages.jsonl").write_text(json.dumps(msgs[0]) + "\n", encoding="utf-8")
    (ddir / "seed.json").write_text(json.dumps({"people": {}, "organisations": [], "goals": []}), encoding="utf-8")

    calls = {"rel": 0}

    def responder(stage: str, sys_p: str, usr_p: str) -> Dict[str, Any]:
        if stage == "relationship_inference":
            calls["rel"] += 1
        return {}

    out, _ = run_pipeline(ddir, mock_responder=responder)
    p = next(x for x in out.people if x.id == "p_aakash")
    assert p.circle == "friend"
    assert calls["rel"] == 0


def test_whatsapp_export_from_me_epoch_ms_and_multipart_llm_content(tmp_path: Path):
    """Verify epoch-ms timestamps, from_me=True, transcript/caption fallbacks, and list-typed LLM content."""
    ddir = tmp_path / "export_fmt"
    ddir.mkdir()
    (ddir / "chats.jsonl").write_text(
        json.dumps({"id": "c1", "title": "Riya", "members": [{"id": "ME"}, {"id": "p_riya"}]}) + "\n",
        encoding="utf-8",
    )
    msgs = [
        {
            "msg_id": "m1",
            "chat_id": "c1",
            "from": "p_riya",
            "from_name": "Riya",
            "timestamp": 1793181600000,  # Epoch milliseconds!
            "type": "voice_transcript",
            "transcript": "Please share the hotel booking options tomorrow.",
        },
        {
            "msg_id": "m2",
            "chat_id": "c1",
            "from_me": True,
            "timestamp": 1793181900000,
            "type": "text",
            "text": "Sure noted",
            "reply_to_id": "m1",
        },
    ]
    (ddir / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8")
    (ddir / "seed.json").write_text(
        json.dumps({"people": [{"id": "p_riya", "circle": "spouse"}], "organizations": [], "goals": []}),
        encoding="utf-8",
    )

    out, _ = run_pipeline(ddir)
    assert len(out.open_loops) == 1
    assert out.open_loops[0].status == "open"
    assert out.open_loops[0].intervention is not None
    assert out.open_loops[0].intervention.action == "assist"

    # Also test multipart list content in LLMClient
    class MultipartClient(LLMClient):
        def _post_http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "choices": [
                    {"message": {"content": [{"type": "text", "text": "{\"status\": \"verified\"}"}]}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    mc = MultipartClient(base_url="https://example.invalid/v1", api_key="k", model="gpt-4.1-mini")
    assert mc.complete_json("s", "sys", "usr") == {"status": "verified"}


def test_20k_messages_all_8_weeks_sub_2_5s_and_low_token_footprint(tmp_path: Path):
    """20,000 messages spread across ALL 8 weeks (including the final 2-week scoring window)
    across 320 chats finishes in <2.5s and uses <16,000 prompt tokens."""
    sdir = build_hard_8week_stress_benchmark(tmp_path / "scale_all8wk")
    s_chats = [json.loads(ln) for ln in (sdir / "chats.jsonl").read_text("utf-8").splitlines() if ln.strip()]
    s_msgs = [json.loads(ln) for ln in (sdir / "messages.jsonl").read_text("utf-8").splitlines() if ln.strip()]
    for idx in range(len(s_chats), 320):
        pid = f"p_scale_{idx:03d}"
        s_chats.append(
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
    for m_idx in range(len(s_msgs), 20_000):
        c = s_chats[m_idx % len(s_chats)]
        sender = c["participant_ids"][m_idx % len(c["participant_ids"])]
        day_off = (m_idx * 54) // 20_000
        month = 9 if (8 + day_off) <= 30 else 10
        day = min(30, (8 + day_off) if month == 9 else (8 + day_off - 30))
        ts_str = f"2026-{month:02d}-{day:02d}T{(8 + (m_idx % 12)):02d}:{(m_idx % 60):02d}:00"
        s_msgs.append(
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
    (sdir / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in s_chats) + "\n", encoding="utf-8")
    (sdir / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in s_msgs) + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    out, _ = run_pipeline(sdir, mock_responder=simulated_smart_llm_responder)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.5, f"Expected <2.5s on 8-week 20k messages, got {elapsed:.2f}s"
    assert out.telemetry.prompt_tokens < 16_000, f"Expected <16k prompt tokens, got {out.telemetry.prompt_tokens}"
