"""Identity-independent regressions for task grounding and typed delivery."""
import json
from pathlib import Path

import pytest

from src.conversation_index import ConversationIndex
from src.data_loader import load_dataset
from src.output_builder import run_pipeline


def dataset(root: Path, chats, turns, seed=None):
    root.mkdir()
    messages = [dict(id=f"m{i}", chat_id=cid, sender_id=sender,
                     sender_name=sender, ts=1793851200 + i * 60,
                     kind="text", body=body, quoted_id=None)
                for i, (cid, sender, body) in enumerate(turns)]
    for name, rows in (("chats", chats), ("messages", messages)):
        (root / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf8")
    (root / "seed.json").write_text(json.dumps(seed or {"circles": {}, "organisations": [], "goals": []}), encoding="utf8")
    return root


def dm():
    return [dict(chat_id="c", name="Morgan", is_group=False, participants=["ME", "p"])]


@pytest.mark.parametrize("prompt,reply,status", [
    ("Send me the technician's phone number", "Morgan 9123456789", "done"),
    ("Send your office address", "https://maps.app.goo.gl/ExamplePin", "done"),
    ("Please buy 6 Atlas shares", "Bought 6 Atlas shares", "done"),
    ("Send your office address", "https://example.com/article", "open"),
    ("Send me the account statement", "Morgan 9123456789", "open"),
    ("Please buy 6 Atlas shares", "Not bought the Atlas shares yet", "open"),
    ("Send your office address", "Okay", "open"),
    ("Send me the technician's phone number", "9123456789 is the wrong number", "open"),
])
def test_typed_delivery_requires_corresponding_artifact(tmp_path, prompt, reply, status):
    root = dataset(tmp_path / "d", dm(), [("c", "p", prompt), ("c", "ME", reply)])
    _, out = run_pipeline(root)
    loop = next(l for l in out["open_loops"] if l["source_msg"] == "m0")
    assert loop["owner"] == "ME"
    assert loop["status"] == status


@pytest.mark.parametrize("body,sender,owner", [
    ("Pl get me a copy of the contract", "p", "ME"),
    ("Send your delivery address", "p", "ME"),
    ("Let's arrange a review call tomorrow", "ME", "ME"),
    ("Reverting tomorrow on this", "ME", "ME"),
    ("Need to find a plumber", "ME", "ME"),
    ("Invitation to be sent", "ME", "ME"),
    ("We need the signed document to process the application", "p", "ME"),
    ("You can go ahead and book the inspection", "p", "ME"),
])
def test_informal_requests_preserve_speaker_ownership(tmp_path, body, sender, owner):
    root = dataset(tmp_path / "d", dm(), [("c", sender, body)])
    _, out = run_pipeline(root)
    assert any(l["source_msg"] == "m0" and l["owner"] == owner for l in out["open_loops"])


def test_sentence_words_are_not_me_names_or_group_addressees(tmp_path):
    group = dict(chat_id="g", name="Operations", is_group=True,
                 participants=["ME"] + [f"p{i}" for i in range(10)])
    root = dataset(tmp_path / "d", [group], [
        ("g", "p0", "You can please check the delivery manifest"),
        ("g", "p1", "Also, please check the warehouse inventory"),
        ("g", "p2", "You can please check the packing sheet"),
        ("g", "p3", "I will send the warehouse inventory tomorrow"),
    ])
    index = ConversationIndex(load_dataset(root))
    assert not (index.me_aliases & {"you", "also"})
    _, out = run_pipeline(root)
    assert out["open_loops"] == []


def test_company_does_not_prevent_workspace_discovery(tmp_path):
    chats = [dict(chat_id="c", name="Morgan Office", is_group=False, participants=["ME", "p"])]
    root = dataset(tmp_path / "d", chats, [("c", "p", "I am at the office")],
                   {"circles": {"p": "staff"}, "organisations": ["Cedar Labs"], "goals": []})
    _, out = run_pipeline(root)
    workspaces = {o["id"] for o in out["organisations"] if o["type"] == "workspace"}
    assert len(workspaces) == 1
    person = next(p for p in out["people"] if p["id"] == "p")
    assert any(l["organisation"] in workspaces for l in person["links"])


def test_external_project_brand_does_not_make_customer_a_colleague(tmp_path):
    chats = [dict(chat_id="g", name="Harbor Retail: Cedar Labs - Ad Targeting", is_group=True,
                  participants=["ME", "p", "q"])]
    root = dataset(tmp_path / "d", chats, [("g", "p", "Hello everyone")],
                   {"circles": {"q": "colleague"}, "organisations": ["Cedar Labs"], "goals": []})
    _, out = run_pipeline(root)
    assert next(p for p in out["people"] if p["id"] == "p")["circle"] == "client"


def test_business_politeness_is_not_spouse_evidence(tmp_path):
    root = dataset(tmp_path / "d", dm(), [("c", "ME", "Would love to hear about your pilot project")])
    _, out = run_pipeline(root)
    assert next(p for p in out["people"] if p["id"] == "p")["circle"] != "spouse"


def test_negated_need_is_not_a_task(tmp_path):
    root = dataset(tmp_path / "d", dm(), [("c", "ME", "We do not need to book an inspection")])
    _, out = run_pipeline(root)
    assert out["open_loops"] == []
