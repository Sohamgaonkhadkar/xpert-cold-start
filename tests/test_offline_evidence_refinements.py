"""Identity-independent regressions for the final offline evidence refinements."""
import json
from pathlib import Path

import pytest

from src.conversation_index import ConversationIndex
from src.data_loader import load_dataset
from src.output_builder import run_pipeline
from src.relationship_inference import _explicit_family_mentions, build_contact_dossiers


def fixture(root: Path, chats, turns, seed=None):
    root.mkdir()
    messages = [dict(id=f"m{i}", chat_id=cid, sender_id=sender, sender_name=sender,
                     ts=1793851200 + i * 60, kind="text", body=body, quoted_id=None)
                for i, (cid, sender, body) in enumerate(turns)]
    for name, rows in (("chats", chats), ("messages", messages)):
        (root / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf8")
    (root / "seed.json").write_text(json.dumps(seed or {"circles": {}, "organisations": [], "goals": []}), encoding="utf8")
    return root


def chats():
    return [dict(chat_id="target", name="Morgan Patel", is_group=False, participants=["ME", "p"]),
            dict(chat_id="other", name="Casey Rao", is_group=False, participants=["ME", "q"])]


@pytest.mark.parametrize("role,circle", [("brother", "sibling"), ("sister", "sibling"), ("mother", "parent"), ("husband", "spouse")])
@pytest.mark.parametrize("name", ["Morgan Patel", "Avery Chen"])
def test_explicit_family_introduction_in_different_chat(tmp_path, role, circle, name):
    rows = chats(); rows[0]["name"] = name
    root = fixture(tmp_path / "d", rows, [("target", "p", "Hello"),
                   ("other", "ME", f"My {role} is {name}, here are the details")])
    _, out = run_pipeline(root)
    assert next(p for p in out["people"] if p["id"] == "p")["circle"] == circle


@pytest.mark.parametrize("sender,text", [
    ("q", "My brother is Morgan Patel"),
    ("ME", "My friend's brother is Morgan Patel"),
    ("ME", 'He said: "my brother is Morgan Patel"'),
    ("ME", "My brother is Morgan, call him"),
    ("ME", "My brother is not Morgan Patel"),
    ("ME", "My brother is Morgan Patel's colleague"),
    ("ME", "My brother is Morgan Patel’s friend"),
])
def test_family_evidence_does_not_use_others_relatives_quotes_or_partial_names(tmp_path, sender, text):
    root = fixture(tmp_path / "d", chats(), [("other", sender, text)])
    index = ConversationIndex(load_dataset(root))
    assert "p" not in _explicit_family_mentions(index, build_contact_dossiers(index))


def test_conflicting_family_statements_do_not_form_an_explicit_anchor(tmp_path):
    root = fixture(tmp_path / "d", chats(), [("other", "ME", "My brother is Morgan Patel"),
                   ("other", "ME", "My father is Morgan Patel")])
    index = ConversationIndex(load_dataset(root))
    assert "p" not in _explicit_family_mentions(index, build_contact_dossiers(index))


def test_duplicate_full_names_do_not_select_first_person(tmp_path):
    rows = chats(); rows.append(dict(chat_id="duplicate", name="Morgan Patel", is_group=False, participants=["ME", "r"]))
    root = fixture(tmp_path / "d", rows, [("other", "ME", "My brother is Morgan Patel")])
    index = ConversationIndex(load_dataset(root))
    evidence = _explicit_family_mentions(index, build_contact_dossiers(index))
    assert not ({"p", "r"} & evidence.keys())


def test_family_introduction_does_not_override_seed(tmp_path):
    root = fixture(tmp_path / "d", chats(), [("other", "ME", "My brother is Morgan Patel")],
                   {"circles": {"p": "relative"}, "organisations": [], "goals": []})
    _, out = run_pipeline(root)
    assert next(p for p in out["people"] if p["id"] == "p")["circle"] == "relative"


@pytest.mark.parametrize("reply,status", [
    ("Here is the campaign update: the audience is growing steadily and conversion quality is improving across the tested locations.", "done"),
    ("Thanks for asking about the campaign and taking such an interest in the project; I really appreciate your support today.", "open"),
    ("The campaign has no useful update yet because we need more time to examine the numbers and prepare an answer.", "open"),
    ("I will check the campaign results and send the detailed answer when we have reviewed all of the reports.", "open"),
    ("The warehouse inventory is now available to the accounting team and the stock numbers are visible for every location.", "open"),
    ("Here is the campaign update? Could we talk first about the campaign objectives before we look at the audience figures?", "open"),
    ("Here is the campaign update part 1: the audience is growing steadily and conversion quality is improving across the tested locations.", "open"),
])
def test_information_promise_requires_substantive_on_topic_answer(tmp_path, reply, status):
    root = fixture(tmp_path / "d", chats(), [
        ("other", "q", "Any initial feedback on the campaign"),
        ("other", "ME", "Reverting tomorrow on this."),
        ("other", "ME", reply),
    ])
    _, out = run_pipeline(root)
    loop = next(l for l in out["open_loops"] if l["source_msg"] == "m1")
    assert loop["status"] == status
    if status == "done":
        assert loop["closed_by"] == "m2"


def test_reply_does_not_complete_underlying_repair_promise(tmp_path):
    reply = "Here is the campaign update: the audience is growing steadily and conversion quality is improving across the tested locations."
    root = fixture(tmp_path / "d", chats(), [("other", "ME", "I will fix the campaign tomorrow"), ("other", "ME", reply)])
    _, out = run_pipeline(root)
    assert next(l for l in out["open_loops"] if l["source_msg"] == "m0")["status"] == "open"


def test_completed_preparation_does_not_complete_dated_meeting_proposal(tmp_path):
    root = fixture(tmp_path / "d", chats(), [("other", "q", "The preparation is done, so we can meet tomorrow"), ("other", "ME", "Okay")])
    _, out = run_pipeline(root)
    loop = next(l for l in out["open_loops"] if l["source_msg"] == "m0")
    assert loop["owner"] == "ME" and loop["status"] == "open"


def test_undated_possibility_does_not_become_meeting_obligation(tmp_path):
    root = fixture(tmp_path / "d", chats(), [("other", "q", "We can meet sometime")])
    _, out = run_pipeline(root)
    assert not out["open_loops"]


@pytest.mark.parametrize("reason", ["late", "unrelated_quote"])
def test_information_answer_is_bounded_to_its_thread_and_time(tmp_path, reason):
    reply = "Here is the campaign update: the audience is growing steadily and conversion quality is improving across the tested locations."
    root = fixture(tmp_path / "d", chats(), [("other", "q", "Any initial feedback on the campaign"),
                   ("other", "ME", "Reverting tomorrow on this."), ("other", "ME", reply)])
    path = root / "messages.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    if reason == "late":
        rows[2]["ts"] += 3 * 86400
    else:
        rows[2]["quoted_id"] = "unrelated"
        rows.append(dict(id="unrelated", chat_id="other", sender_id="q", sender_name="q",
                         ts=rows[0]["ts"] - 60, kind="text", quoted_id=None,
                         body="Please check the warehouse inventory"))
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf8")
    _, out = run_pipeline(root)
    assert next(l for l in out["open_loops"] if l["source_msg"] == "m1")["status"] == "open"


def test_business_name_house_does_not_override_explicit_company_brand(tmp_path):
    rows = [dict(chat_id="g", name="Elm Shirt House: Acme - Delivery", is_group=True, participants=["ME", "p", "q"]),
            dict(chat_id="home", name="Household Cook", is_group=False, participants=["ME", "staff"])]
    root = fixture(tmp_path / "d", rows, [("g", "p", "Hello"), ("home", "staff", "Hello")],
                   {"circles": {"p": "client", "q": "colleague", "staff": "staff"}, "organisations": ["Acme"], "goals": []})
    _, out = run_pipeline(root)
    company = next(o["id"] for o in out["organisations"] if o["name"] == "Acme")
    assert next(c for c in out["chats"] if c["chat_id"] == "g")["organisation"] == company
