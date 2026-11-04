"""Tests for live-interview user corrections (all 5 override types + cascade isolation) and run.sh CLI."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from scripts.generate_synthetic_suite import build_synthetic_benchmark
from src.conversation_index import ConversationIndex
from src.corrections import apply_corrections, load_corrections_file
from src.data_loader import load_dataset
from src.llm_client import LLMClient
from src.output_builder import run_pipeline


def test_all_five_live_correction_types_and_cascade_isolation(tmp_path: Path):
    """Verify circle, org_membership, chat_ownership, task_ownership, and task_status overrides + isolated cascade."""
    ddir = tmp_path / "corr_test"
    build_synthetic_benchmark(ddir)
    bundle = load_dataset(ddir)
    index = ConversationIndex(bundle)
    out, _ = run_pipeline(ddir)

    baseline_goals = copy.deepcopy(out.goals)
    unrelated_person = copy.deepcopy(next(p for p in out.people if p.id == "p_vendor_ca"))
    unrelated_loop = copy.deepcopy(next(l for l in out.open_loops if l.message_id == "m130"))

    hiring_loop = next(l for l in out.open_loops if l.message_id == "m117")
    proposal_loop = next(l for l in out.open_loops if l.message_id == "m101")
    zephyr_org = next(o for o in out.organisations if "zephyr" in o.name.lower())

    corrections_payload = {
        "statements": [
            # 1. Natural-language circle override
            "Sneha Iyer is actually my partner",
            # 5. Natural-language task_status override
            f"{proposal_loop.id} is done via m102",
        ],
        "people": {
            # 2. Structured org_membership override
            "p_friend_1": {
                "organisation_links": [
                    {"organisation_id": zephyr_org.id, "relation": "supports"}
                ]
            }
        },
        "chats": {
            # 3. Chat ownership override
            "c_grp_friends": {"owner_type": "organisation", "owner_id": zephyr_org.id}
        },
        "open_loops": {
            # 4. Task ownership override (delegating ME's hiring budget loop -> cascades to leave(someone_elses)!)
            hiring_loop.id: {"owner": "p_colleague_1"}
        },
    }

    applied = apply_corrections(out, corrections_payload, index)
    assert len(applied) >= 5

    # Check 1: Sneha updated to partner with source="corrected"
    sneha = next(p for p in out.people if p.id == "p_colleague_2")
    assert sneha.circle == "partner"
    assert sneha.source == "corrected"

    # Check 2: Org membership added cleanly and reflected in org.people
    vikram = next(p for p in out.people if p.id == "p_friend_1")
    assert any(
        lk.organisation_id == zephyr_org.id and lk.relation == "supports"
        for lk in vikram.organisation_links
    )
    assert "p_friend_1" in zephyr_org.people

    # Check 3: Chat ownership updated and reflected in org.chats
    c_friends = next(c for c in out.chats if c.chat_id == "c_grp_friends")
    assert c_friends.owner_type == "organisation"
    assert c_friends.owner_id == zephyr_org.id
    assert "c_grp_friends" in zephyr_org.chats

    # Check 4: Reassigning loop owner from ME -> colleague automatically cascaded intervention to leave(someone_elses)!
    updated_hiring = next(l for l in out.open_loops if l.id == hiring_loop.id)
    assert updated_hiring.owner == "p_colleague_1"
    assert updated_hiring.intervention is not None
    assert updated_hiring.intervention.action == "leave"
    assert updated_hiring.intervention.leave_reason == "someone_elses"
    assert updated_hiring.intervention.source == "corrected_cascade"

    # Check 5: Marking open loop done cleared intervention and set closure_message_id!
    updated_prop = next(l for l in out.open_loops if l.id == proposal_loop.id)
    assert updated_prop.status == "done"
    assert updated_prop.closure_message_id == "m102"
    assert updated_prop.intervention is None

    # Isolation guarantee: unrelated goals, people, and loops were NOT perturbed!
    assert out.goals == baseline_goals
    assert next(p for p in out.people if p.id == "p_vendor_ca") == unrelated_person
    assert next(l for l in out.open_loops if l.message_id == "m130") == unrelated_loop


def test_run_sh_cli_end_to_end_and_llm_client_caching(tmp_path: Path):
    """Verify `python -m src.corrections` + `bash run.sh <input_folder> <output_path>` CLI contract and LLMClient cache."""
    ddir = tmp_path / "cli_input"
    out_file = tmp_path / "predictions" / "out.json"
    corr_file = tmp_path / "corrections.json"
    build_synthetic_benchmark(ddir)

    repo_root = Path(__file__).resolve().parent.parent
    # Build corrections.json using the live-interview CLI helper (`python -m src.corrections`)
    c_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.corrections",
            "--file",
            str(corr_file),
            "--person-circle",
            "p_colleague_3",
            "partner",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert c_proc.returncode == 0, c_proc.stderr
    assert load_corrections_file(corr_file)["people"]["p_colleague_3"]["circle"] == "partner"

    proc = subprocess.run(
        ["bash", "run.sh", str(ddir), str(out_file), "--corrections", str(corr_file)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_file.exists()

    payload = json.loads(out_file.read_text("utf-8"))
    dev_person = next(p for p in payload["people"] if p["id"] == "p_colleague_3")
    assert dev_person["circle"] == "partner"

    # Also verify LLMClient deduplicates identical calls via prompt cache
    call_counter = {"n": 0}

    def responder(stage: str, sp: str, up: str):
        call_counter["n"] += 1
        return {"ok": True}

    client = LLMClient(model="gpt-4.1-mini", mock_responder=responder)
    r1 = client.complete_json("test_stage", "sys", "same user prompt")
    r2 = client.complete_json("test_stage", "sys", "same user prompt")
    assert r1 == {"ok": True} and r2 == {"ok": True}
    assert call_counter["n"] == 1
    assert client.usage_snapshot()["cache_hits"] == 1
