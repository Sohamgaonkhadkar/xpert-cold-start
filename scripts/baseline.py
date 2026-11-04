"""Naive Starter Baseline (Experiment 0) for Controlled Ablation Benchmarking."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.conversation_index import ConversationIndex
from src.data_loader import load_dataset
from src.models import USER_ID


def run_naive_baseline(input_folder: str | Path, output_path: str | Path) -> Dict[str, Any]:
    """Simple baseline with unconditioned group co-membership, English-only loop regex, naive 'ok'/past-due closure, and default 'remind'."""
    bundle = load_dataset(input_folder)
    index = ConversationIndex(bundle)

    # 1. People: copy seed, otherwise propagate any co-member's circle naively (classic group homophily trap!)
    people_out: List[Dict[str, Any]] = []
    for pid in index.all_person_ids():
        if pid in bundle.seed.people:
            circ = bundle.seed.people[pid].circle
        else:
            circ = "friend"
            for cid in index.person_chats.get(pid, set()):
                for other in index.chats[cid].participant_ids:
                    if other in bundle.seed.people:
                        circ = bundle.seed.people[other].circle
                        break
        people_out.append(
            {
                "id": pid,
                "name": index.best_name(pid),
                "circle": circ,
                "organisation_links": [],
                "supports_people": [],
                "evidence": [],
            }
        )

    # 2. Organisations: only seeded names + naive group name promotion (falls into client-org trap!)
    orgs_out: List[Dict[str, Any]] = []
    for idx, so in enumerate(bundle.seed.organisations):
        orgs_out.append(
            {
                "id": f"base_org_{idx + 1}",
                "name": so.name,
                "type": "company",
                "side": "work",
                "role": "owner",
                "people": [],
                "chats": [],
            }
        )

    # 3. Chats: 1:1 -> person, group -> first org if any
    chats_out: List[Dict[str, Any]] = []
    for cid, chat in sorted(index.chats.items()):
        non_me = [p for p in chat.participant_ids if p != USER_ID]
        if not chat.is_group and non_me:
            chats_out.append({"chat_id": cid, "owner_type": "person", "owner_id": non_me[0]})
        elif orgs_out:
            orgs_out[0]["chats"].append(cid)
            chats_out.append(
                {"chat_id": cid, "owner_type": "organisation", "owner_id": orgs_out[0]["id"]}
            )
        else:
            chats_out.append(
                {"chat_id": cid, "owner_type": "person", "owner_id": non_me[0] if non_me else "p1"}
            )

    # 4. Goals: naive regex on word "goal" (captures friend's marathon & misses implicit!)
    goals_out: List[Dict[str, Any]] = []
    for m in bundle.messages:
        if "goal" in m.body.lower() or "wish i" in m.body.lower():
            goals_out.append(
                {
                    "id": f"bg_{len(goals_out) + 1}",
                    "title": m.body[:70],
                    "area": "self",
                    "type": "milestone",
                    "horizon": "1yr",
                    "message_ids": [m.id],
                }
            )
            if len(goals_out) >= 10:
                break

    # 5. Open loops: English-only trigger + naive "ok"/"done" closure + default "remind"
    loops_out: List[Dict[str, Any]] = []
    for m in bundle.messages:
        low = m.body.lower()
        if any(k in low for k in ("please", "can you", "i'll", "deadline")):
            status = "open"
            closure_id = None
            for rid in index.replies_to.get(m.id, []):
                rm = index.messages_by_id[rid]
                if any(w in rm.body.lower() for w in ("ok", "done", "sent")):
                    status = "done"
                    closure_id = rm.id
                    break
            iv = (
                {
                    "action": "remind",
                    "next_step": None,
                    "needs_approval": None,
                    "remind_at": "2026-11-01T09:00:00",
                    "leave_reason": None,
                }
                if status == "open"
                else None
            )
            loops_out.append(
                {
                    "id": f"bl_{len(loops_out) + 1:03d}",
                    "title": m.body[:80],
                    "type": "commitment",
                    "owner": USER_ID,
                    "chat_id": m.chat_id,
                    "message_id": m.id,
                    "due_date": None,
                    "status": status,
                    "closure_message_id": closure_id,
                    "intervention": iv,
                }
            )

    payload = {
        "people": people_out,
        "organisations": orgs_out,
        "chats": chats_out,
        "goals": goals_out,
        "open_loops": loops_out,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
