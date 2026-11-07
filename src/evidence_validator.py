"""Deterministic Evidence & Referential Integrity Validator."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Dict, List, Set

from .conversation_index import ConversationIndex
from .models import (
    ORG_TYPE_ALIASES,
    USER_ID,
    VALID_CIRCLES,
    VALID_GOAL_AREAS,
    VALID_GOAL_HORIZONS,
    VALID_GOAL_TYPES,
    VALID_INTERVENTION_ACTIONS,
    VALID_LEAVE_REASONS,
    VALID_LOOP_STATUSES,
    VALID_LOOP_TYPES,
    VALID_ORG_RELATIONS,
    VALID_ORG_ROLES,
    VALID_ORG_SIDES,
    VALID_ORG_TYPES,
    ChatPrediction,
    ColdStartOutput,
    GoalPrediction,
    InterventionPrediction,
    OpenLoopPrediction,
    PersonPrediction,
)

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_and_repair(
    output: ColdStartOutput, index: ConversationIndex
) -> List[str]:
    """Deterministically enforce all schema, referential, and temporal invariants."""
    repairs: List[str] = []
    valid_msg_ids: Set[str] = set(index.messages_by_id.keys())
    valid_chat_ids: Set[str] = set(index.chats.keys())
    chat_participant_pids: Set[str] = {
        p for c in index.chats.values() for p in c.participant_ids if p and p != USER_ID
    }
    # The official check.py derives IDs strictly from chats.jsonl participants. Never surface
    # departed historical senders or seed-only IDs, even when there are no participant IDs.
    all_expected_pids: Set[str] = chat_participant_pids

    # 1. Validate People: exact 1 circle per non-ME chat participant, no missing people, valid org/person links
    seen_pids: Dict[str, PersonPrediction] = {}
    for p in output.people:
        if (
            not p.id
            or p.id == USER_ID
            or p.id in seen_pids
            or (chat_participant_pids and p.id not in chat_participant_pids)
        ):
            repairs.append(f"Dropped duplicate/non-participant person record '{p.id}'")
            continue
        if p.circle not in VALID_CIRCLES:
            repairs.append(f"Repaired invalid circle '{p.circle}' -> 'friend' for {p.id}")
            p.circle = "friend"
        p.evidence = [m for m in p.evidence if m in valid_msg_ids]
        seen_pids[p.id] = p

    for missing_pid in sorted(all_expected_pids - set(seen_pids.keys())):
        fallback_circle = (
            index.bundle.seed.people[missing_pid].circle
            if missing_pid in index.bundle.seed.people
            else "friend"
        )
        seen_pids[missing_pid] = PersonPrediction(
            id=missing_pid,
            name=index.best_name(missing_pid),
            circle=fallback_circle,
            source="repair",
        )
        repairs.append(f"Added missing person '{missing_pid}' with fallback circle '{fallback_circle}'")

    output.people = [seen_pids[pid] for pid in sorted(seen_pids.keys())]
    valid_person_ids = set(seen_pids.keys())

    # 2. Validate Organisations
    valid_org_ids: Set[str] = set()
    cleaned_orgs = []
    for org in output.organisations:
        if not org.id or org.id in valid_org_ids:
            continue
        org.type = ORG_TYPE_ALIASES.get(org.type, org.type)
        if org.type not in VALID_ORG_TYPES:
            org.type = "other"
            repairs.append(f"Repaired org {org.id} type -> 'other'")
        if org.side not in VALID_ORG_SIDES:
            org.side = "work" if org.type == "company" else "home"
            repairs.append(f"Repaired org {org.id} side -> '{org.side}'")
        if org.role not in VALID_ORG_ROLES:
            org.role = "owner"
            repairs.append(f"Repaired org {org.id} role -> 'owner'")
        org.people = sorted(p for p in set(org.people) if p in valid_person_ids)
        org.chats = sorted(c for c in set(org.chats) if c in valid_chat_ids)
        valid_org_ids.add(org.id)
        cleaned_orgs.append(org)
    output.organisations = cleaned_orgs

    # WhatsApp "Notes to self" chats often have participants == ["ME"] and still must be tied once.
    self_chat_ids = {
        cid
        for cid, chat in index.chats.items()
        if not any(pid and pid != USER_ID for pid in chat.participant_ids)
    }
    self_chat_org_id: str | None = None
    if self_chat_ids:
        residence = next((org for org in output.organisations if org.type == "residence"), None)
        existing_personal = next(
            (
                org for org in output.organisations
                if org.type == "other" and org.side == "home" and org.name.strip().lower() == "personal"
            ),
            None,
        )
        if residence is not None:
            self_chat_org_id = residence.id
        elif existing_personal is not None:
            self_chat_org_id = existing_personal.id
        else:
            used_org_ids = {org.id for org in output.organisations}
            self_chat_org_id = "personal"
            suffix = 2
            while self_chat_org_id in used_org_ids:
                self_chat_org_id = f"personal_{suffix}"
                suffix += 1
            output.organisations.append(
                OrganisationPrediction(
                    id=self_chat_org_id,
                    name="Personal",
                    type="other",
                    side="home",
                    role="owner",
                    source="repair",
                )
            )
            valid_org_ids.add(self_chat_org_id)
            repairs.append("Added Personal organisation for self-only chat(s)")

    # Clean person links after valid_org_ids & valid_person_ids are finalized
    for p in output.people:
        p.organisation_links = [
            lk
            for lk in p.organisation_links
            if lk.organisation_id in valid_org_ids and lk.relation in VALID_ORG_RELATIONS
        ]
        if p.circle == "staff":
            p.supports_people = sorted(
                sp for sp in set(p.supports_people) if sp in valid_person_ids and sp != p.id
            )
        else:
            p.supports_people = []

    # 3. Validate Chats: every chat in chats.jsonl tied to 1 valid person or organisation
    seen_chats: Dict[str, ChatPrediction] = {}
    fallback_owner_type = "organisation" if valid_org_ids else "person"
    fallback_owner_id = (
        sorted(valid_org_ids)[0]
        if valid_org_ids
        else (sorted(valid_person_ids)[0] if valid_person_ids else USER_ID)
    )

    for cp in output.chats:
        if cp.chat_id not in valid_chat_ids or cp.chat_id in seen_chats:
            continue
        if cp.owner_type == "person" and cp.owner_id not in valid_person_ids:
            cp.owner_type = fallback_owner_type
            cp.owner_id = fallback_owner_id
            repairs.append(f"Repaired invalid person owner on chat {cp.chat_id}")
        elif cp.owner_type == "organisation" and cp.owner_id not in valid_org_ids:
            cp.owner_type = fallback_owner_type
            cp.owner_id = fallback_owner_id
            repairs.append(f"Repaired invalid org owner on chat {cp.chat_id}")
        elif cp.owner_type not in ("person", "organisation"):
            cp.owner_type = fallback_owner_type
            cp.owner_id = fallback_owner_id
        seen_chats[cp.chat_id] = cp

    for missing_cid in sorted(valid_chat_ids - set(seen_chats.keys())):
        chat = index.chats[missing_cid]
        non_me = [p for p in chat.participant_ids if p != USER_ID and p in valid_person_ids]
        if missing_cid in self_chat_ids and self_chat_org_id:
            seen_chats[missing_cid] = ChatPrediction(
                chat_id=missing_cid,
                owner_type="organisation",
                owner_id=self_chat_org_id,
                source="repair",
            )
        elif non_me:
            seen_chats[missing_cid] = ChatPrediction(
                chat_id=missing_cid, owner_type="person", owner_id=non_me[0], source="repair"
            )
        else:
            seen_chats[missing_cid] = ChatPrediction(
                chat_id=missing_cid,
                owner_type=fallback_owner_type,
                owner_id=fallback_owner_id,
                source="repair",
            )
        repairs.append(f"Added missing chat assignment for '{missing_cid}'")

    output.chats = [seen_chats[cid] for cid in sorted(seen_chats.keys())]

    # Sync org.chats and org.people
    for org in output.organisations:
        org.chats = sorted(
            c.chat_id
            for c in output.chats
            if c.owner_type == "organisation" and c.owner_id == org.id
        )
        org.people = sorted(
            p.id
            for p in output.people
            if any(lk.organisation_id == org.id for lk in p.organisation_links)
        )

    # 4. Validate Goals: max 10, non-empty valid message_ids, valid enums
    valid_goals: List[GoalPrediction] = []
    for g in output.goals:
        g.message_ids = [m for m in dict.fromkeys(g.message_ids) if m in valid_msg_ids]
        if not g.message_ids or not g.title.strip():
            repairs.append(f"Dropped unsupported goal '{g.id}' with 0 valid message_ids")
            continue
        if g.area not in VALID_GOAL_AREAS:
            g.area = "self"
        if g.type not in VALID_GOAL_TYPES:
            g.type = "milestone"
        if g.horizon not in VALID_GOAL_HORIZONS:
            g.horizon = "3mo"
        valid_goals.append(g)
        if len(valid_goals) >= 10:
            break
    output.goals = valid_goals

    # 5. Validate Open Loops & Interventions: provenance, temporal monotonicity, enum coherence
    valid_owners = valid_person_ids | {USER_ID}
    cleaned_loops: List[OpenLoopPrediction] = []
    seen_loop_ids: Set[str] = set()

    for idx, lp in enumerate(output.open_loops):
        if lp.message_id not in valid_msg_ids:
            repairs.append(f"Dropped loop '{lp.id}' with non-existent origin message '{lp.message_id}'")
            continue
        origin_msg = index.messages_by_id[lp.message_id]
        if lp.chat_id not in valid_chat_ids:
            lp.chat_id = origin_msg.chat_id
            repairs.append(f"Aligned loop '{lp.id}' chat_id to origin message chat '{lp.chat_id}'")

        if not lp.id or lp.id in seen_loop_ids:
            lp.id = f"loop_{idx + 1:03d}"
        seen_loop_ids.add(lp.id)

        if lp.type not in VALID_LOOP_TYPES:
            lp.type = "commitment" if lp.owner == USER_ID else "followup"
        if lp.owner not in valid_owners:
            lp.owner = USER_ID
            repairs.append(f"Repaired invalid owner on loop '{lp.id}' -> ME")
        if lp.due_date is not None and not ISO_DATE_RE.match(lp.due_date):
            lp.due_date = None
            repairs.append(f"Cleared malformed due_date on loop '{lp.id}'")
        if lp.status not in VALID_LOOP_STATUSES:
            lp.status = "open"

        # Temporal Monotonicity & Closure Evidence Guard
        if lp.status in ("done", "dropped"):
            if not lp.closure_message_id or lp.closure_message_id not in valid_msg_ids:
                lp.status = "open"
                lp.closure_message_id = None
                repairs.append(
                    f"Downgraded loop '{lp.id}' to 'open' because closure_message_id was missing/invalid"
                )
            else:
                cl_msg = index.messages_by_id[lp.closure_message_id]
                if cl_msg.ts < origin_msg.ts or cl_msg.id == origin_msg.id:
                    lp.status = "open"
                    lp.closure_message_id = None
                    repairs.append(
                        f"Downgraded loop '{lp.id}' to 'open' because closure preceded or equaled creation"
                    )
                else:
                    lp.intervention = None

        if lp.status == "open":
            lp.closure_message_id = None
            if lp.intervention is None or lp.intervention.action not in VALID_INTERVENTION_ACTIONS:
                default_reason = "someone_elses" if lp.owner != USER_ID else "annoying"
                lp.intervention = InterventionPrediction(
                    action="leave" if lp.owner != USER_ID else "remind",
                    remind_at=(
                        (index.anchor_ts + timedelta(days=1)).strftime("%Y-%m-%dT09:00:00")
                        if lp.owner == USER_ID
                        else None
                    ),
                    leave_reason=default_reason if lp.owner != USER_ID else None,
                    source="repair",
                )
                repairs.append(f"Synthesized missing intervention on open loop '{lp.id}'")
            else:
                iv = lp.intervention
                if iv.action == "assist":
                    if not iv.next_step or not iv.next_step.strip():
                        iv.next_step = f"Prepare options and next step for '{lp.title[:50]}'"
                    if not isinstance(iv.needs_approval, bool):
                        iv.needs_approval = True
                    iv.remind_at = None
                    iv.leave_reason = None
                elif iv.action == "remind":
                    if not iv.remind_at:
                        iv.remind_at = (index.anchor_ts + timedelta(days=1)).strftime(
                            "%Y-%m-%dT09:00:00"
                        )
                    iv.next_step = None
                    iv.needs_approval = None
                    iv.leave_reason = None
                elif iv.action == "leave":
                    if iv.leave_reason not in VALID_LEAVE_REASONS:
                        iv.leave_reason = (
                            "someone_elses" if lp.owner != USER_ID else "annoying"
                        )
                    iv.next_step = None
                    iv.needs_approval = None
                    iv.remind_at = None

        cleaned_loops.append(lp)

    output.open_loops = cleaned_loops
    output.telemetry.validation_repairs = repairs
    return repairs
