"""Live-Interview Correction Engine: structured & natural-language overrides with isolated downstream cascade."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .conversation_index import ConversationIndex
from .intervention_policy import apply_intervention_policy
from .models import (
    USER_ID,
    VALID_CIRCLES,
    VALID_INTERVENTION_ACTIONS,
    VALID_LEAVE_REASONS,
    VALID_LOOP_STATUSES,
    VALID_ORG_RELATIONS,
    VALID_ORG_ROLES,
    VALID_ORG_SIDES,
    VALID_ORG_TYPES,
    ColdStartOutput,
    InterventionPrediction,
    OrgLink,
    OrganisationPrediction,
)


def load_corrections_file(path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corrections file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def parse_natural_language_correction(
    statement: str, output: ColdStartOutput
) -> Dict[str, Any]:
    """Convert a live-interview natural-language correction (e.g. 'Rohan is actually my staff') into a structured patch."""
    s = statement.strip()
    low = s.lower()
    patch: Dict[str, Any] = {}

    # Resolve person by ID, full display name, or base name before "- OrgTag" / "@domain"
    matched_pid: Optional[str] = None
    matches = []
    for p in output.people:
        base_name = re.split(r"\s*[-–—|(@/]", p.name or "")[0].strip().lower()
        if (
            p.id.lower() in low.split()
            or (p.name and len(p.name) >= 3 and p.name.lower() in low)
            or (len(base_name) >= 3 and re.search(rf"\b{re.escape(base_name)}\b", low))
        ):
            # Prefer the longest whole name: 'Sneha Iyer' outranks 'Sneha'.
            specificity = max((len(name) for name in (p.name.lower(), base_name)
                               if name and re.search(rf"\b{re.escape(name)}\b", low)), default=0)
            matches.append((specificity, p.id))
    exact_ids = [p.id for p in output.people if p.id.lower() in low.split()]
    if len(exact_ids) == 1:
        matched_pid = exact_ids[0]
    elif matches:
        best = max(score for score, _ in matches)
        best_ids = [pid for score, pid in matches if score == best]
        if len(best_ids) != 1:
            raise ValueError("Ambiguous person correction; use the exact person ID")
        matched_pid = best_ids[0]

    # Pattern 1: Circle correction ("<Name/ID> is [actually] [my] <circle>")
    if matched_pid:
        for circ in VALID_CIRCLES:
            if re.search(rf"\b{re.escape(circ)}\b", low):
                patch["people"] = {matched_pid: {"circle": circ}}
                return patch

    # Pattern 2: Loop status/owner correction ("loop_003 is done via msg_12" / "msg_10 is open")
    m_loop = re.search(r"\b(loop_\d+|msg_[a-z0-9_]+|m\d+)\b", low)
    if m_loop:
        target_ref = m_loop.group(1)
        loop_patch: Dict[str, Any] = {}
        for st in VALID_LOOP_STATUSES:
            if re.search(rf"\b{st}\b", low):
                loop_patch["status"] = st
                break
        m_ev = re.search(r"\b(?:via|by|in)\s+([a-z0-9_-]+)\b", low)
        if m_ev:
            loop_patch["closure_message_id"] = m_ev.group(1)
        if "owned by me" in low or "my task" in low:
            loop_patch["owner"] = USER_ID
        elif matched_pid and ("owned by" in low or "belongs to" in low):
            loop_patch["owner"] = matched_pid
        for act in VALID_INTERVENTION_ACTIONS:
            if re.search(rf"\b{act}\b", low):
                loop_patch["intervention_action"] = act
        if loop_patch:
            patch["open_loops"] = {target_ref: loop_patch}
            return patch

    return patch


def apply_corrections(
    output: ColdStartOutput,
    corrections: Dict[str, Any],
    index: ConversationIndex,
    llm_Dummy_None: Any = None,
) -> List[str]:
    """Apply user-supplied corrections and deterministically cascade ONLY affected records."""
    if not corrections:
        return []

    applied_notes: List[str] = []
    people_map = {p.id: p for p in output.people}
    org_map = {o.id: o for o in output.organisations}
    chat_map = {c.chat_id: c for c in output.chats}

    # Support natural-language statements array as convenience
    for stmt in corrections.get("statements", []) or []:
        if isinstance(stmt, str) and stmt.strip():
            parsed = parse_natural_language_correction(stmt, output)
            for k, v in parsed.items():
                if isinstance(v, dict):
                    corrections.setdefault(k, {}).update(v)

    affected_person_ids: Set[str] = set()
    explicitly_corrected_loops: Set[str] = set()

    # 1. Organisation additions or edits
    raw_org_corr = corrections.get("organisations") or {}
    if isinstance(raw_org_corr, dict):
        for oid, spec in raw_org_corr.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("delete") is True and oid in org_map:
                del org_map[oid]
                output.organisations = [o for o in output.organisations if o.id != oid]
                applied_notes.append(f"Deleted organisation {oid}")
                continue
            if oid not in org_map:
                new_org = OrganisationPrediction(
                    id=oid,
                    name=str(spec.get("name") or oid),
                    type=spec.get("type") if spec.get("type") in VALID_ORG_TYPES else "company",
                    side=spec.get("side") if spec.get("side") in VALID_ORG_SIDES else "work",
                    role=spec.get("role") if spec.get("role") in VALID_ORG_ROLES else "owner",
                    source="corrected",
                )
                output.organisations.append(new_org)
                org_map[oid] = new_org
                applied_notes.append(f"Added organisation {oid}")
            else:
                org = org_map[oid]
                if "name" in spec:
                    org.name = str(spec["name"])
                if spec.get("type") in VALID_ORG_TYPES:
                    org.type = spec["type"]
                if spec.get("side") in VALID_ORG_SIDES:
                    org.side = spec["side"]
                if spec.get("role") in VALID_ORG_ROLES:
                    org.role = spec["role"]
                org.source = "corrected"
                applied_notes.append(f"Updated organisation {oid}")

    # 2. People circle / org-link / supports_people corrections
    raw_people_corr = corrections.get("people") or {}
    if isinstance(raw_people_corr, dict):
        for pid, spec in raw_people_corr.items():
            if pid not in people_map:
                continue
            person = people_map[pid]
            if isinstance(spec, str) and spec in VALID_CIRCLES:
                spec = {"circle": spec}
            if not isinstance(spec, dict):
                continue

            if spec.get("circle") in VALID_CIRCLES:
                old_circle = person.circle
                person.circle = spec["circle"]
                person.source = "corrected"
                person.confidence = 1.0
                affected_person_ids.add(pid)
                applied_notes.append(f"Person {pid} circle: {old_circle} -> {person.circle}")
                output.ledger.record(
                    subject_type="person",
                    subject_id=pid,
                    field_name="circle",
                    predicted_value=person.circle,
                    strength="user_correction",
                    confidence=1.0,
                    rationale="Live user correction override",
                    stage="corrections",
                )

                # Cascade person's default org relation if `organisation_links` wasn't explicitly overridden
                if "organisation_links" not in spec:
                    if person.circle == "client":
                        # Clients NEVER run or support ME's organisations!
                        person.organisation_links = []
                    elif person.circle == "staff":
                        person.organisation_links = [
                            OrgLink(organisation_id=lk.organisation_id, relation="supports")
                            for lk in person.organisation_links
                        ]
                        if not person.organisation_links:
                            home_oid = next(
                                (o.id for o in output.organisations if o.type == "home"), None
                            )
                            if home_oid:
                                person.organisation_links = [
                                    OrgLink(organisation_id=home_oid, relation="supports")
                                ]
                    elif person.circle in ("colleague", "partner"):
                        person.organisation_links = [
                            OrgLink(organisation_id=lk.organisation_id, relation="runs")
                            for lk in person.organisation_links
                            if lk.organisation_id in org_map and org_map[lk.organisation_id].side == "work"
                        ]
                if person.circle != "staff" and "supports_people" not in spec:
                    person.supports_people = []

            if "organisation_links" in spec and isinstance(spec["organisation_links"], list):
                new_links: List[OrgLink] = []
                for lk in spec["organisation_links"]:
                    if isinstance(lk, dict):
                        loid = str(lk.get("organisation_id") or "")
                        lrel = str(lk.get("relation") or "")
                        if loid in org_map and lrel in VALID_ORG_RELATIONS:
                            new_links.append(OrgLink(organisation_id=loid, relation=lrel))
                person.organisation_links = new_links
                person.source = "corrected"
                applied_notes.append(f"Updated org links for person {pid}")

            if "supports_people" in spec and isinstance(spec["supports_people"], list):
                person.supports_people = [
                    str(sp) for sp in spec["supports_people"] if str(sp) in people_map
                ]
                person.source = "corrected"

    # 3. Chat ownership corrections
    raw_chat_corr = corrections.get("chats") or {}
    if isinstance(raw_chat_corr, dict):
        for cid, spec in raw_chat_corr.items():
            if cid not in chat_map or not isinstance(spec, dict):
                continue
            otype = str(spec.get("owner_type") or "")
            oid = str(spec.get("owner_id") or "")
            if (otype == "person" and oid in people_map) or (
                otype == "organisation" and oid in org_map
            ):
                chat_map[cid].owner_type = otype
                chat_map[cid].owner_id = oid
                chat_map[cid].source = "corrected"
                chat_map[cid].confidence = 1.0
                applied_notes.append(f"Chat {cid} owner -> {otype}:{oid}")
                output.ledger.record(
                    subject_type="chat",
                    subject_id=cid,
                    field_name="owner",
                    predicted_value=f"{otype}:{oid}",
                    strength="user_correction",
                    confidence=1.0,
                    rationale="Live user correction override",
                    stage="corrections",
                )

    # Re-sync organisation `.people` and `.chats` lists cleanly after people/chat edits
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

    # 4. Open-loop corrections (match by loop.id, origin message_id, or title substring)
    raw_loop_corr = corrections.get("open_loops") or {}
    if isinstance(raw_loop_corr, dict):
        for key, spec in raw_loop_corr.items():
            if not isinstance(spec, dict):
                continue
            exact = [lp for lp in output.open_loops if lp.id == key or lp.message_id == key]
            targets = exact or [lp for lp in output.open_loops if key.lower() in lp.title.lower()]
            if len(targets) > 1:
                raise ValueError("Ambiguous loop correction; use the exact loop or source message ID")
            if not targets:
                continue
            target_loop = targets[0]
            if spec.get("status") in ("done", "dropped"):
                closure_id = spec.get("closure_message_id") or target_loop.closure_message_id
                closure = index.messages_by_id.get(str(closure_id))
                origin = index.messages_by_id[target_loop.message_id]
                if not closure or closure.id == origin.id or closure.ts < origin.ts:
                    raise ValueError("Closing a corrected loop requires a valid later closure_message_id")

            explicitly_corrected_loops.add(target_loop.id)
            target_loop.source = "corrected"

            if "owner" in spec:
                new_owner = str(spec["owner"]).strip()
                if new_owner == USER_ID or new_owner in people_map:
                    target_loop.owner = new_owner
                    applied_notes.append(f"Loop {target_loop.id} owner -> {new_owner}")

            if "due_date" in spec:
                target_loop.due_date = spec["due_date"]

            if spec.get("status") in VALID_LOOP_STATUSES:
                target_loop.status = spec["status"]
                if target_loop.status == "open":
                    target_loop.closure_message_id = None
                else:
                    cmid = spec.get("closure_message_id")
                    if cmid and str(cmid) in index.messages_by_id:
                        target_loop.closure_message_id = str(cmid)
                    target_loop.intervention = None
                applied_notes.append(f"Loop {target_loop.id} status -> {target_loop.status}")

            # Explicit intervention override or re-derivation on open loop
            if target_loop.status == "open":
                act = spec.get("intervention_action") or (
                    spec.get("intervention", {}).get("action")
                    if isinstance(spec.get("intervention"), dict)
                    else None
                )
                if act in VALID_INTERVENTION_ACTIONS:
                    iv_spec = spec.get("intervention") if isinstance(spec.get("intervention"), dict) else spec
                    if act == "assist":
                        target_loop.intervention = InterventionPrediction(
                            action="assist",
                            next_step=str(iv_spec.get("next_step") or f"Prepare next step for '{target_loop.title[:50]}'"),
                            needs_approval=bool(iv_spec.get("needs_approval", True)),
                            confidence=1.0,
                            source="corrected",
                        )
                    elif act == "remind":
                        rem_at = str(
                            iv_spec.get("remind_at")
                            or (index.anchor_ts.strftime("%Y-%m-%dT09:00:00"))
                        )
                        target_loop.intervention = InterventionPrediction(
                            action="remind",
                            remind_at=rem_at,
                            confidence=1.0,
                            source="corrected",
                        )
                    else:
                        lreason = str(iv_spec.get("leave_reason") or "someone_elses")
                        if lreason not in VALID_LEAVE_REASONS:
                            lreason = "someone_elses"
                        target_loop.intervention = InterventionPrediction(
                            action="leave",
                            leave_reason=lreason,
                            confidence=1.0,
                            source="corrected",
                        )
                elif "owner" in spec:
                    # Owner changed on an open loop without explicit intervention override -> recompute ONLY this loop!
                    from .llm_client import LLMClient

                    offline_llm = LLMClient(api_key="", base_url="")
                    apply_intervention_policy(
                        [target_loop], index, people_map, offline_llm, output.ledger
                    )
                    if target_loop.intervention:
                        target_loop.intervention.source = "corrected_cascade"

            output.ledger.record(
                subject_type="open_loop",
                subject_id=target_loop.id,
                field_name="correction",
                predicted_value={"status": target_loop.status, "owner": target_loop.owner},
                strength="user_correction",
                confidence=1.0,
                message_ids=[target_loop.message_id],
                rationale="Live user correction override",
                stage="corrections",
            )

    # 5. Isolated Downstream Cascade: if a person's circle changed (e.g., vendor -> staff),
    # recompute intervention ONLY on open loops directly involving that person!
    if affected_person_ids:
        from .llm_client import LLMClient

        offline_llm = LLMClient(api_key="", base_url="")
        impacted_loops = []
        for lp in output.open_loops:
            if lp.status != "open" or lp.id in explicitly_corrected_loops:
                continue
            chat_parts = (
                set(index.chats[lp.chat_id].participant_ids)
                if lp.chat_id in index.chats
                else set()
            )
            if lp.owner in affected_person_ids or (
                not index.chats[lp.chat_id].is_group and (chat_parts & affected_person_ids)
            ):
                impacted_loops.append(lp)

        if impacted_loops:
            apply_intervention_policy(
                impacted_loops, index, people_map, offline_llm, output.ledger
            )
            for lp in impacted_loops:
                if lp.intervention:
                    lp.intervention.source = "corrected_cascade"
            applied_notes.append(
                f"Cascaded circle update to {len(impacted_loops)} affected open loop(s)"
            )

    return applied_notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or append live-interview corrections to a corrections JSON file."
    )
    parser.add_argument("--file", default="corrections.json", help="Path to corrections JSON")
    parser.add_argument("--person-circle", nargs=2, metavar=("PERSON_ID", "CIRCLE"))
    parser.add_argument("--chat-owner", nargs=3, metavar=("CHAT_ID", "OWNER_TYPE", "OWNER_ID"))
    parser.add_argument("--loop-owner", nargs=2, metavar=("LOOP_OR_MSG_ID", "OWNER_ID"))
    parser.add_argument("--loop-status", nargs=2, metavar=("LOOP_OR_MSG_ID", "STATUS"))
    parser.add_argument("--closure-msg", metavar="MSG_ID", default=None)
    parser.add_argument("--say", metavar="STATEMENT", help="Natural-language correction statement")
    args = parser.parse_args()

    path = Path(args.file)
    data: Dict[str, Any] = json.loads(path.read_text("utf-8")) if path.exists() else {}

    if args.person_circle:
        pid, circ = args.person_circle
        if circ not in VALID_CIRCLES:
            raise SystemExit(f"Invalid circle '{circ}'. Valid: {VALID_CIRCLES}")
        data.setdefault("people", {})[pid] = {"circle": circ}

    if args.chat_owner:
        cid, otype, oid = args.chat_owner
        if otype not in ("person", "organisation"):
            raise SystemExit("OWNER_TYPE must be 'person' or 'organisation'")
        data.setdefault("chats", {})[cid] = {"owner_type": otype, "owner_id": oid}

    if args.loop_owner:
        lid, oid = args.loop_owner
        data.setdefault("open_loops", {}).setdefault(lid, {})["owner"] = oid

    if args.loop_status:
        lid, st = args.loop_status
        if st not in VALID_LOOP_STATUSES:
            raise SystemExit(f"Invalid status '{st}'. Valid: {VALID_LOOP_STATUSES}")
        entry = data.setdefault("open_loops", {}).setdefault(lid, {})
        entry["status"] = st
        if args.closure_msg:
            entry["closure_message_id"] = args.closure_msg

    if args.say:
        data.setdefault("statements", []).append(args.say)

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved corrections to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
