"""Read-only final validation of serialized prediction references and semantics."""
from datetime import date, datetime

from .models import USER_ID


def validate_references(payload, index):
    errors = []
    people = {p for c in index.chats.values() for p in c.participant_ids} - {USER_ID}
    organisations = {o["id"] for o in payload.get("organisations", [])}
    # Custom schema adaptation is supported, but validate official semantics using
    # canonical serialization at the call site if aliases are requested.
    if "organisations" not in payload or any("id" not in p for p in payload.get("people", [])):
        return ["Reference validation requires the official output field names"]
    for section, key in (("people", "id"), ("organisations", "id"), ("chats", "chat_id")):
        ids = [item[key] for item in payload[section]]
        if len(ids) != len(set(ids)):
            errors.append(f"{section}: duplicate IDs")
    if {p["id"] for p in payload["people"]} != people:
        errors.append("people: must list each non-ME input participant exactly once")
    if {c["chat_id"] for c in payload["chats"]} != set(index.chats):
        errors.append("chats: must tie every input chat exactly once")
    if any(not org["id"].strip() or not org["name"].strip() for org in payload["organisations"]):
        errors.append("organisations: nonempty ID and name required")
    for person in payload["people"]:
        if any(lk["organisation"] not in organisations for lk in person.get("links", [])):
            errors.append("people: unknown organisation link")
        if set(person.get("looks_after", [])) - people:
            errors.append("people: unknown looks_after person")
    for chat in payload["chats"]:
        if ("person" in chat) == ("organisation" in chat):
            errors.append("chats: exactly one owner required")
        if "person" in chat and chat["person"] not in people:
            errors.append("chats: unknown person owner")
        if "organisation" in chat and chat["organisation"] not in organisations:
            errors.append("chats: unknown organisation owner")
    for goal in payload["goals"]:
        if not goal["title"].strip() or not goal["evidence"] or set(goal["evidence"]) - set(index.messages_by_id):
            errors.append("goals: empty title or invalid evidence")
    for loop in payload["open_loops"]:
        source = index.messages_by_id.get(loop["source_msg"])
        closure = index.messages_by_id.get(loop.get("closed_by"))
        if not source or source.chat_id != loop["chat_id"]:
            errors.append("open_loops: source must belong to the originating chat")
        if loop["owner"] not in people | {USER_ID}:
            errors.append("open_loops: unknown owner")
        if not loop["title"].strip():
            errors.append("open_loops: empty title")
        if loop.get("due"):
            try:
                date.fromisoformat(loop["due"])
            except ValueError:
                errors.append("open_loops: invalid due date")
        if loop["status"] in ("done", "dropped"):
            if not closure or not source or closure.id == source.id or closure.ts < source.ts:
                errors.append("open_loops: closure evidence must follow its source")
            if loop.get("help") is not None:
                errors.append("open_loops: closed loop cannot have help")
        elif loop.get("closed_by") is not None:
            errors.append("open_loops: open loop cannot have closure evidence")
        help_ = loop.get("help")
        if loop["status"] == "open":
            if not help_:
                errors.append("open_loops: open loop needs help decision")
            elif help_["action"] == "assist" and (not help_.get("next_step") or not isinstance(help_.get("needs_yes"), bool)):
                errors.append("help: assist needs a next_step and boolean needs_yes")
            elif help_["action"] == "leave" and help_.get("reason") not in ("someone_elses", "personal", "annoying"):
                errors.append("help: leave needs a supported reason")
            elif help_["action"] == "remind":
                try:
                    remind = datetime.fromisoformat(help_.get("remind_at", "").replace("Z", "+00:00"))
                    if remind.tzinfo:
                        from .data_loader import _extract_aware_or_naive
                        ref, _ = _extract_aware_or_naive(source.ts_raw)
                        if ref and ref.tzinfo:
                            remind = remind.astimezone(ref.tzinfo)
                    if remind.replace(tzinfo=None) < index.anchor_ts:
                        errors.append("help: reminder precedes dataset end")
                except ValueError:
                    errors.append("help: invalid reminder timestamp")
    return errors
