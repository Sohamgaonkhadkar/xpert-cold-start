"""Deterministic ingestion of chats.jsonl, messages.jsonl, and seed.json."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import (
    ORG_TYPE_ALIASES,
    USER_ID,
    VALID_CIRCLES,
    VALID_GOAL_AREAS,
    VALID_GOAL_HORIZONS,
    VALID_GOAL_TYPES,
    VALID_ORG_ROLES,
    VALID_ORG_SIDES,
    VALID_ORG_TYPES,
    ChatRecord,
    DatasetBundle,
    MessageRecord,
    SeedData,
    SeedGoal,
    SeedOrganisation,
    SeedPerson,
)

# Default timezone for Unix epoch timestamps in Indian WhatsApp exports (+05:30 IST)
DEFAULT_EPOCH_TZ = timezone(timedelta(hours=5, minutes=30))
DEFAULT_EPOCH_SUFFIX = "+05:30"


def _epoch_timezone() -> Tuple[timezone, str]:
    raw_off = os.environ.get("COLDSTART_TZ_OFFSET", DEFAULT_EPOCH_SUFFIX).strip()
    if raw_off in ("Z", "+00:00", "UTC"):
        return timezone.utc, "+00:00"
    if len(raw_off) == 6 and raw_off[0] in ("+", "-") and raw_off[3] == ":":
        try:
            sign = 1 if raw_off[0] == "+" else -1
            hh = int(raw_off[1:3])
            mm = int(raw_off[4:6])
            return timezone(sign * timedelta(hours=hh, minutes=mm)), raw_off
        except ValueError:
            pass
    return DEFAULT_EPOCH_TZ, DEFAULT_EPOCH_SUFFIX


def _extract_aware_or_naive(raw: Any) -> Tuple[Optional[datetime], str]:
    """Parse raw timestamp preserving tzinfo (if present) and raw offset suffix."""
    if isinstance(raw, (int, float)):
        val = float(raw)
        if val > 1e12:
            val /= 1000.0
        ep_tz, ep_sfx = _epoch_timezone()
        return datetime.fromtimestamp(val, tz=ep_tz), ep_sfx
    if not isinstance(raw, str):
        return None, ""
    s = raw.strip()
    if not s:
        return None, ""
    if s.isdigit():
        val = float(s)
        if val > 1e12:
            val /= 1000.0
        ep_tz, ep_sfx = _epoch_timezone()
        return datetime.fromtimestamp(val, tz=ep_tz), ep_sfx
    orig_suffix = ""
    if s.endswith("Z"):
        orig_suffix = "Z"
        s = s[:-1] + "+00:00"
    elif len(s) >= 6 and s[-6] in ("+", "-") and s[-3] == ":":
        orig_suffix = s[-6:]
    try:
        return datetime.fromisoformat(s), orig_suffix
    except ValueError:
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y, %H:%M",
        ):
            try:
                return datetime.strptime(s, fmt), ""
            except ValueError:
                continue
    return None, ""


def parse_timestamp(raw: Any, target_tz: Optional[timezone] = None) -> datetime:
    """Parse ISO-8601 strings or epoch numbers into naive local datetime in target_tz (preserving local calendar date)."""
    dt, _ = _extract_aware_or_naive(raw)
    if dt is None:
        return datetime(1970, 1, 1)
    if dt.tzinfo is not None:
        if target_tz is not None:
            dt = dt.astimezone(target_tz).replace(tzinfo=None)
        else:
            # Keep the timestamp's own local wall-clock calendar date/time rather than shifting IST -> UTC!
            dt = dt.replace(tzinfo=None)
    return dt


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                yield idx, obj


def _parse_chats(chats_path: Path) -> Dict[str, ChatRecord]:
    chats: Dict[str, ChatRecord] = {}
    for _, obj in _iter_jsonl(chats_path):
        chat_id = str(obj.get("chat_id") or obj.get("id") or "").strip()
        if not chat_id:
            continue
        name = str(obj.get("name") or obj.get("title") or chat_id).strip()
        raw_participants = (
            obj.get("participant_ids")
            if "participant_ids" in obj
            else obj.get("participants", obj.get("members", []))
        )
        participants: List[str] = []
        if isinstance(raw_participants, list):
            for p in raw_participants:
                if isinstance(p, dict):
                    pid = str(p.get("id") or p.get("person_id") or "").strip()
                else:
                    pid = str(p).strip()
                if pid and pid not in participants:
                    participants.append(pid)
        is_group = bool(
            obj.get("is_group")
            if "is_group" in obj
            else (len([p for p in participants if p != USER_ID]) > 1)
        )
        # Preserve canonical sorted order for set invariance while keeping ME first if present
        ordered_participants = tuple(
            sorted(participants, key=lambda x: (0 if x == USER_ID else 1, x))
        )
        chats[chat_id] = ChatRecord(
            chat_id=chat_id,
            name=name,
            is_group=is_group,
            participant_ids=ordered_participants,
        )
    return dict(sorted(chats.items(), key=lambda kv: kv[0]))


def _parse_messages(messages_path: Path) -> Tuple[List[MessageRecord], str]:
    raw_rows = list(_iter_jsonl(messages_path))
    # First pass: detect dominant local timezone offset (e.g., +05:30 IST) so midnight messages keep local dates
    offset_counts: Counter[timedelta] = Counter()
    suffix_by_offset: Dict[timedelta, str] = {}
    for _, obj in raw_rows[:500]:
        ts_val = obj.get("ts") if "ts" in obj else obj.get("timestamp", "")
        dt, sfx = _extract_aware_or_naive(ts_val)
        if dt is not None and dt.tzinfo is not None:
            off = dt.utcoffset()
            if off is not None:
                offset_counts[off] += 1
                if sfx and off not in suffix_by_offset:
                    suffix_by_offset[off] = sfx

    target_tz: Optional[timezone] = None
    tz_suffix = ""
    if offset_counts:
        # Prefer dominant non-UTC local offset if any exists; otherwise UTC
        non_utc = [(off, cnt) for off, cnt in offset_counts.items() if off != timedelta(0)]
        chosen_off = (
            max(non_utc, key=lambda x: x[1])[0]
            if non_utc
            else offset_counts.most_common(1)[0][0]
        )
        target_tz = timezone(chosen_off)
        tz_suffix = suffix_by_offset.get(chosen_off, "")

    records: List[MessageRecord] = []
    seen_ids: set[str] = set()
    for idx, obj in raw_rows:
        msg_id = str(obj.get("id") or obj.get("message_id") or obj.get("msg_id") or f"msg_{idx}").strip()
        if msg_id in seen_ids:
            continue
        seen_ids.add(msg_id)
        chat_id = str(obj.get("chat_id") or "").strip()
        sender_id = str(obj.get("sender_id") or obj.get("from") or "").strip()
        if not sender_id and obj.get("from_me") is True:
            sender_id = USER_ID
        sender_name = str(
            obj.get("sender_name") or obj.get("from_name") or sender_id
        ).strip()
        ts_val = obj.get("ts") if "ts" in obj else obj.get("timestamp", "")
        ts_raw = str(ts_val) if ts_val is not None else ""
        ts = parse_timestamp(ts_val, target_tz=target_tz)
        kind = str(obj.get("kind") or obj.get("type") or "text").strip().lower()
        if kind not in ("text", "voice_transcript", "image_caption", "document"):
            kind = "text"
        body_val = (
            obj.get("body")
            if "body" in obj and obj.get("body") is not None
            else (
                obj.get("text")
                or obj.get("content")
                or obj.get("transcript")
                or obj.get("caption")
                or obj.get("filename")
                or obj.get("file_name")
                or ""
            )
        )
        body = str(body_val or "").strip()
        quoted_raw = (
            obj.get("quoted_id")
            if "quoted_id" in obj
            else obj.get("quoted_message_id", obj.get("reply_to_id", obj.get("reply_to")))
        )
        quoted_id = str(quoted_raw).strip() if quoted_raw else None
        if quoted_id in ("", "None", "null"):
            quoted_id = None
        records.append(
            MessageRecord(
                id=msg_id,
                chat_id=chat_id,
                sender_id=sender_id,
                sender_name=sender_name,
                ts_raw=ts_raw,
                ts=ts,
                kind=kind,
                body=body,
                quoted_id=quoted_id,
                line_index=idx,
            )
        )
    # Strictly order-invariant chronological sort: (timestamp, message_id)
    records.sort(key=lambda m: (m.ts, m.id))
    return records, tz_suffix


def _parse_seed(seed_path: Path) -> SeedData:
    if not seed_path.exists():
        return SeedData()
    with seed_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        return SeedData()

    people: Dict[str, SeedPerson] = {}
    raw_people = raw.get("people", raw.get("circles", {}))
    if isinstance(raw_people, dict):
        for pid, val in raw_people.items():
            pid_s = str(pid).strip()
            if not pid_s or pid_s == USER_ID:
                continue
            if isinstance(val, str) and val in VALID_CIRCLES:
                people[pid_s] = SeedPerson(id=pid_s, circle=val)
            elif isinstance(val, dict):
                circ = str(val.get("circle") or "").strip().lower()
                if circ in VALID_CIRCLES:
                    people[pid_s] = SeedPerson(
                        id=pid_s,
                        circle=circ,
                        name=val.get("name"),
                        supports_people=list(val.get("supports_people", [])),
                    )
    elif isinstance(raw_people, list):
        for item in raw_people:
            if not isinstance(item, dict):
                continue
            pid_s = str(item.get("id") or item.get("person_id") or "").strip()
            circ = str(item.get("circle") or "").strip().lower()
            if pid_s and pid_s != USER_ID and circ in VALID_CIRCLES:
                people[pid_s] = SeedPerson(
                    id=pid_s,
                    circle=circ,
                    name=item.get("name"),
                    supports_people=list(item.get("supports_people", [])),
                )

    organisations: List[SeedOrganisation] = []
    raw_orgs = raw.get("organisations", raw.get("organizations", []))
    if isinstance(raw_orgs, list):
        for item in raw_orgs:
            if isinstance(item, str) and item.strip():
                organisations.append(SeedOrganisation(name=item.strip()))
            elif isinstance(item, dict):
                nm = str(item.get("name") or "").strip()
                if nm:
                    raw_otype = str(item.get("type") or "").strip().lower()
                    otype = ORG_TYPE_ALIASES.get(raw_otype, raw_otype)
                    oside = item.get("side")
                    orole = item.get("role")
                    organisations.append(
                        SeedOrganisation(
                            name=nm,
                            id=item.get("id"),
                            type=otype if otype in VALID_ORG_TYPES else None,
                            side=oside if oside in VALID_ORG_SIDES else None,
                            role=orole if orole in VALID_ORG_ROLES else None,
                        )
                    )

    goals: List[SeedGoal] = []
    raw_goals = raw.get("goals")
    if raw_goals is None and "goal" in raw and raw["goal"] is not None:
        raw_goals = [raw["goal"]]
    if isinstance(raw_goals, list):
        for item in raw_goals:
            if isinstance(item, str) and item.strip():
                goals.append(SeedGoal(title=item.strip()))
            elif isinstance(item, dict):
                title = str(
                    item.get("title") or item.get("description") or item.get("goal") or ""
                ).strip()
                if title:
                    area = item.get("area")
                    gtype = item.get("type")
                    horizon = item.get("horizon")
                    mids = [
                        str(m)
                        for m in item.get("message_ids", item.get("evidence", []))
                        if m
                    ]
                    goals.append(
                        SeedGoal(
                            title=title,
                            id=item.get("id"),
                            area=area if area in VALID_GOAL_AREAS else None,
                            type=gtype if gtype in VALID_GOAL_TYPES else None,
                            horizon=horizon if horizon in VALID_GOAL_HORIZONS else None,
                            message_ids=mids,
                        )
                    )

    return SeedData(
        people=people,
        organisations=organisations,
        goals=goals,
        raw=raw,
    )


def load_dataset(input_folder: str | Path) -> DatasetBundle:
    """Load and deterministically normalize an Xpert Cold Start input directory."""
    folder = Path(input_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    chats = _parse_chats(folder / "chats.jsonl")
    messages, tz_suffix = _parse_messages(folder / "messages.jsonl")
    seed = _parse_seed(folder / "seed.json")

    # Synthesize fallback ChatRecord if messages reference a chat missing from chats.jsonl
    participants_by_chat: Dict[str, set[str]] = {}
    for m in messages:
        if m.chat_id:
            participants_by_chat.setdefault(m.chat_id, set()).add(m.sender_id)
    for cid, parts in sorted(participants_by_chat.items()):
        if cid not in chats:
            parts_with_me = set(parts) | {USER_ID}
            non_me = [p for p in parts_with_me if p != USER_ID]
            chats[cid] = ChatRecord(
                chat_id=cid,
                name=cid,
                is_group=len(non_me) > 1,
                participant_ids=tuple(
                    sorted(parts_with_me, key=lambda x: (0 if x == USER_ID else 1, x))
                ),
            )

    messages_by_id = {m.id: m for m in messages}
    if messages:
        anchor_ts = max(m.ts for m in messages)
        min_ts = min(m.ts for m in messages)
    else:
        anchor_ts = datetime(2026, 1, 1)
        min_ts = datetime(2026, 1, 1)

    return DatasetBundle(
        chats=chats,
        messages=messages,
        messages_by_id=messages_by_id,
        seed=seed,
        anchor_ts=anchor_ts,
        min_ts=min_ts,
        tz_suffix=tz_suffix,
    )
