"""Schema Introspection Adapter & Validator (native Cold Start schema.json + custom schema support)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .models import ORG_TYPE_ALIASES, ColdStartOutput

try:
    import jsonschema  # type: ignore

    _HAS_JSONSCHEMA = True
except ImportError:  # pragma: no cover
    jsonschema = None
    _HAS_JSONSCHEMA = False


def load_schema(schema_path: Optional[str | Path] = None) -> Dict[str, Any]:
    if schema_path is None:
        candidate = Path(__file__).resolve().parent.parent / "schema.json"
    else:
        candidate = Path(schema_path)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _pick_alias(props: Set[str], candidates: Iterable[str], default: str) -> str:
    if not props:
        return default
    for c in candidates:
        if c in props:
            return c
    return default


def _item_schema(schema: Dict[str, Any], top_key: str) -> Dict[str, Any]:
    top = (schema.get("properties") or {}).get(top_key) or {}
    if top.get("type") == "array":
        items = top.get("items") or {}
        if isinstance(items, dict):
            return items
    return {}


def _item_props(schema: Dict[str, Any], top_key: str) -> Set[str]:
    return set((_item_schema(schema, top_key).get("properties") or {}).keys())


def _match_enum(val: Optional[str], prop_schema: Dict[str, Any]) -> Optional[str]:
    """Adapt minor enum spelling differences (e.g. 'home' -> 'residence', 'holiday_home' -> 'vacation home')."""
    if val is None or not isinstance(prop_schema, dict):
        return val
    allowed = prop_schema.get("enum")
    if not isinstance(allowed, list) or not allowed:
        return val
    str_allowed = [a for a in allowed if isinstance(a, str)]
    if not str_allowed or val in str_allowed:
        return val
    if val in ORG_TYPE_ALIASES and ORG_TYPE_ALIASES[val] in str_allowed:
        return ORG_TYPE_ALIASES[val]
    norm_v = re.sub(r"[^a-z0-9]", "", val.lower())
    for cand in str_allowed:
        norm_c = re.sub(r"[^a-z0-9]", "", cand.lower())
        if norm_c == norm_v or norm_c in norm_v or norm_v in norm_c:
            return cand
    return str_allowed[0]


def serialize_output_for_schema(
    output: ColdStartOutput, schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Serialize ColdStartOutput into strict Cold Start schema.json (or custom schema if provided)."""
    schema = schema or {}
    top_props = set((schema.get("properties") or {}).keys())
    orgs_top_key = _pick_alias(top_props, ("organisations", "organizations"), "organisations")

    # 1. People
    p_item_sc = _item_schema(schema, "people")
    p_props = set((p_item_sc.get("properties") or {}).keys())
    p_circle_sc = (p_item_sc.get("properties") or {}).get("circle") or {}
    p_id_key = _pick_alias(p_props, ("id", "person_id"), "id")
    p_links_key = _pick_alias(
        p_props, ("links", "organisation_links", "org_links", "organisations"), "links"
    )
    p_sup_key = _pick_alias(
        p_props, ("looks_after", "supports_people", "supported_people"), "looks_after"
    )
    link_item_props = set(
        (((p_item_sc.get("properties") or {}).get(p_links_key) or {}).get("items") or {})
        .get("properties", {})
        .keys()
    )
    lk_org_key = _pick_alias(
        link_item_props, ("organisation", "organisation_id", "org_id"), "organisation"
    )
    lk_rel_key = _pick_alias(link_item_props, ("link", "relation", "role"), "link")

    people_list: List[Dict[str, Any]] = []
    for p in output.people:
        rec: Dict[str, Any] = {
            p_id_key: p.id,
            "circle": _match_enum(p.circle, p_circle_sc),
        }
        if "name" in p_props:
            rec["name"] = p.name
        if not p_props or p_links_key in p_props:
            serialized_links = [
                {lk_org_key: lk.organisation_id, lk_rel_key: lk.relation}
                for lk in p.organisation_links
            ]
            if serialized_links or p_links_key in (p_item_sc.get("required") or []):
                rec[p_links_key] = serialized_links
        if not p_props or p_sup_key in p_props:
            sup_list = list(p.supports_people)
            if sup_list or p_sup_key in (p_item_sc.get("required") or []):
                rec[p_sup_key] = sup_list
        if "evidence" in p_props:
            rec["evidence"] = list(p.evidence)
        people_list.append(rec)

    # 2. Organisations
    o_item_sc = _item_schema(schema, orgs_top_key)
    o_props = set((o_item_sc.get("properties") or {}).keys())
    o_id_key = _pick_alias(o_props, ("id", "organisation_id", "org_id"), "id")
    orgs_list: List[Dict[str, Any]] = []
    for o in output.organisations:
        canon_type = ORG_TYPE_ALIASES.get(o.type, o.type)
        orec: Dict[str, Any] = {
            o_id_key: o.id,
            "name": o.name,
            "type": _match_enum(canon_type, (o_item_sc.get("properties") or {}).get("type") or {}),
            "side": _match_enum(o.side, (o_item_sc.get("properties") or {}).get("side") or {}),
            "role": _match_enum(o.role, (o_item_sc.get("properties") or {}).get("role") or {}),
        }
        if "people" in o_props:
            orec["people"] = list(o.people)
        if "chats" in o_props:
            orec["chats"] = list(o.chats)
        orgs_list.append(orec)

    # 3. Chats
    c_props = _item_props(schema, "chats")
    c_id_key = _pick_alias(c_props, ("chat_id", "id"), "chat_id")
    use_oneof_chat_tie = (not c_props) or ("person" in c_props and "organisation" in c_props)
    c_type_key = _pick_alias(c_props, ("owner_type", "target_type", "kind"), "owner_type")
    c_owner_key = _pick_alias(c_props, ("owner_id", "target_id", "owner"), "owner_id")
    chats_list: List[Dict[str, Any]] = []
    for c in output.chats:
        if use_oneof_chat_tie:
            if c.owner_type == "organisation":
                chats_list.append({c_id_key: c.chat_id, "organisation": c.owner_id})
            else:
                chats_list.append({c_id_key: c.chat_id, "person": c.owner_id})
        else:
            chats_list.append(
                {
                    c_id_key: c.chat_id,
                    c_type_key: c.owner_type,
                    c_owner_key: c.owner_id,
                }
            )

    # 4. Goals
    g_props = _item_props(schema, "goals")
    g_title_key = _pick_alias(g_props, ("title", "description", "goal"), "title")
    g_msgs_key = _pick_alias(
        g_props, ("evidence", "message_ids", "messages"), "evidence"
    )
    goals_list: List[Dict[str, Any]] = []
    for g in output.goals:
        grec: Dict[str, Any] = {}
        if "id" in g_props:
            grec["id"] = g.id
        elif "goal_id" in g_props:
            grec["goal_id"] = g.id
        grec[g_title_key] = g.title
        grec["area"] = g.area
        grec["type"] = g.type
        grec["horizon"] = g.horizon
        grec[g_msgs_key] = list(g.message_ids)
        goals_list.append(grec)

    # 5. Open Loops & Help/Interventions
    l_item_sc = _item_schema(schema, "open_loops")
    l_props = set((l_item_sc.get("properties") or {}).keys())
    l_owner_key = _pick_alias(l_props, ("owner", "owner_id", "owed_by"), "owner")
    l_chat_key = _pick_alias(l_props, ("chat_id", "origin_chat_id", "chat"), "chat_id")
    l_msg_key = _pick_alias(
        l_props,
        ("source_msg", "message_id", "origin_message_id", "source_message_id"),
        "source_msg",
    )
    l_due_key = _pick_alias(l_props, ("due", "due_date"), "due")
    l_close_key = _pick_alias(
        l_props,
        ("closed_by", "closure_message_id", "closed_by_message_id", "evidence_message_id"),
        "closed_by",
    )
    l_help_key = _pick_alias(l_props, ("help", "intervention"), "help")
    help_sc = ((l_item_sc.get("properties") or {}).get(l_help_key) or {})
    help_props = set((help_sc.get("properties") or {}).keys())
    needs_key = _pick_alias(help_props, ("needs_yes", "needs_approval"), "needs_yes")
    reason_key = _pick_alias(help_props, ("reason", "leave_reason"), "reason")

    loops_list: List[Dict[str, Any]] = []
    separate_interventions: List[Dict[str, Any]] = []

    for lp in output.open_loops:
        iv_dict: Optional[Dict[str, Any]] = None
        if lp.status == "open" and lp.intervention is not None:
            iv = lp.intervention
            iv_dict = {"action": iv.action}
            if iv.action == "assist":
                if iv.next_step:
                    iv_dict["next_step"] = iv.next_step
                if iv.needs_approval is not None:
                    iv_dict[needs_key] = bool(iv.needs_approval)
            elif iv.action == "remind":
                if iv.remind_at:
                    iv_dict["remind_at"] = iv.remind_at
            elif iv.action == "leave":
                if iv.leave_reason:
                    iv_dict[reason_key] = str(iv.leave_reason)

        lrec: Dict[str, Any] = {}
        if "id" in l_props:
            lrec["id"] = lp.id
        elif "loop_id" in l_props:
            lrec["loop_id"] = lp.id
        lrec["title"] = lp.title
        lrec["type"] = lp.type
        lrec[l_owner_key] = lp.owner
        lrec[l_chat_key] = lp.chat_id
        lrec[l_msg_key] = lp.message_id
        if not l_props or l_due_key in l_props:
            lrec[l_due_key] = lp.due_date
        lrec["status"] = lp.status
        lrec[l_close_key] = lp.closure_message_id if lp.status in ("done", "dropped") else None
        if not l_props or l_help_key in l_props:
            lrec[l_help_key] = iv_dict if lp.status == "open" else None
        loops_list.append(lrec)

        if iv_dict is not None:
            separate_interventions.append({"loop_id": lp.id, **iv_dict})

    payload: Dict[str, Any] = {
        "people": people_list,
        orgs_top_key: orgs_list,
        "chats": chats_list,
        "goals": goals_list,
        "open_loops": loops_list,
    }
    if "interventions" in top_props:
        payload["interventions"] = separate_interventions

    return payload


def validate_json_against_schema(
    payload: Dict[str, Any], schema: Dict[str, Any]
) -> List[str]:
    """Validate serialized output JSON against schema.json; return list of error strings (empty if valid)."""
    if not schema:
        return []
    if _HAS_JSONSCHEMA and jsonschema is not None:
        validator_cls = jsonschema.validators.validator_for(schema)
        validator = validator_cls(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
        return [
            f"{'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}"
            for err in errors
        ]

    # Pure-Python structural validator when `jsonschema` package is absent
    errs: List[str] = []
    if not isinstance(payload, dict):
        return ["<root>: output must be a JSON object"]
    for req in schema.get("required", []):
        if req not in payload:
            errs.append(f"<root>: '{req}' is a required property")
    if schema.get("additionalProperties") is False:
        allowed_top = set((schema.get("properties") or {}).keys())
        extra = set(payload.keys()) - allowed_top
        if extra:
            errs.append(f"<root>: unknown top-level keys {sorted(extra)}")
    return errs
