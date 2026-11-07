"""Local Official-Rubric-Aligned Checker & Scorer for Xpert Cold Start Challenge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conversation_index import tokenize
from src.models import ORG_TYPE_ALIASES
from src.schema_adapter import load_schema, validate_json_against_schema


def _normalize_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize both official Cold Start wire format (schema.json) and legacy internal dicts into canonical scoring dicts."""
    people_norm: List[Dict[str, Any]] = []
    org_people_from_links: Dict[str, Set[str]] = {}
    org_chats_from_ties: Dict[str, Set[str]] = {}

    for p in raw.get("people", []) or []:
        p_copy = dict(p)
        sup_p = p.get("looks_after") if "looks_after" in p else p.get("supports_people", [])
        p_copy["supports_people"] = list(sup_p or [])
        links = p.get("links") if "links" in p else p.get("organisation_links", [])
        for lk in links or []:
            oid = lk.get("organisation") or lk.get("organisation_id")
            if oid:
                org_people_from_links.setdefault(str(oid), set()).add(str(p["id"]))
        people_norm.append(p_copy)

    chats_norm: List[Dict[str, Any]] = []
    for c in raw.get("chats", []) or []:
        if "owner_type" in c and "owner_id" in c:
            otype, oid = c["owner_type"], c["owner_id"]
        elif "organisation" in c:
            otype, oid = "organisation", c["organisation"]
        elif "person" in c:
            otype, oid = "person", c["person"]
        else:
            otype, oid = "person", ""
        if otype == "organisation" and oid:
            org_chats_from_ties.setdefault(str(oid), set()).add(str(c["chat_id"]))
        chats_norm.append({"chat_id": c["chat_id"], "owner_type": otype, "owner_id": oid})

    orgs_norm: List[Dict[str, Any]] = []
    for o in raw.get("organisations", []) or []:
        o_copy = dict(o)
        oid = str(o["id"])
        raw_t = str(o.get("type") or "")
        o_copy["type"] = ORG_TYPE_ALIASES.get(raw_t, raw_t)
        o_copy["chats"] = sorted(set(o.get("chats", [])) | org_chats_from_ties.get(oid, set()))
        o_copy["people"] = sorted(set(o.get("people", [])) | org_people_from_links.get(oid, set()))
        orgs_norm.append(o_copy)

    goals_norm: List[Dict[str, Any]] = []
    for g in raw.get("goals", []) or []:
        g_copy = dict(g)
        mids = g.get("evidence") if "evidence" in g else g.get("message_ids", [])
        g_copy["message_ids"] = list(mids or [])
        goals_norm.append(g_copy)

    loops_norm: List[Dict[str, Any]] = []
    for lp in raw.get("open_loops", []) or []:
        l_copy = dict(lp)
        if "message_id" not in l_copy and "source_msg" in l_copy:
            l_copy["message_id"] = l_copy["source_msg"]
        if "due_date" not in l_copy and "due" in l_copy:
            l_copy["due_date"] = l_copy["due"]
        if "closure_message_id" not in l_copy and "closed_by" in l_copy:
            l_copy["closure_message_id"] = l_copy["closed_by"]
        raw_help = l_copy.get("help") if "help" in l_copy else l_copy.get("intervention")
        if isinstance(raw_help, str):
            l_copy["intervention"] = {"action": raw_help}
        elif isinstance(raw_help, dict):
            iv_copy = dict(raw_help)
            if "needs_approval" not in iv_copy and "needs_yes" in iv_copy:
                iv_copy["needs_approval"] = iv_copy["needs_yes"]
            if "leave_reason" not in iv_copy and "reason" in iv_copy:
                iv_copy["leave_reason"] = iv_copy["reason"]
            l_copy["intervention"] = iv_copy
        else:
            l_copy["intervention"] = None
        if "source_msgs" in l_copy and "acceptable_origin_ids" not in l_copy:
            l_copy["acceptable_origin_ids"] = list(l_copy["source_msgs"])
            if "message_id" not in l_copy and l_copy["source_msgs"]:
                l_copy["message_id"] = l_copy["source_msgs"][0]
        if "closing_msgs" in l_copy and "acceptable_closure_ids" not in l_copy:
            l_copy["acceptable_closure_ids"] = list(l_copy["closing_msgs"])
        loops_norm.append(l_copy)

    return {
        "people": people_norm,
        "organisations": orgs_norm,
        "chats": chats_norm,
        "goals": goals_norm,
        "open_loops": loops_norm,
    }


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def match_organisations(
    pred_orgs: List[Dict[str, Any]], ref_orgs: List[Dict[str, Any]]
) -> Tuple[Dict[str, str], float]:
    """Match predicted orgs to reference orgs by attached chats + people (NOT by name!)."""
    pairs: List[Tuple[float, str, str]] = []
    for po in pred_orgs:
        p_set = {f"c:{c}" for c in po.get("chats", [])} | {
            f"p:{p}" for p in po.get("people", [])
        }
        for ro in ref_orgs:
            r_set = {f"c:{c}" for c in ro.get("chats", [])} | {
                f"p:{p}" for p in ro.get("people", [])
            }
            sim = _jaccard(p_set, r_set)
            if po.get("side") == ro.get("side"):
                sim += 0.10
            if po.get("type") == ro.get("type"):
                sim += 0.10
            pairs.append((sim, po["id"], ro["id"]))

    pairs.sort(key=lambda x: -x[0])
    pred_to_ref: Dict[str, str] = {}
    used_ref: Set[str] = set()
    matched_jacc_sum = 0.0

    ref_by_id = {ro["id"]: ro for ro in ref_orgs}
    pred_by_id = {po["id"]: po for po in pred_orgs}

    for _, pid, rid in pairs:
        if pid in pred_to_ref or rid in used_ref:
            continue
        po = pred_by_id[pid]
        ro = ref_by_id[rid]
        p_set = {f"c:{c}" for c in po.get("chats", [])} | {
            f"p:{p}" for p in po.get("people", [])
        }
        r_set = {f"c:{c}" for c in ro.get("chats", [])} | {
            f"p:{p}" for p in ro.get("people", [])
        }
        jacc = _jaccard(p_set, r_set)
        if jacc >= 0.20 or (not r_set and po.get("type") == ro.get("type")):
            pred_to_ref[pid] = rid
            used_ref.add(rid)
            attr_bonus = (
                0.5 * float(po.get("side") == ro.get("side"))
                + 0.5 * float(po.get("role") == ro.get("role"))
            )
            matched_jacc_sum += 0.75 * jacc + 0.25 * attr_bonus

    denom = max(1, max(len(ref_orgs), len(pred_orgs)))
    false_org_penalty = 0.15 * max(0, len(pred_orgs) - len(ref_orgs))
    raw_ratio = max(0.0, min(1.0, (matched_jacc_sum / denom) - false_org_penalty))
    return pred_to_ref, raw_ratio


def score_prediction(
    pred: Dict[str, Any],
    ref: Dict[str, Any],
    seed: Dict[str, Any],
    dollars_per_1k_msgs: float = 0.0,
    schema_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Score output.json against reference answers using the official 100-point breakdown."""
    schema = load_schema(schema_path)
    schema_errors = validate_json_against_schema(pred, schema)
    pred = _normalize_payload(pred)
    ref = _normalize_payload(ref)

    # Extract seeded people IDs to exclude from Circle scoring per official rubric
    seeded_pids: Set[str] = set()
    raw_sp = seed.get("people", seed.get("circles", {}))
    if isinstance(raw_sp, dict):
        seeded_pids.update(raw_sp.keys())
    elif isinstance(raw_sp, list):
        for item in raw_sp:
            if isinstance(item, dict) and item.get("id"):
                seeded_pids.add(str(item["id"]))

    # 1. Circles (12 points) — people in seed.json excluded + staff supports_people accuracy
    pred_people = {p["id"]: p for p in pred.get("people", [])}
    ref_people = {p["id"]: p for p in ref.get("people", [])}
    eval_pids = [pid for pid in ref_people if pid not in seeded_pids]

    correct_circles = 0
    circle_errors: List[Dict[str, str]] = []
    staff_sup_hits = 0
    staff_sup_total = 0

    for pid in eval_pids:
        rp = ref_people[pid]
        pp = pred_people.get(pid)
        if pp and pp.get("circle") == rp.get("circle"):
            correct_circles += 1
        else:
            circle_errors.append(
                {
                    "id": pid,
                    "expected": rp.get("circle", ""),
                    "predicted": pp.get("circle", "MISSING") if pp else "MISSING",
                }
            )

    for pid, rp in ref_people.items():
        if rp.get("circle") == "staff" and rp.get("supports_people"):
            staff_sup_total += 1
            pp = pred_people.get(pid, {})
            if set(pp.get("supports_people", [])) == set(rp.get("supports_people", [])):
                staff_sup_hits += 1

    circle_acc = correct_circles / max(1, len(eval_pids))
    sup_acc = (staff_sup_hits / staff_sup_total) if staff_sup_total > 0 else 1.0
    pts_circles = round(12.0 * (0.88 * circle_acc + 0.12 * sup_acc), 3)

    # 2. Organisations (8 points) — matched by chats & people
    pred_to_ref_org, org_ratio = match_organisations(
        pred.get("organisations", []), ref.get("organisations", [])
    )
    pts_orgs = round(8.0 * org_ratio, 3)

    # 3. Chats (8 points) — tied to right person or matched organisation
    pred_chats = {c["chat_id"]: c for c in pred.get("chats", [])}
    ref_chats = {c["chat_id"]: c for c in ref.get("chats", [])}
    correct_chats = 0
    chat_errors: List[Dict[str, str]] = []

    for cid, rc in ref_chats.items():
        pc = pred_chats.get(cid)
        if not pc:
            chat_errors.append({"chat_id": cid, "expected": f"{rc['owner_type']}:{rc['owner_id']}", "predicted": "MISSING"})
            continue
        if pc["owner_type"] == rc["owner_type"]:
            if pc["owner_type"] == "person" and pc["owner_id"] == rc["owner_id"]:
                correct_chats += 1
                continue
            if (
                pc["owner_type"] == "organisation"
                and pred_to_ref_org.get(pc["owner_id"]) == rc["owner_id"]
            ):
                correct_chats += 1
                continue
        chat_errors.append(
            {
                "chat_id": cid,
                "expected": f"{rc['owner_type']}:{rc['owner_id']}",
                "predicted": f"{pc['owner_type']}:{pc['owner_id']}",
            }
        )
    pts_chats = round(8.0 * (correct_chats / max(1, len(ref_chats))), 3)

    # 4. Goals (8 points) — evidence-backed matches minus penalty for invented goals
    pred_goals = pred.get("goals", [])
    ref_goals = ref.get("goals", [])
    matched_ref_goals: Set[int] = set()
    invented_goals = 0

    for pg in pred_goals:
        p_mids = set(pg.get("message_ids", []))
        p_toks = set(tokenize(pg.get("title", ""), drop_stopwords=True))
        best_ri = None
        best_g_sim = 0.0
        for ri, rg in enumerate(ref_goals):
            if ri in matched_ref_goals:
                continue
            r_mids = set(rg.get("message_ids", []))
            r_toks = set(tokenize(rg.get("title", ""), drop_stopwords=True))
            ev_overlap = len(p_mids & r_mids) / max(1, len(r_mids))
            tok_jacc = _jaccard(p_toks, r_toks)
            area_match = 1.0 if pg.get("area") == rg.get("area") else 0.0
            sim = 0.50 * ev_overlap + 0.30 * tok_jacc + 0.20 * area_match
            if sim > best_g_sim:
                best_g_sim = sim
                best_ri = ri
        if best_ri is not None and best_g_sim >= 0.38:
            matched_ref_goals.add(best_ri)
        else:
            invented_goals += 1

    goal_recall = len(matched_ref_goals) / max(1, len(ref_goals))
    pts_goals = round(max(0.0, 8.0 * goal_recall - 1.5 * invented_goals), 3)

    # 5. Open Loops Discovery (14 points) — precision & recall
    pred_loops = pred.get("open_loops", [])
    ref_loops = ref.get("open_loops", [])

    matched_loop_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    used_ref_loops: Set[int] = set()
    used_pred_loops: Set[int] = set()

    for pi, pl in enumerate(pred_loops):
        p_toks = set(tokenize(pl.get("title", ""), drop_stopwords=True))
        best_ri = None
        best_score = 0.0
        for ri, rl in enumerate(ref_loops):
            if ri in used_ref_loops:
                continue
            score = 0.0
            # Exact or Acceptable Thread Origin Message Match
            allowed_origins = set(rl.get("acceptable_origin_ids", [rl["message_id"]]))
            if pl.get("message_id") in allowed_origins:
                score += 0.70
            elif pl.get("chat_id") == rl.get("chat_id"):
                score += 0.20
            r_toks = set(tokenize(rl.get("title", ""), drop_stopwords=True))
            score += 0.30 * _jaccard(p_toks, r_toks)
            if score > best_score:
                best_score = score
                best_ri = ri
        if best_ri is not None and best_score >= 0.45:
            used_ref_loops.add(best_ri)
            used_pred_loops.add(pi)
            matched_loop_pairs.append((pl, ref_loops[best_ri]))

    tp_loops = len(matched_loop_pairs)
    loop_prec = tp_loops / max(1, len(pred_loops)) if pred_loops else 0.0
    loop_rec = tp_loops / max(1, len(ref_loops)) if ref_loops else 1.0
    loop_f1 = (
        (2 * loop_prec * loop_rec) / (loop_prec + loop_rec)
        if (loop_prec + loop_rec) > 0
        else 0.0
    )
    # Also factor owner & due_date accuracy on matched loops
    owner_hits = sum(
        1 for pl, rl in matched_loop_pairs if pl.get("owner") == rl.get("owner")
    )
    due_hits = sum(
        1 for pl, rl in matched_loop_pairs if pl.get("due_date") == rl.get("due_date")
    )
    meta_acc = (
        0.65 * (owner_hits / max(1, tp_loops)) + 0.35 * (due_hits / max(1, tp_loops))
        if tp_loops > 0
        else 0.0
    )
    pts_loop_discovery = round(14.0 * (0.75 * loop_f1 + 0.25 * meta_acc), 3)

    # 6. Open Loops Closure (14 points) — asymmetric penalty on false `done`!
    closure_earned = 0.0
    closure_max = max(1.0, float(len(matched_loop_pairs)))
    false_done_count = 0
    missed_done_count = 0
    closure_errors: List[Dict[str, Any]] = []

    for pl, rl in matched_loop_pairs:
        p_st = pl.get("status")
        r_st = rl.get("status")
        if p_st == r_st:
            if r_st in ("done", "dropped"):
                allowed_cl = set(
                    rl.get("acceptable_closure_ids", [rl.get("closure_message_id")])
                )
                if pl.get("closure_message_id") in allowed_cl:
                    closure_earned += 1.0
                else:
                    closure_earned += 0.45  # right status, wrong closure message
                    closure_errors.append(
                        {"title": rl["title"], "issue": "wrong_closure_evidence", "predicted": pl.get("closure_message_id"), "expected": list(allowed_cl)}
                    )
            else:
                closure_earned += 1.0
        else:
            if p_st == "done" and r_st != "done":
                # Heavy asymmetric penalty for false DONE ("Saying 'done' when it isn't costs more than missing a done")
                closure_earned -= 1.0
                false_done_count += 1
                closure_errors.append(
                    {"title": rl["title"], "issue": "FALSE_DONE", "predicted": p_st, "expected": r_st}
                )
            elif r_st == "done" and p_st == "open":
                missed_done_count += 1
                closure_errors.append(
                    {"title": rl["title"], "issue": "missed_done", "predicted": p_st, "expected": r_st}
                )
            else:
                closure_earned -= 0.35
                closure_errors.append(
                    {"title": rl["title"], "issue": "status_mismatch", "predicted": p_st, "expected": r_st}
                )

    pts_loop_closure = round(14.0 * max(0.0, closure_earned / closure_max), 3)

    # 7. Where to Help / Intervention (10 points) — heavy penalty for assist/remind on `leave`!
    open_ref_pairs = [
        (pl, rl)
        for pl, rl in matched_loop_pairs
        if rl.get("status") == "open" and pl.get("status") == "open"
    ]
    interv_earned = 0.0
    interv_max = max(1.0, float(len(open_ref_pairs)))
    false_interventions_on_leave = 0
    interv_errors: List[Dict[str, Any]] = []

    for pl, rl in open_ref_pairs:
        piv = pl.get("intervention") or {}
        riv = rl.get("intervention") or {}
        p_act = piv.get("action")
        r_act = riv.get("action")

        if p_act == r_act:
            sub_credit = 1.0
            if r_act == "assist" and riv.get("needs_approval") is not None:
                if piv.get("needs_approval") != riv.get("needs_approval"):
                    sub_credit = 0.65
            elif r_act == "leave" and riv.get("leave_reason"):
                if piv.get("leave_reason") != riv.get("leave_reason"):
                    sub_credit = 0.80
            interv_earned += sub_credit
        else:
            if r_act == "leave" and p_act in ("assist", "remind"):
                # Heaviest penalty per official rubric
                interv_earned -= 1.25
                false_interventions_on_leave += 1
                interv_errors.append(
                    {"title": rl["title"], "issue": "FALSE_INTERVENTION_ON_LEAVE", "predicted": p_act, "expected": "leave"}
                )
            else:
                interv_earned -= 0.20
                interv_errors.append(
                    {"title": rl["title"], "issue": "action_mismatch", "predicted": p_act, "expected": r_act}
                )

    pts_intervention = round(10.0 * max(0.0, interv_earned / interv_max), 3)

    # 8. API Cost (12 points) — based on $/1,000 messages
    if dollars_per_1k_msgs <= 0.015:
        pts_cost = 12.0
    elif dollars_per_1k_msgs >= 0.50:
        pts_cost = 0.0
    else:
        pts_cost = round(12.0 * (1.0 - ((dollars_per_1k_msgs - 0.015) / 0.485) ** 0.65), 3)

    # 9. Reliable Execution check (7 points; write-up 7 points is human-graded)
    pts_execution = 7.0 if not schema_errors else 0.0

    automated_86 = round(
        pts_circles
        + pts_orgs
        + pts_chats
        + pts_goals
        + pts_loop_discovery
        + pts_loop_closure
        + pts_intervention
        + pts_cost,
        3,
    )

    return {
        "schema_valid": len(schema_errors) == 0,
        "schema_errors": schema_errors,
        "automated_score_out_of_86": automated_86,
        "automated_plus_exec_out_of_93": round(automated_86 + pts_execution, 3),
        "breakdown": {
            "circles_12": pts_circles,
            "organisations_8": pts_orgs,
            "chats_8": pts_chats,
            "goals_8": pts_goals,
            "open_loops_found_14": pts_loop_discovery,
            "open_loops_closed_14": pts_loop_closure,
            "intervention_10": pts_intervention,
            "cost_12": pts_cost,
            "execution_7": pts_execution,
        },
        "diagnostics": {
            "circle_accuracy_unseeded": round(circle_acc, 4),
            "staff_support_accuracy": round(sup_acc, 4),
            "org_match_ratio": round(org_ratio, 4),
            "chat_accuracy": round(correct_chats / max(1, len(ref_chats)), 4),
            "goal_recall": round(goal_recall, 4),
            "invented_goals": invented_goals,
            "loop_precision": round(loop_prec, 4),
            "loop_recall": round(loop_rec, 4),
            "loop_f1": round(loop_f1, 4),
            "false_done_count": false_done_count,
            "missed_done_count": missed_done_count,
            "false_interventions_on_leave": false_interventions_on_leave,
            "dollars_per_1k_messages": round(dollars_per_1k_msgs, 5),
        },
        "errors": {
            "circles": circle_errors,
            "chats": chat_errors,
            "closures": closure_errors,
            "interventions": interv_errors,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score output.json against reference answers.")
    parser.add_argument("data_dir", help="Dataset directory containing seed.json and answers/reference.json")
    parser.add_argument("output_json", help="Predicted output.json")
    parser.add_argument("--cost-per-1k", type=float, default=0.0)
    args = parser.parse_args()

    dpath = Path(args.data_dir)
    pred = json.loads(Path(args.output_json).read_text("utf-8"))
    seed = json.loads((dpath / "seed.json").read_text("utf-8"))
    ref_path = dpath / "answers" / "reference.json"
    ref = json.loads(ref_path.read_text("utf-8"))

    report = score_prediction(pred, ref, seed, dollars_per_1k_msgs=args.cost_per_1k)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["schema_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
