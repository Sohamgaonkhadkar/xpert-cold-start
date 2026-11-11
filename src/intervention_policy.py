"""PRISM-inspired Risk-Sensitive Asymmetric Intervention Policy (assist | remind | leave)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from .conversation_index import ConversationIndex
from .indic_normalizer import normalize_indic_for_matching
from .llm_client import LLMClient
from .models import (
    USER_ID,
    VALID_INTERVENTION_ACTIONS,
    VALID_LEAVE_REASONS,
    EvidenceLedger,
    InterventionPrediction,
    OpenLoopPrediction,
    PersonPrediction,
)

# Asymmetric penalty matrix L(predicted_action, true_action)
# Official rule: "A wrong `assist` or `remind` on a `leave` costs most."
LOSS_MATRIX: Dict[str, Dict[str, float]] = {
    "assist": {"assist": 0.0, "remind": 1.05, "leave": 3.20},
    "remind": {"assist": 0.95, "remind": 0.0, "leave": 2.85},
    "leave": {"assist": 1.15, "remind": 1.05, "leave": 0.0},
}

# Tasks where an AI assistant can add concrete preparatory value (`assist`)
ASSIST_CAPABLE_RE = re.compile(
    r"\b(options|shortlist|compare|research|find|lookup|look\s+up|khoj|pata\s+kar|"
    r"phone\s+number|contact\s+number|number\s+ta|address|details|"
    r"hotel|hotels|homestay|flight|flights|tickets|ticket|book|booking|order|itinerary|restaurant|gift|cake|bakery|dokan|dokaner|"
    r"draft|reply|email|proposal|deck|slides|spreadsheet|summary|summarize|"
    r"review\s+the\s+(?:doc|pdf|agreement|contract|deck)|prepare|compile|agenda|schedule|slot|slots|"
    r"quote|quotes|vendor|vendors|plan\s+the)\b",
    re.IGNORECASE,
)

# Outward-facing or irreversible actions that require user approval (`needs_approval = True`)
NEEDS_APPROVAL_RE = re.compile(
    r"\b(send|share|email|message|reply|book|reserve|pay|transfer|order|purchase|confirm\s+with|submit|sign)\b",
    re.IGNORECASE,
)

# Actions that ONLY the user can physically or personally execute (`remind`)
PERSONAL_EXECUTION_ONLY_RE = re.compile(
    r"\b(call\s+|phone\s+|talk\s+to|speak\s+to|meet\s+|attend|visit|bring|carry|pick\s+up|"
    r"sign\s+the\s+physical|hand\s+over|cash|otp|bank\s+kyc|blood\s+test|medicine|workout|gym|run)\b",
    re.IGNORECASE,
)

# Repeated pushy/nagging pings or low-value pestering (`leave: annoying`)
ANNOYING_NAG_RE = re.compile(
    r"\b(3rd\s+time|third\s+time|4th\s+time|stop\s+pinging|stop\s+nagging|already\s+reminded|"
    r"don't\s+remind|dont\s+remind|jab\s+free\s+ho|no\s+rush\s+whenever)\b",
    re.IGNORECASE,
)

# Emotionally intimate, sensitive, or uncommitted soft household musing (`leave: personal`)
PERSONAL_SENSITIVE_RE = re.compile(
    r"\b(apologize|upset|angry|fight|argument|miss\s+you|love\s+you|feelings|"
    r"therapy|counselling|private|personal|secret|anniversary\s+note|prayer|condolence|funeral|"
    r"kuch\s+sochna\s+padega|kuch\s+dekhna\s+padega)\b",
    re.IGNORECASE,
)


def _select_bayes_optimal_action(probs: Dict[str, float]) -> str:
    """Choose action minimizing expected asymmetric loss sum_y P(y|x) * L(action, y)."""
    best_act = "leave"
    best_loss = float("inf")
    for candidate_act, row in LOSS_MATRIX.items():
        exp_loss = sum(probs.get(true_y, 0.0) * cost for true_y, cost in row.items())
        if exp_loss < best_loss:
            best_loss = exp_loss
            best_act = candidate_act
    return best_act


def _compute_remind_timestamp(
    due_date_str: Optional[str], anchor_ts: datetime, tz_suffix: str = ""
) -> str:
    """Compute a sensible single reminder timestamp >= anchor_ts (never in the past), preserving dataset timezone."""
    anchor_date = anchor_ts.date()
    if due_date_str:
        try:
            due_d = datetime.strptime(due_date_str, "%Y-%m-%d").date()
            if due_d > anchor_date:
                if (due_d - anchor_date).days >= 2:
                    rem_dt = datetime.combine(
                        due_d - timedelta(days=1), datetime.min.time()
                    ).replace(hour=18, minute=0)
                else:
                    rem_dt = datetime.combine(due_d, datetime.min.time()).replace(
                        hour=9, minute=0
                    )
                if rem_dt > anchor_ts:
                    return rem_dt.strftime("%Y-%m-%dT%H:%M:%S") + tz_suffix
        except ValueError:
            pass

    if anchor_ts.hour < 14:
        rem_dt = anchor_ts.replace(hour=18, minute=0, second=0, microsecond=0)
    else:
        next_day = anchor_date + timedelta(days=1)
        rem_dt = datetime.combine(next_day, datetime.min.time()).replace(
            hour=9, minute=0
        )
    return rem_dt.strftime("%Y-%m-%dT%H:%M:%S") + tz_suffix


def _build_assist_step(
    loop: OpenLoopPrediction, origin_body: str
) -> Tuple[str, bool]:
    """Generate a concise 1-line actionable next_step and calibrate needs_approval."""
    low = f"{loop.title} {origin_body}".lower()

    if any(w in low for w in ("book", "booking", "tickets", "ticket")) and any(
        w in low for w in ("homestay", "hotel", "flight", "resort", "trip", "stay")
    ):
        return (
            f"Shortlist options and rates for '{loop.title[:50]}' and ask for approval before booking.",
            True,
        )
    if any(w in low for w in ("hotel", "homestay", "flight", "itinerary", "resort", "trip", "stay")):
        return (
            f"Shortlist and compare top options with current rates for '{loop.title[:55]}' for review.",
            False,
        )
    if any(w in low for w in ("khoj", "pata kar", "number", "address", "dokan", "dokaner")):
        return (
            f"Look up verified contact number and address details for '{loop.title[:55]}' privately for review.",
            False,
        )
    if any(w in low for w in ("cake", "gift", "restaurant", "bakery", "florist")):
        return (
            f"Find top nearby options and availability for '{loop.title[:55]}' before placing any order.",
            False,
        )
    if any(w in low for w in ("review", "agreement", "contract", "pdf", "document", "clauses")):
        return (
            f"Summarize key points and open questions in '{loop.title[:55]}' privately for review.",
            False,
        )
    if any(w in low for w in ("slot", "slots", "schedule", "calendar", "meeting")):
        return (
            f"Check free calendar slots and draft a scheduling reply for '{loop.title[:50]}' awaiting user approval.",
            True,
        )
    if loop.owner != USER_ID and loop.type == "followup":
        return (
            f"Draft a polite follow-up message checking status on '{loop.title[:50]}' for user approval before sending.",
            True,
        )
    if any(w in low for w in ("proposal", "deck", "slides", "reply", "email", "quote", "invoice")):
        return (
            f"Prepare a draft response and supporting details for '{loop.title[:50]}' and ask for approval before sending.",
            True,
        )

    needs_yes = bool(NEEDS_APPROVAL_RE.search(low))
    if needs_yes:
        return (
            f"Prepare a draft for '{loop.title[:55]}' and confirm with the user before sending.",
            True,
        )
    return (
        f"Compile relevant details and options for '{loop.title[:55]}' privately for review.",
        False,
    )


INTERVENTION_SYSTEM_PROMPT = """You select the optimal assistant intervention for OPEN WhatsApp loops of ME.
CRITICAL ASYMMETRIC PENALTY: Choosing `assist` or `remind` when the right answer is `leave` is HEAVILY PENALIZED.
Choose ONE per loop:
- `leave`: Stay out of it! Required when:
  * `someone_elses`: Owned by staff (driver/cook/maid), spouse, colleague, or third party already handling it, and not an overdue item blocking ME.
  * `personal`: Intimate/emotional spouse, child, parent, or close friend conversation.
  * `annoying`: Low-stakes chatter or routine item where poking ME would be annoying.
- `assist`: Assistant can move it forward itself (research options, summarize a doc, compile numbers, or draft a reply).
  * `needs_approval`: `false` ONLY if the step is 100% private & reversible preparation on ME's phone; `true` if sending a message, booking, or spending money.
- `remind`: Only ME personally can do it (make a personal phone call, physically bring/sign something, pay via OTP, attend). Provide `remind_at` (ISO timestamp after dataset end).

Return JSON:
{"decisions": [{"id": "<loop_id>", "action": "assist|remind|leave", "next_step": "<1 line or null>", "needs_approval": true, "remind_at": "<ISO or null>", "leave_reason": "someone_elses|personal|annoying|null", "p_leave": 0.0, "p_assist": 0.0, "p_remind": 0.0}]}"""


def apply_intervention_policy(
    loops: List[OpenLoopPrediction],
    index: ConversationIndex,
    people: Dict[str, PersonPrediction],
    llm: LLMClient,
    ledger: EvidenceLedger,
) -> None:
    """Populate `loop.intervention` for open loops (and `None` for closed/dropped loops)."""
    boundary_loops: List[OpenLoopPrediction] = []

    for loop in loops:
        if loop.status != "open":
            loop.intervention = None
            continue

        origin_msg = index.messages_by_id.get(loop.message_id)
        origin_body = origin_msg.body if origin_msg else loop.title
        raw_comb = f"{loop.title} {origin_body}"
        combined_text = normalize_indic_for_matching(raw_comb) if not raw_comb.isascii() else raw_comb

        # Identify circle of owner / counterparty
        chat = index.chats.get(loop.chat_id)
        non_me_parts = [p for p in (chat.participant_ids if chat else ()) if p != USER_ID]
        counterparty_id = (
            loop.owner
            if loop.owner != USER_ID
            else (
                origin_msg.sender_id
                if origin_msg and origin_msg.sender_id != USER_ID
                else (non_me_parts[0] if non_me_parts else None)
            )
        )
        cp_circle = (
            people[counterparty_id].circle
            if counterparty_id and counterparty_id in people
            else "colleague"
        )

        # Hand-set class weights; dev intervention labels do not calibrate them.
        probs = _estimate_action_posteriors(
            loop=loop,
            combined_text=combined_text,
            cp_circle=cp_circle,
            origin_msg_ts=origin_msg.ts if origin_msg else index.anchor_ts,
            anchor_ts=index.anchor_ts,
            waiting_for_me=bool(origin_msg and (
                origin_msg.sender_id == USER_ID
                or (chat and not chat.is_group and loop.type == "followup")
            )),
        )
        chosen = _select_bayes_optimal_action(probs)
        max_p = max(probs.values())

        loop.intervention = _construct_intervention(
            action=chosen,
            loop=loop,
            origin_body=origin_body,
            cp_circle=cp_circle,
            anchor_ts=index.anchor_ts,
            confidence=max_p,
            tz_suffix=index.bundle.tz_suffix,
        )

        if max_p < 0.78:
            boundary_loops.append(loop)

    # Selective LLM slow deliberation ONLY on boundary open loops (`PRISM` Festina Lente gate)
    if llm.is_available and boundary_loops:
        lines: List[str] = [
            f"Dataset End Timestamp: {index.anchor_ts.strftime('%Y-%m-%dT%H:%M:%S')}"
        ]
        for lp in boundary_loops[:20]:
            om = index.messages_by_id.get(lp.message_id)
            cp_info = []
            if lp.chat_id in index.chats:
                for p in index.chats[lp.chat_id].participant_ids:
                    if p != USER_ID and p in people:
                        cp_info.append(f"{p}({people[p].name}:{people[p].circle})")
            lines.append(
                f"LOOP {lp.id} | type={lp.type} | owner={lp.owner} | due={lp.due_date} | "
                f"participants={cp_info[:5]} | title=\"{lp.title}\"\n"
                f"Origin: {index.format_turn(om) if om else lp.title}"
            )
        res = llm.complete_json(
            "intervention_policy",
            INTERVENTION_SYSTEM_PROMPT,
            "\n\n".join(lines),
            max_tokens=1500,
        )
        by_id = {lp.id: lp for lp in boundary_loops}
        if isinstance(res, dict):
            for item in res.get("decisions", []) or []:
                if not isinstance(item, dict):
                    continue
                lid = str(item.get("id") or "").strip()
                if lid not in by_id:
                    continue
                lp = by_id[lid]
                # Re-apply expected-loss gate over LLM's posterior probabilities if supplied!
                p_l = float(item.get("p_leave") or 0.0)
                p_a = float(item.get("p_assist") or 0.0)
                p_r = float(item.get("p_remind") or 0.0)
                if (p_l + p_a + p_r) > 0.5:
                    tot = p_l + p_a + p_r
                    act = _select_bayes_optimal_action(
                        {"leave": p_l / tot, "assist": p_a / tot, "remind": p_r / tot}
                    )
                else:
                    act = str(item.get("action") or "leave").strip().lower()
                if act not in VALID_INTERVENTION_ACTIONS:
                    continue

                # Hard structural override: never let LLM intervene on a staff-owned routine task!
                if lp.owner != USER_ID and lp.owner in people and people[lp.owner].circle == "staff":
                    act = "leave"

                om = index.messages_by_id.get(lp.message_id)
                default_iv = _construct_intervention(
                    action=act,
                    loop=lp,
                    origin_body=om.body if om else lp.title,
                    cp_circle=people[lp.owner].circle if lp.owner in people else "colleague",
                    anchor_ts=index.anchor_ts,
                    confidence=0.82,
                    tz_suffix=index.bundle.tz_suffix,
                )
                if act == "assist":
                    ns = str(item.get("next_step") or "").strip() or default_iv.next_step
                    na = item.get("needs_approval")
                    if not isinstance(na, bool):
                        na = default_iv.needs_approval
                    # Safety floor: any outward-facing verb MUST require approval!
                    if ns and NEEDS_APPROVAL_RE.search(ns):
                        na = True
                    default_iv.next_step = ns
                    default_iv.needs_approval = bool(na)
                elif act == "remind":
                    ra = str(item.get("remind_at") or "").strip()
                    if re.match(r"^202\d-\d{2}-\d{2}", ra):
                        default_iv.remind_at = ra
                elif act == "leave":
                    lr = str(item.get("leave_reason") or "").strip()
                    if lr in VALID_LEAVE_REASONS:
                        default_iv.leave_reason = lr
                lp.intervention = default_iv

    for loop in loops:
        if loop.status == "open" and loop.intervention is not None:
            ledger.record(
                subject_type="intervention",
                subject_id=loop.id,
                field_name="action",
                predicted_value={
                    "action": loop.intervention.action,
                    "needs_approval": loop.intervention.needs_approval,
                    "leave_reason": loop.intervention.leave_reason,
                },
                strength="explicit" if loop.intervention.confidence >= 0.8 else "contextual",
                confidence=loop.intervention.confidence,
                message_ids=[loop.message_id],
                rationale=f"PRISM expected-loss optimal action={loop.intervention.action}",
                stage="intervention_policy",
            )


def _estimate_action_posteriors(
    loop: OpenLoopPrediction,
    combined_text: str,
    cp_circle: str,
    origin_msg_ts: datetime,
    anchor_ts: datetime,
    waiting_for_me: bool = False,
) -> Dict[str, float]:
    """Compute hand-set action weights from structural and lexical features."""
    # Case 1: Owned by someone OTHER than ME (`loop.owner != "ME"`)
    if loop.owner != USER_ID:
        # Family members' own plans stay theirs. Respect explicit no-nudge requests.
        if cp_circle in ("spouse", "child", "parent", "in-law", "sibling", "relative", "friend") or ANNOYING_NAG_RE.search(combined_text):
            return {"leave": 0.94, "remind": 0.04, "assist": 0.02}

        # Official FAQ (11 Nov): someone else is handling it -> leave; if ME is
        # waiting and it is overdue -> remind. Age alone does not establish a due date.
        is_overdue = bool(loop.due_date and loop.due_date < anchor_ts.date().isoformat())
        if is_overdue and waiting_for_me and cp_circle in ("staff", "vendor", "client", "colleague", "partner"):
            return {"remind": 0.88, "assist": 0.04, "leave": 0.08}
        return {"leave": 0.88, "remind": 0.08, "assist": 0.04}

    # Case 2: Owned by ME (`loop.owner == "ME"`)
    # Intimate / emotionally sensitive family or friend matter -> `leave` (`personal`)
    if PERSONAL_SENSITIVE_RE.search(combined_text) and cp_circle in (
        "spouse", "child", "parent", "in-law", "sibling", "relative", "friend"
    ):
        return {"leave": 0.92, "remind": 0.06, "assist": 0.02}

    if ANNOYING_NAG_RE.search(combined_text):
        return {"leave": 0.90, "remind": 0.06, "assist": 0.04}

    # Casual social banter / trivial peer question with no real stakes
    if cp_circle == "friend" and loop.due_date is None and not ASSIST_CAPABLE_RE.search(combined_text):
        if len(combined_text.split()) < 8:
            return {"leave": 0.78, "remind": 0.16, "assist": 0.06}

    # Dated creative/work deadlines where ME is already executing ("due Friday, I'm on it") or judgment decisions ("Should we take Zest?") -> remind!
    if loop.type == "deadline" or (
        loop.due_date is not None and re.search(r"\b(i'm\s+on\s+it|i\s+am\s+on\s+it|is\s+due)\b", combined_text, re.IGNORECASE)
    ):
        return {"remind": 0.85, "assist": 0.10, "leave": 0.05}

    if loop.type == "question" and not ASSIST_CAPABLE_RE.search(combined_text):
        return {"remind": 0.84, "assist": 0.08, "leave": 0.08}

    has_assist = bool(ASSIST_CAPABLE_RE.search(combined_text))
    has_personal_exec = bool(PERSONAL_EXECUTION_ONLY_RE.search(combined_text))

    if has_assist and not has_personal_exec:
        return {"assist": 0.84, "remind": 0.11, "leave": 0.05}
    if has_assist and has_personal_exec:
        # E.g. "Find a cake shop and call Mum" -> assist can still prepare options privately!
        if any(w in combined_text.lower() for w in ("find", "options", "shortlist", "draft", "compare", "hotel", "flight", "cake")):
            return {"assist": 0.76, "remind": 0.18, "leave": 0.06}
        return {"remind": 0.76, "assist": 0.17, "leave": 0.07}
    if has_personal_exec or loop.due_date is not None or cp_circle in ("client", "partner", "colleague", "vendor", "parent", "spouse", "child"):
        return {"remind": 0.80, "assist": 0.10, "leave": 0.10}

    return {"leave": 0.65, "remind": 0.25, "assist": 0.10}


def _construct_intervention(
    action: str,
    loop: OpenLoopPrediction,
    origin_body: str,
    cp_circle: str,
    anchor_ts: datetime,
    confidence: float,
    tz_suffix: str = "",
) -> InterventionPrediction:
    if action == "assist":
        step, needs_yes = _build_assist_step(loop, origin_body)
        return InterventionPrediction(
            action="assist",
            next_step=step,
            needs_approval=needs_yes,
            remind_at=None,
            leave_reason=None,
            confidence=round(confidence, 3),
        )
    if action == "remind":
        rem_at = _compute_remind_timestamp(loop.due_date, anchor_ts, tz_suffix=tz_suffix)
        return InterventionPrediction(
            action="remind",
            next_step=None,
            needs_approval=None,
            remind_at=rem_at,
            leave_reason=None,
            confidence=round(confidence, 3),
        )
    # `leave`
    if loop.owner != USER_ID:
        reason = "someone_elses"
    elif ANNOYING_NAG_RE.search(f"{loop.title} {origin_body}"):
        reason = "annoying"
    elif PERSONAL_SENSITIVE_RE.search(f"{loop.title} {origin_body}") or cp_circle in (
        "spouse", "child", "parent", "in-law"
    ):
        reason = "personal"
    else:
        reason = "annoying"
    return InterventionPrediction(
        action="leave",
        next_step=None,
        needs_approval=None,
        remind_at=None,
        leave_reason=reason,
        confidence=round(confidence, 3),
    )
