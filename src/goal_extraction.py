"""Precision-first, evidence-grounded Goal Extraction (capped at <= 10)."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .conversation_index import ConversationIndex, tokenize
from .llm_client import LLMClient
from .models import (
    USER_ID,
    VALID_GOAL_AREAS,
    VALID_GOAL_HORIZONS,
    VALID_GOAL_TYPES,
    EvidenceLedger,
    GoalPrediction,
    MessageRecord,
    PersonPrediction,
)

# Explicit goal / rhythm / target markers in English, Hinglish, and Bengali
EXPLICIT_GOAL_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"\b(i want to|i'm trying to|i am trying to|my goal|our goal|aiming to|aiming for|"
        r"target is|is the target|targeting|working towards|plan to|planning to|committed to|"
        r"going to start|consistent with|sticking to|by (?:march|april|may|june|july|august|september|october|november|december|january|february|q[1-4]|diwali|year end|next year))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d+\s*(?:times|days|x|runs|workouts|sessions|km|k|books)\s*(?:a|per)\s*(?:week|month)|every\s+(?:morning|day|tuesday|thursday|weekend|sunday)|"
        r"roz\s+|har\s+hafte|haftay\s+mein|protidin|shoptah)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(planning\s+(?:a|our|the)\s+[a-z\s]{2,25}\s*(?:trip|vacation|holiday|trek|getaway)|"
        r"book\s+the\s+tickets\s+and\s+(?:the\s+)?(?:homestay|hotel|resort)|"
        r"(?:half\s+)?marathon\s+in\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*|"
        r"chalte\s+hain\s+[a-z]+|trip\s+plan)\b",
        re.IGNORECASE,
    ),
]

WEAK_WISH_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(wish i lived|someday maybe|in another life|just kidding|jk\b|lol\b|haha\b|imagine if)\b",
    re.IGNORECASE,
)

AREA_KEYWORDS: Dict[str, Set[str]] = {
    "health": {
        "gym", "workout", "run", "running", "jog", "yoga", "fitness", "weight", "kg",
        "diet", "protein", "calories", "veg", "sleep", "walk", "steps", "marathon",
        "sugar", "cholesterol", "health", "swimming", "tennis", "exercise",
    },
    "work": {
        "clients", "revenue", "arr", "mrr", "launch", "product", "hire", "hiring",
        "sales", "pitch", "fundraise", "seed", "series", "company", "growth", "paying",
        "enterprise", "pipeline", "promotion", "role",
    },
    "finances": {
        "invest", "sip", "mutual", "portfolio", "savings", "save", "tax", "loan",
        "mortgage", "emi", "corpus", "net worth", "budget", "insurance", "fd",
    },
    "home": {
        "renovate", "renovation", "kitchen", "interior", "painting", "furniture",
        "solar", "garden", "buy a home", "flat", "apartment", "move in", "shifting",
    },
    "experiences": {
        "trip", "travel", "vacation", "holiday", "holidays", "goa", "coorg", "manali", "kashmir",
        "kerala", "japan", "europe", "hills", "mountains", "trek", "safari", "concert", "dive",
        "scuba", "itinerary", "resort", "homestay",
    },
    "kids": {
        "maths", "math", "exam", "school", "admission", "reading", "chess", "piano",
        "coding", "grades", "tuition", "tutor", "olympiad",
    },
    "parents": {
        "checkup", "surgery", "cataract", "physiotherapy", "parents", "maa", "papa",
        "baba", "mummy", "pilgrimage",
    },
    "spouse": {
        "date night", "anniversary", "couples",
    },
    "friends": {
        "reunion", "monthly dinner", "catchup", "batch",
    },
    "self": {
        "read", "reading", "books", "meditate", "meditation", "journal", "hindi",
        "bengali", "spanish", "guitar", "learn", "course", "writing",
    },
}


def _classify_area_type_horizon(
    text: str, involved_circles: Set[str]
) -> Tuple[str, str, str]:
    low = text.lower()
    toks = set(tokenize(low, drop_stopwords=False))

    best_area = "self"
    best_score = 0
    for area, kws in AREA_KEYWORDS.items():
        hits = sum(1 for kw in kws if (kw in toks if " " not in kw else kw in low))
        if hits > best_score:
            best_score = hits
            best_area = area

    if best_score == 0:
        if "child" in involved_circles:
            best_area = "kids"
        elif "parent" in involved_circles:
            best_area = "parents"
        elif "colleague" in involved_circles or "partner" in involved_circles:
            best_area = "work"

    # Type: habit | milestone | experience
    has_milestone_event = bool(
        re.search(
            r"\b(marathon|half\s+marathon|10k|triathlon|clients\s+by|revenue|launch|exam|admission|renovate)\b",
            low,
        )
    )
    if best_area == "experiences" or any(
        w in low for w in ("trip", "vacation", "holiday", "holidays", "trek", "getaway", "tour", "homestay")
    ):
        gtype = "experience"
    elif not has_milestone_event and re.search(
        r"\b(\d+\s*(?:times|days|x|runs|workouts|sessions)\s*(?:a|per)\s*week|every\s+(?:day|morning|night|week|tuesday|thursday|weekend)|daily|weekly|roz|protidin|routine|habit)\b",
        low,
    ):
        gtype = "habit"
    else:
        gtype = "milestone"

    # Horizon: 3mo | 1yr | 3yr | 5yr
    if any(w in low for w in ("5 year", "5yr", "five year", "retire")):
        horizon = "5yr"
    elif any(w in low for w in ("3 year", "3yr", "three year")):
        horizon = "3yr"
    elif any(w in low for w in ("this year", "next year", "1 year", "1yr", "annual", "12 months")):
        horizon = "1yr"
    else:
        horizon = "3mo"

    return best_area, gtype, horizon


GOAL_VERIFY_SYSTEM_PROMPT = """You extract up to 10 genuine goals held by ME from candidate WhatsApp messages.
CRITICAL SCORING RULE: Invented or weakly implied goals COST POINTS. Only output goals with clear evidence in the provided messages!
Reject:
- Jokes, fleeting complaints ("wish I lived on a beach lol"), or one-off errands ("buy milk").
- Goals belonging ONLY to another person ("my friend is running a marathon").

Allowed enums:
- area: self, spouse, kids, parents, work, friends, experiences, finances, health, home
- type: habit (recurring rhythm), milestone (measurable target), experience (planned trip/activity)
- horizon: 3mo, 1yr, 3yr, 5yr

Return JSON:
{"goals": [{"title": "<concise goal>", "area": "<area>", "type": "<type>", "horizon": "<horizon>", "message_ids": ["<id>"], "confidence": 0.0}]}"""


def extract_goals(
    index: ConversationIndex,
    people: Dict[str, PersonPrediction],
    llm: LLMClient,
    ledger: EvidenceLedger,
    max_goals: int = 10,
) -> List[GoalPrediction]:
    """Extract <= 10 high-precision, evidence-backed goals for ME."""
    goals: List[GoalPrediction] = []
    used_msg_sets: List[Set[str]] = []

    # 1. Anchor Seeded Goals that appear in the conversation data
    for idx, s_goal in enumerate(index.bundle.seed.goals):
        matched_ids = [
            mid for mid in s_goal.message_ids if mid in index.messages_by_id
        ]
        if not matched_ids:
            seed_toks = {
                t for t in tokenize(s_goal.title, drop_stopwords=True)
                if t not in {"by", "target", "goal", "plan"}
            }
            hits = index.search(s_goal.title, top_k=8)
            for msg, score in hits:
                m_toks = set(tokenize(msg.body, drop_stopwords=True))
                min_ov = 2 if len(seed_toks) >= 2 else 1
                if (
                    score >= 2.8
                    and len(m_toks & seed_toks) >= min_ov
                    and (msg.sender_id == USER_ID or "my goal" not in msg.body.lower())
                    and len(index.chats[msg.chat_id].participant_ids) <= 6
                ):
                    matched_ids.append(msg.id)
                if len(matched_ids) >= 4:
                    break

        if matched_ids:
            involved_circles = {
                people[index.messages_by_id[mid].sender_id].circle
                for mid in matched_ids
                if index.messages_by_id[mid].sender_id in people
            }
            comb_text = s_goal.title + " " + " ".join(
                index.messages_by_id[mid].body for mid in matched_ids[:3]
            )
            inf_area, inf_type, inf_horizon = _classify_area_type_horizon(
                comb_text, involved_circles
            )
            gp = GoalPrediction(
                id=s_goal.id or f"goal_seed_{idx + 1}",
                title=s_goal.title,
                area=s_goal.area or inf_area,
                type=s_goal.type or inf_type,
                horizon=s_goal.horizon or inf_horizon,
                message_ids=matched_ids[:5],
                confidence=0.97,
                source="seed",
            )
            goals.append(gp)
            used_msg_sets.append(set(gp.message_ids))
            ledger.record(
                subject_type="goal",
                subject_id=gp.id,
                field_name="goal",
                predicted_value=gp.title,
                strength="seed",
                confidence=0.97,
                message_ids=gp.message_ids,
                rationale="Seeded goal corroborated by conversation turns",
                stage="goal_seed",
            )

    # 2. Scan ME-authored messages (and replies endorsed by ME) for strong goal candidates
    candidate_clusters: Dict[str, List[MessageRecord]] = defaultdict(list)

    for msg in index.messages:
        body = msg.body or ""
        if len(body) < 12 or WEAK_WISH_PATTERNS.search(body):
            continue

        is_me = msg.sender_id == USER_ID
        # Allow non-ME message ONLY if it invites/involves ME (never another person's "My goal is...") AND ME endorses it
        me_endorsed_reply = False
        endorsed_reply_msg: Optional[MessageRecord] = None
        other_solo_goal = bool(
            re.search(r"\b(my\s+goal|i'm\s+training\s+for|i\s+am\s+training\s+for|wish\s+me\s+luck)\b", body, re.IGNORECASE)
        )
        if not is_me and not other_solo_goal:
            candidate_replies: List[MessageRecord] = [
                index.messages_by_id[r_id] for r_id in index.replies_to.get(msg.id, [])
            ]
            et = index.enriched_by_id.get(msg.id)
            if et is not None:
                c_msgs = index.by_chat.get(msg.chat_id, [])
                if et.chat_index + 1 < len(c_msgs):
                    nxt = c_msgs[et.chat_index + 1]
                    if (nxt.ts - msg.ts).total_seconds() <= 3600 and nxt not in candidate_replies:
                        candidate_replies.append(nxt)
            for rm in candidate_replies:
                if (
                    rm.sender_id == USER_ID
                    and re.search(
                        r"\b(yes|haan|let's do|lets do|definitely|pakka|done|agreed|aiming|i'll\s+book|will\s+book|doing\s+\d+)\b",
                        rm.body,
                        re.IGNORECASE,
                    )
                    and not re.search(r"\b(cheering\s+for\s+you|proud\s+of\s+you|good\s+luck|all\s+the\s+best)\b", rm.body, re.IGNORECASE)
                ):
                    me_endorsed_reply = True
                    endorsed_reply_msg = rm
                    break
        if not (is_me or me_endorsed_reply):
            continue

        # Exclude operational instructions directed at household staff or external vendors unless explicitly framed as ME's goal
        chat_obj = index.chats.get(msg.chat_id)
        non_me_in_chat = [
            p for p in (chat_obj.participant_ids if chat_obj else ()) if p != USER_ID
        ]
        if (
            len(non_me_in_chat) == 1
            and non_me_in_chat[0] in people
            and people[non_me_in_chat[0]].circle in ("staff", "vendor", "client")
            and not re.search(r"\b(my goal|our goal|aiming to|want to)\b", body, re.IGNORECASE)
        ):
            continue

        pair_text = f"{body} {endorsed_reply_msg.body}" if endorsed_reply_msg else body
        et_self = index.enriched_by_id.get(msg.id)
        prev_prompt_msg: Optional[MessageRecord] = None
        if is_me and et_self is not None and et_self.chat_index > 0:
            c_msgs = index.by_chat.get(msg.chat_id, [])
            pm = c_msgs[et_self.chat_index - 1]
            pm_is_other_solo = bool(
                re.search(r"\b(my\s+goal|i'm\s+training\s+for|i\s+am\s+training\s+for)\b", pm.body, re.IGNORECASE)
            )
            me_commits_in_reply = bool(
                re.search(
                    r"\b(yes|haan|let's do|lets do|definitely|pakka|i'll\s+book|will\s+book|doing\s+\d+)\b",
                    body,
                    re.IGNORECASE,
                )
            ) and not bool(
                re.search(r"\b(cheering\s+for\s+you|proud\s+of\s+you|good\s+luck|all\s+the\s+best)\b", body, re.IGNORECASE)
            )
            if (
                pm.sender_id != USER_ID
                and (msg.ts - pm.ts).total_seconds() <= 1800
                and not pm_is_other_solo
                and me_commits_in_reply
            ):
                prev_prompt_msg = pm
                pair_text = f"{pm.body} {body}"

        if any(pat.search(pair_text) for pat in EXPLICIT_GOAL_PATTERNS):
            chat = index.chats.get(msg.chat_id)
            involved = {
                people[p].circle
                for p in (chat.participant_ids if chat else ())
                if p in people
            }
            area, gtype, _ = _classify_area_type_horizon(pair_text, involved)
            # Topic key: area + chat_id so two sides of the same 2-turn exchange merge into one goal!
            cluster_key = f"{area}:{msg.chat_id}"
            if prev_prompt_msg is not None and prev_prompt_msg not in candidate_clusters[cluster_key]:
                candidate_clusters[cluster_key].append(prev_prompt_msg)
            if msg not in candidate_clusters[cluster_key]:
                candidate_clusters[cluster_key].append(msg)
            if endorsed_reply_msg is not None and endorsed_reply_msg not in candidate_clusters[cluster_key]:
                candidate_clusters[cluster_key].append(endorsed_reply_msg)

    # Expand each candidate cluster with tightly matching supporting turns by ME or in same chat
    vetted_candidates: List[Tuple[str, List[MessageRecord], str, str, str, float]] = []
    for ckey, anchor_msgs in sorted(candidate_clusters.items()):
        primary = anchor_msgs[0]
        if any(primary.id in s for s in used_msg_sets):
            continue
        related = index.search(
            primary.body,
            top_k=6,
            candidate_chat_ids={m.chat_id for m in anchor_msgs},
        )
        ev_msgs: List[MessageRecord] = list(anchor_msgs)
        seen_ids = {m.id for m in ev_msgs}
        for rm, sc in related:
            if sc >= 3.2 and rm.id not in seen_ids and not WEAK_WISH_PATTERNS.search(rm.body):
                ev_msgs.append(rm)
                seen_ids.add(rm.id)
        ev_msgs.sort(key=lambda m: (m.ts, m.id))

        # Precision Gate: require either explicit self-goal phrasing by ME with concrete area keyword
        # OR >= 2 corroborating messages
        chat = index.chats.get(primary.chat_id)
        involved = {
            people[p].circle
            for p in (chat.participant_ids if chat else ())
            if p in people
        }
        comb = " ".join(m.body for m in ev_msgs[:4])
        area, gtype, horizon = _classify_area_type_horizon(comb, involved)

        area_kw_hits = sum(
            1
            for kw in AREA_KEYWORDS.get(area, set())
            if kw in comb.lower()
        )
        me_count = sum(1 for m in ev_msgs if m.sender_id == USER_ID)
        if me_count == 0 or (len(ev_msgs) == 1 and area_kw_hits == 0):
            continue

        conf = min(0.92, 0.58 + 0.10 * len(ev_msgs) + 0.06 * area_kw_hits)
        title = _summarize_goal_title(primary.body)
        vetted_candidates.append((title, ev_msgs[:5], area, gtype, horizon, conf))

    # 3. Optional LLM Precision Verification & Deduplication
    if llm.is_available and vetted_candidates:
        lines: List[str] = []
        for idx, (title, ev_msgs, area, gtype, horizon, conf) in enumerate(
            vetted_candidates[:16]
        ):
            turns_txt = "\n".join(
                index.format_turn(m, include_chat=True) for m in ev_msgs
            )
            lines.append(
                f"CANDIDATE {idx + 1} (prior area={area}, type={gtype}, horizon={horizon}):\n{turns_txt}"
            )
        res = llm.complete_json(
            "goal_extraction",
            GOAL_VERIFY_SYSTEM_PROMPT,
            "\n\n".join(lines),
            max_tokens=1400,
        )
        valid_mids = set(index.messages_by_id.keys())
        if isinstance(res, dict) and isinstance(res.get("goals"), list):
            for idx, item in enumerate(res["goals"]):
                if len(goals) >= max_goals or not isinstance(item, dict):
                    break
                g_conf = float(item.get("confidence") or 0.75)
                if g_conf < 0.65:
                    continue
                raw_mids = item.get("message_ids") or item.get("evidence") or []
                mids = []
                for m in raw_mids:
                    cleaned_m = re.sub(r"[^A-Za-z0-9_-]", "", str(m))
                    if cleaned_m in valid_mids and cleaned_m not in mids:
                        mids.append(cleaned_m)
                if not mids:
                    continue
                if any(len(set(mids) & s) > 0 for s in used_msg_sets):
                    continue
                title = str(item.get("title") or "").strip()
                area = str(item.get("area") or "self").strip()
                gtype = str(item.get("type") or "milestone").strip()
                horizon = str(item.get("horizon") or "3mo").strip()
                if (
                    not title
                    or area not in VALID_GOAL_AREAS
                    or gtype not in VALID_GOAL_TYPES
                    or horizon not in VALID_GOAL_HORIZONS
                ):
                    continue
                gid = f"goal_{len(goals) + 1}"
                gp = GoalPrediction(
                    id=gid,
                    title=title,
                    area=area,
                    type=gtype,
                    horizon=horizon,
                    message_ids=mids[:5],
                    confidence=round(g_conf, 3),
                    source="inferred",
                )
                goals.append(gp)
                used_msg_sets.append(set(mids))
                ledger.record(
                    subject_type="goal",
                    subject_id=gid,
                    field_name="goal",
                    predicted_value=title,
                    strength="explicit" if g_conf >= 0.8 else "contextual",
                    confidence=g_conf,
                    message_ids=gp.message_ids,
                    rationale="Verified by LLM goal precision filter",
                    stage="goal_llm",
                )
            # Only short-circuit deterministic fallback if LLM actually returned >=1 valid non-seed goal
            # OR explicitly evaluated candidates without dropping high-confidence (>=0.82) non-overlapping ones
            vetted_candidates = [c for c in vetted_candidates if c[5] >= 0.82]

    # Deterministic fallback selection sorted by confidence
    vetted_candidates.sort(key=lambda x: -x[5])
    for title, ev_msgs, area, gtype, horizon, conf in vetted_candidates:
        if len(goals) >= max_goals:
            break
        mids = [m.id for m in ev_msgs]
        if any(len(set(mids) & s) > 0 for s in used_msg_sets):
            continue
        gid = f"goal_{len(goals) + 1}"
        gp = GoalPrediction(
            id=gid,
            title=title,
            area=area,
            type=gtype,
            horizon=horizon,
            message_ids=mids,
            confidence=round(conf, 3),
            source="inferred",
        )
        goals.append(gp)
        used_msg_sets.append(set(mids))
        ledger.record(
            subject_type="goal",
            subject_id=gid,
            field_name="goal",
            predicted_value=title,
            strength="explicit" if conf >= 0.78 else "contextual",
            confidence=conf,
            message_ids=mids,
            rationale="Tier-0 precision-gated goal candidate",
            stage="goal_tier0",
        )

    return goals[:max_goals]


def _summarize_goal_title(body: str) -> str:
    clean = re.sub(r"\s+", " ", body).strip()
    # Strip conversational lead-ins
    clean = re.sub(
        r"^(?:hey\s+\w+,?\s*|yaar\s+|bhai\s+|honestly\s+|actually\s+)*(?:i'm\s+trying\s+to|i\s+am\s+trying\s+to|i\s+want\s+to|my\s+goal\s+is\s+to|our\s+goal\s+is\s+to|aiming\s+to|planning\s+to|plan\s+to)\s+",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = clean.split(". ")[0].split("! ")[0].strip(" .,!?")
    if clean:
        clean = clean[0].upper() + clean[1:]
    return clean[:90] if clean else body[:90]
