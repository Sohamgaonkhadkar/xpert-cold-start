"""Multilingual Open-Loop Candidate Extraction, Relative Date Normalization, and Cross-Chat Deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from .conversation_index import ConversationIndex, tokenize
from .indic_normalizer import normalize_indic_for_matching
from .llm_client import LLMClient
from .models import (
    USER_ID,
    VALID_LOOP_TYPES,
    EvidenceLedger,
    MessageRecord,
    OpenLoopPrediction,
    PersonPrediction,
)

MONTH_MAP: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

WEEKDAY_MAP: Dict[str, int] = {
    "monday": 0, "mon": 0, "somwar": 0, "sombar": 0,
    "tuesday": 1, "tue": 1, "tues": 1, "mangalwar": 1, "mongolbar": 1,
    "wednesday": 2, "wed": 2, "budhwar": 2, "budhbar": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "guruwar": 3, "brihospotibar": 3,
    "friday": 4, "fri": 4, "shukrawar": 4, "shukrobar": 4,
    "saturday": 5, "sat": 5, "shaniwar": 5, "shonibar": 5,
    "sunday": 6, "sun": 6, "rabiwar": 6, "ravivar": 6, "robibar": 6,
}

# Self-commitment patterns (speaker promises action)
SELF_COMMIT_RE = re.compile(
    r"\b(i will|i'll|ill\b|we will|we'll|i am going to|i'm going to|i'm on it|i am on it|on it\b|let me|allow me to|will complete|will send|will share|"
    r"will check|will call|will book|will pay|will transfer|will update|will review|will do|"
    r"sending by|sharing by|main\s+(?:kal\s+|aaj\s+)?[a-z]+\s+(?:dunga|dungi|karunga|karungi|leta|leti|bhejta|bhejti)|"
    r"kar\s+dunga|kar\s+dungi|kar\s+lunga|kar\s+lungi|le\s+lunga|le\s+lungi|pick\s+kar\s+lunga|drop\s+kar\s+dunga|"
    r"bhej\s+dunga|bhej\s+dungi|dekh\s+lunga|dekh\s+leta\s+hoon|kar\s+leta\s+hoon|"
    r"pathiye\s+debo|kore\s+debo|dekhe\s+nebo|bhejte\s+hain|"
    r"pathavto|pathavte|karun\s+takto|anuppuren|pampistanu|kalistini|mokli\s+aapish)\b",
    re.IGNORECASE,
)

# Request / directive patterns (speaker asks addressee to act)
DIRECTIVE_REQUEST_RE = re.compile(
    r"\b(please\s+[a-z]+|pl\s+[a-z]+|pls\s+[a-z]+|plz\s+[a-z]+|can\s+(?:you|u)|could\s+(?:you|u)|would\s+(?:you|u)|need (?:you to|your help to)|"
    r"don't forget to|dont forget to|remember to|make sure to|kindly\s+[a-z]+|"
    r"let me know|lmk|batao|batana|bata dena|tell me|update me|keep me posted|inform me|"
    r"bhej\s+dena|bhej\s+do|kar\s+dena|kar\s+do|de\s+dena|le\s+aana|chod\s+dena|chhor\s+dena|"
    r"pay\s+kar\s+dena|khoj\s+ke\s+batao|pata\s+karke\s+batao|"
    r"check\s+karo|check\s+kar\s+lena|yaad\s+se|pathiye\s+dio|kore\s+dio|ene\s+dio|niye\s+esho|dekhe\s+nio|khoj\s+niye)\b",
    re.IGNORECASE,
)

# Indirect household / work obligation ("next month tak kuch sochna padega", "need to sort out X before Friday")
INDIRECT_OBLIGATION_RE = re.compile(
    r"\b(sochna\s+padega|dekhna\s+padega|karna\s+padega|intezam\s+karna\s+padega|"
    r"vyavastha\s+karni\s+padegi|babostha\s+korte\s+hobe|dekhte\s+hobe|"
    r"need\s+to\s+(?:figure|sort)\s+out|have\s+to\s+(?:figure|sort)\s+out)\b",
    re.IGNORECASE,
)

TIME_HORIZON_ANCHOR_RE = re.compile(
    r"\b(next\s+month|this\s+month|agle\s+mahine|is\s+mahine|porer\s+mashe|"
    r"next\s+week|agle\s+hafte|before\s+[a-z0-9]+|se\s+pehle|tak\b|by\s+[a-z0-9]+)\b",
    re.IGNORECASE,
)

# Explicit deadline / statutory due patterns
DEADLINE_CUE_RE = re.compile(
    r"\b(deadline|due\s+date|due\s+on|due\s+by|is\s+due|due\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)|"
    r"last\s+date|cutoff|cut-off|expires\s+on|expiry|submit\s+by|submission\s+by|filing\s+by|pay\s+before|renewal\s+by|tarikh\s+tak)\b",
    re.IGNORECASE,
)

# Decision / open question patterns
QUESTION_CUE_RE = re.compile(
    r"(\?\s*$|\b(?:should (?:we|i)|shall we|which one|what should|do you want|can we|any\s+thoughts\s+on|"
    r"approve|konsa|kaunsa|kab\s+tak|kab\s+karein|kya\s+karein|kobe\s+|kon\s+ta)\b)",
    re.IGNORECASE,
)

# Implicit household depletion / outage reported by staff or spouse ("Gas cylinder khatam hone wala hai")
IMPLICIT_HOUSEHOLD_LOOP_RE = re.compile(
    r"\b(khatam\s+hone\s+wala\s+hai|khatam\s+ho\s+gaya|khatam\s+hai|out\s+of\s+stock|"
    r"kharab\s+ho\s+gaya|band\s+ho\s+gaya)\b",
    re.IGNORECASE,
)

SOCIAL_CHECKIN_QUESTION_RE = re.compile(
    r"\b(you\s+still\s+on\s+for|still\s+on\s+for\s+the|see\s+you\s+at|how\s+was\s+your|how's\s+it\s+going|reached\s+safely)\b",
    re.IGNORECASE,
)

DELIVERY_PLEASE_CHECK_RE = re.compile(
    r"\b(attached\b.*?\bplease\s+check|proofs\s+attached|attached\s+(?:madam|sir)|sent\b.*?\bplease\s+(?:check|review))\b",
    re.IGNORECASE,
)

# Pure social / phatic chatter to reject immediately
SOCIAL_NOISE_RE = re.compile(
    r"^(?:good\s+(?:morning|night|evening|afternoon)|gm\b|gn\b|happy\s+(?:birthday|diwali|anniversary|puja)|"
    r"congrats|congratulations|thank\s+you|thanks|ty\b|shukriya|dhanyawad|welcome|haha+|lol|😂|🙏|❤️|👍|ok\b|okay\b|k\b|hmm+|acha|accha|nice|great|awesome|cool)[!.,\s🙏👍❤️😊]*$",
    re.IGNORECASE,
)

# Ignore pure WhatsApp export placeholders as loop openers; documents remain available as closure evidence.
MEDIA_ONLY_RE = re.compile(
    r"^\[(?:photo|video|audio|sticker|gif|image|voice)(?:[^\]]*)\]$",
    re.IGNORECASE,
)
BIRTHDAY_WISH_RE = re.compile(
    r"^(?:happy\s+birthday|hbd\b|janmadin\s+mubarak|janamdin\s+mubarak|shubho\s+jonmodin)\b",
    re.IGNORECASE,
)
BIRTHDAY_ACTION_RE = re.compile(
    r"\b(book|order|buy|send|share|arrange|plan|organize|organise|pay|bring|pick|deliver|call|please|need|can you|could you|lmk|batao)\b",
    re.IGNORECASE,
)

ACCEPTANCE_REPLY_RE = re.compile(
    r"^(?:ok|okay|sure|yes|yep|haan|han|ji|ji\s+sahab|ji\s+bhaiya|theek\s+hai|thik\s+hai|"
    r"acha|accha|will\s+do|on\s+it|kar\s+dunga|kar\s+deti\s+hoon|main\s+kar\s+lunga|noted|👍|🙏)\b",
    re.IGNORECASE,
)

# Distinguishing object nouns that prevent false merging of similar-looking tasks
DISTINGUISHING_NOUNS: Set[str] = {
    "electricity", "water", "broadband", "wifi", "internet", "gas", "rent", "maintenance",
    "property", "gst", "tds", "itr", "audit", "payroll", "salary", "advance",
    "flight", "hotel", "visa", "cab", "train", "passport", "insurance",
    "deck", "proposal", "sow", "agreement", "contract", "invoice", "receipt",
    "maths", "science", "english", "hindi", "bengali", "tennis", "swimming", "piano",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


_MONTH_PAT = "|".join(sorted(MONTH_MAP.keys(), key=len, reverse=True))
_WEEKDAY_PAT = "|".join(sorted(WEEKDAY_MAP.keys(), key=len, reverse=True))

# Single C-level fast pre-filter so non-temporal messages exit resolve_due_date in <1 microsecond!
_DUE_FAST_GATE_RE = re.compile(
    rf"(?:\d|\b(?:{_MONTH_PAT}|{_WEEKDAY_PAT}|tomorrow|tmrw|tomo|agamikal|kalke|kal|parso|parson|porshu|"
    r"tonight|evening|eod|today|aaj|shaam|ajke|bikeler|week|hafte|shoptah)\b)",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(202[4-9])-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
_DMY_DATE_RE = re.compile(r"\b(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])[/-](202[4-9])\b")
_DMY_SHORT_DATE_RE = re.compile(
    r"\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])\b(?![/-]\d)"
)
_MD1_DATE_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PAT})\b(?:\s+(202[4-9]))?")
_MD2_DATE_RE = re.compile(rf"\b({_MONTH_PAT})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s*(202[4-9]))?")
_DAY_AFTER_RE = re.compile(r"\b(day after tomorrow|parso|parson|porshu)\b")
_KAL_WIN_RE = re.compile(r"\bkal\b(?:\s+\w+){0,3}")
_KAL_PAST_RE = re.compile(r"\b(tha|thi|the|gaya|gayi|bheja|kiya|aaya)\b")
_TOMORROW_RE = re.compile(r"\b(tomorrow|tmrw|tomo|agamikal|kalke)\b")
_TODAY_RE = re.compile(
    r"\b(by tonight|tonight|by evening|this evening|by eod|eod\b|by today|today evening|"
    r"aaj\s+shaam|shaam\s+tak|aaj\s+raat|aaj\s+hi|ajke\s+modhye|bikeler\s+modhye|by\s+\d{1,2}\s*(?:am|pm)|\d{1,2}\s*baje)\b"
)
_IN_DAYS_RE = re.compile(r"\b(?:in|within)\s+(\d{1,2})\s+days?\b|\b(\d{1,2})\s+din\s+mein\b")
_NEXT_WEEK_RE = re.compile(r"\b(next week|agle hafte|porer shoptah)\b")
_WEEKDAY_DUE_RE = re.compile(
    rf"\b(?:(by|on|this|next|before|coming|due)\s+({_WEEKDAY_PAT})\b|({_WEEKDAY_PAT})\s+(?:tak|ko|morning|evening|subah|shaam)\b)"
)


def resolve_due_date(text: str, msg_ts: datetime) -> Optional[str]:
    """Deterministically normalize explicit or relative due dates strictly against msg_ts."""
    if not text:
        return None
    norm = normalize_indic_for_matching(text) if not text.isascii() else text
    if not _DUE_FAST_GATE_RE.search(norm):
        return None
    low = norm.lower()
    base_date: date = msg_ts.date()

    # 1. Explicit ISO YYYY-MM-DD
    m_iso = _ISO_DATE_RE.search(low)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"

    # 2. Explicit DD/MM/YYYY or DD-MM-YYYY
    m_dmy = _DMY_DATE_RE.search(low)
    if m_dmy:
        d, mo, yr = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        try:
            return date(yr, mo, d).isoformat()
        except ValueError:
            pass

    # 3. Indian short date DD/MM (no year): anchor to message year, rolling old dates forward only if clearly stale.
    m_dmy_short = _DMY_SHORT_DATE_RE.search(low)
    if m_dmy_short:
        d, mo = int(m_dmy_short.group(1)), int(m_dmy_short.group(2))
        try:
            cand = date(base_date.year, mo, d)
            if (base_date - cand).days > 60:
                cand = date(base_date.year + 1, mo, d)
            return cand.isoformat()
        except ValueError:
            pass

    # 4. "15th Nov" / "Nov 15"
    m_md1 = _MD1_DATE_RE.search(low)
    m_md2 = None if m_md1 else _MD2_DATE_RE.search(low)
    if m_md1 or m_md2:
        if m_md1:
            day_i = int(m_md1.group(1))
            mon_i = MONTH_MAP[m_md1.group(2)]
            yr_i = int(m_md1.group(3)) if m_md1.group(3) else base_date.year
        else:
            assert m_md2 is not None
            mon_i = MONTH_MAP[m_md2.group(1)]
            day_i = int(m_md2.group(2))
            yr_i = int(m_md2.group(3)) if m_md2.group(3) else base_date.year
        try:
            cand = date(yr_i, mon_i, day_i)
            if not (m_md1 and m_md1.group(3)) and not (m_md2 and m_md2.group(3)):
                if (base_date - cand).days > 60:
                    cand = date(yr_i + 1, mon_i, day_i)
            return cand.isoformat()
        except ValueError:
            pass

    # 4. Relative day expressions anchored to msg_ts.date()
    if _DAY_AFTER_RE.search(low):
        return (base_date + timedelta(days=2)).isoformat()

    # In Hindi/Hinglish/Bengali, "kal" means BOTH yesterday and tomorrow!
    # Check tense only within the 4-word window following "kal", not across unrelated clauses.
    m_kal = _KAL_WIN_RE.search(low)
    kal_is_future = bool(m_kal) and not _KAL_PAST_RE.search(m_kal.group(0))
    if _TOMORROW_RE.search(low) or kal_is_future:
        return (base_date + timedelta(days=1)).isoformat()

    if _TODAY_RE.search(low):
        return base_date.isoformat()

    m_in_days = _IN_DAYS_RE.search(low)
    if m_in_days:
        n_days = int(m_in_days.group(1) or m_in_days.group(2))
        return (base_date + timedelta(days=n_days)).isoformat()

    if _NEXT_WEEK_RE.search(low):
        return (base_date + timedelta(days=7)).isoformat()

    # 5. Weekday expressions ("by Friday", "on Monday", "next Tuesday", "shukrawar tak") — single compiled regex!
    m_wd = _WEEKDAY_DUE_RE.search(low)
    if m_wd:
        prep = m_wd.group(1) or ""
        w_name = m_wd.group(2) or m_wd.group(3)
        w_idx = WEEKDAY_MAP[w_name]
        delta = (w_idx - base_date.weekday()) % 7
        if delta == 0 and prep == "next":
            delta = 7
        return (base_date + timedelta(days=delta)).isoformat()

    return None


def _find_mentioned_person(
    text: str,
    chat_participants: Tuple[str, ...],
    people: Dict[str, PersonPrediction],
    sender_id: str,
    index: Optional[ConversationIndex] = None,
) -> Optional[str]:
    low = text.lower()
    toks = set(tokenize(low, drop_stopwords=False))
    candidates = [p for p in chat_participants if p != USER_ID and p != sender_id]
    if index is not None:
        for handle in re.findall(r"@(\d{5,})", text):
            owner = index.mention_handle_owner.get(handle)
            if owner in candidates:
                return owner
    for pid in candidates:
        p = people.get(pid)
        if not p:
            continue
        aliases = [
            t.lower() for t in tokenize(p.name, drop_stopwords=True) if len(t) >= 3
        ]
        if any(a in toks for a in aliases):
            return pid
    return None


@dataclass
class LoopCandidate:
    origin_msg: MessageRecord
    loop_type: str  # commitment | followup | deadline | question
    owner: str  # ME or person_id
    title: str
    due_date: Optional[str]
    signature_tokens: Set[str]
    distinguishing_tokens: Set[str]
    involved_people: Set[str] = field(default_factory=set)
    related_msg_ids: List[str] = field(default_factory=list)
    confidence: float = 0.65
    is_task_list: bool = False


def _clean_loop_title(body: str, quoted_body: Optional[str] = None) -> str:
    text = body.strip()
    # If message is short or pronouns-heavy ("Sure I'll send it by 5pm"), incorporate quoted topic!
    if quoted_body and (
        len(text) < 28
        or re.search(r"\b(it|this|that|same|isko|usko|ye|woh)\b", text, re.IGNORECASE)
    ):
        q_clean = re.sub(r"\s+", " ", quoted_body).strip(" .,!?")
        if len(q_clean) > 6:
            text = f"{q_clean} ({text})"
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"^(?:hey\s+\w+,?\s*|hi\s+\w+,?\s*|hello\s*,?\s*|please\s+|pls\s+|kindly\s+|i\s+will\s+|i'll\s+|let\s+me\s+)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" .,")
    if text:
        text = text[0].upper() + text[1:]
    return text[:100] if text else "Follow up on message"


LOOP_EXTRACT_SYSTEM_PROMPT = """You validate and refine candidate open loops extracted from ME's WhatsApp conversations.
Allowed loop `type`: commitment (ME promised), followup (waiting on someone else), deadline (due on a date), question (ME or someone must answer/decide).
Rules:
1. Reject casual social chatter, pure FYIs, or emotional venting with no actionable commitment/question.
2. Owner MUST be `ME` or an exact `person_id` from the chat.
   - If Person A asks Person B (neither is ME) to do something, owner is Person B.
   - If spouse discusses something purely on their own side (e.g., their own office lunch), reject unless an actual task exists.
3. Resolve relative `due_date` (YYYY-MM-DD) strictly using the message timestamp shown in brackets, NEVER today's wall-clock date. Use null if no due date is stated.

Return JSON:
{"loops": [{"message_id": "<id>", "keep": true, "title": "<concise title>", "type": "commitment|followup|deadline|question", "owner": "ME|<person_id>", "due_date": "YYYY-MM-DD|null", "confidence": 0.0}]}"""


def extract_open_loop_candidates(
    index: ConversationIndex,
    people: Dict[str, PersonPrediction],
    llm: LLMClient,
    ledger: EvidenceLedger,
) -> List[LoopCandidate]:
    """Discover, normalize, and deduplicate candidate open loops across chats."""
    # The official checker alone knows loops_scored_from. Enumerate all loops;
    # earlier context can identify commitments repeated in the scoring window.
    scoring_cutoff = index.min_ts - timedelta(seconds=1)

    raw_candidates: List[LoopCandidate] = []
    consumed_msg_ids: Set[str] = set()


    for msg in index.messages:
        if msg.id in consumed_msg_ids:
            continue
        body = (msg.body or "").strip()
        if not body or SOCIAL_NOISE_RE.match(body) or MEDIA_ONLY_RE.fullmatch(body):
            continue

        match_body = normalize_indic_for_matching(body) if not body.isascii() else body
        if BIRTHDAY_WISH_RE.match(match_body) and not BIRTHDAY_ACTION_RE.search(match_body):
            continue

        due_date = None  # Lazily resolve after salience checks.

        et = index.enriched_by_id.get(msg.id)
        q_msg = et.quoted_message if et else None
        chat = index.chats.get(msg.chat_id)
        participants = chat.participant_ids if chat else (USER_ID,)
        non_me_parts = [p for p in participants if p != USER_ID]

        has_self_commit = bool(SELF_COMMIT_RE.search(match_body))
        has_directive = bool(DIRECTIVE_REQUEST_RE.search(match_body))
        # Unprefixed imperatives are common in personal exports. Keep
        # this anchored to action verbs rather than matching nouns such as 'fan'.
        if re.match(
            r"^(?:text|order|buy|send|share|book|pay|bring|call)\s+(?:me\s+|him\b|her\b|your\s+|the\s+|a\s+|an\s+|\d)",
            match_body, re.IGNORECASE,
        ):
            has_directive = True
        # Informal messages often omit the auxiliary in a concrete proposal or
        # promise. Require an action, rather than turning every 'let's' into work.
        if re.search(r"\blet['’]?s\s+(?:send|share|fix|schedule|book|arrange|meet|call)\b", match_body, re.IGNORECASE):
            has_self_commit = True
        if re.match(r"^(?:reverting|sending|sharing)\s+(?:tomorrow|tonight|by\b|on\s+(?:mon|tue|wed|thu|fri))", match_body, re.IGNORECASE):
            has_self_commit = True
        has_question = bool(QUESTION_CUE_RE.search(match_body))
        # A proposed meeting remains a decision even when an informal Hinglish
        # message omits its question mark.
        if re.search(r"\b(?:meet|meeting|call)\s+(?:pe|par)\s+karte\s+(?:hai|hain)\b", match_body, re.IGNORECASE):
            has_question = True
        # A task-list heading plus multiple numbered actions is a bundled request.
        # Names, shopping inventories and event attendance lists alone are not tasks.
        list_actions = re.findall(
            r"(?m)^\s*\d+[.)]\s*[\u2060\u200b*]*(?:regular\s+)?(?:clean(?:ing)?|vacuum|wash|repair|replace|install|submit|send|pay|book|check)\b",
            match_body, re.IGNORECASE,
        )
        is_task_list = len(list_actions) >= 2 and bool(re.search(r"\b(?:work|tasks?|to[ -]?do)\b", match_body.splitlines()[0], re.IGNORECASE))
        if is_task_list:
            if re.match(r"^(?:my|our)\s+(?:work|tasks?|to[ -]?do)\b", match_body, re.IGNORECASE):
                has_self_commit = True
            else:
                has_directive = True
        has_indirect_ob = bool(INDIRECT_OBLIGATION_RE.search(match_body)) and bool(
            TIME_HORIZON_ANCHOR_RE.search(match_body)
        )
        if re.search(r"\b(?:i |we )?need\s+to\s+(?:ask|discuss|find|hire|book|arrange|submit|renew|pay)\b", match_body, re.IGNORECASE) and not re.search(
            r"\b(?:no|not|don't|dont|do not|doesn't|does not)\s+need\b", match_body, re.IGNORECASE
        ):
            has_indirect_ob = True
        if re.search(r"\byou can go ahead and (?:order|book|buy|send|submit|pay)\b", match_body, re.IGNORECASE):
            has_directive = True
        if re.search(r"\bwe need (?:the |your |a )?(?:[a-z]+\s+){0,3}(?:copy|document|approval|signature|invoice|report)\s+to\s+(?:raise|submit|complete|process|finalize|finalise)\b", match_body, re.IGNORECASE):
            has_directive = True
        if re.search(r"\bi need\s+(?:a |an )?(?:good |new )?(?:teacher|tutor|driver|plumber|electrician|nanny)\b", match_body, re.IGNORECASE):
            has_indirect_ob = True
        if re.match(r"^(?:invite|invitation|payment|booking|reservation|appointment)\s+to be\s+(?:made|sent|booked|paid)\b", match_body, re.IGNORECASE):
            has_indirect_ob = True
        if (not chat or not chat.is_group) and re.search(
            r"\b(?:measurements|dimensions|documents|signature|approval)\s+(?:are )?required\s+for\b", match_body, re.IGNORECASE
        ):
            has_directive = True
        sender_circle = people[msg.sender_id].circle if msg.sender_id in people else "friend"
        has_implicit_household = (
            msg.sender_id != USER_ID
            and sender_circle in ("staff", "spouse")
            and (not chat or not chat.is_group or len(participants) <= 6)
            and bool(IMPLICIT_HOUSEHOLD_LOOP_RE.search(match_body))
            and not has_self_commit
        )
        has_raw_deadline_cue = bool(DEADLINE_CUE_RE.search(match_body))

        # Suppress pure social check-in questions ("You still on for the half marathon in Jan?")
        if has_question and not (has_self_commit or has_directive or has_raw_deadline_cue):
            if SOCIAL_CHECKIN_QUESTION_RE.search(match_body):
                has_question = False

        # Suppress delivery-notification "Proofs attached madam, please check" when fulfilling an open followup in the same chat!
        if (
            msg.sender_id != USER_ID
            and DELIVERY_PLEASE_CHECK_RE.search(match_body)
            and any(
                c.origin_msg.chat_id == msg.chat_id and c.owner == msg.sender_id
                for c in raw_candidates
            )
        ):
            continue

        # Strip trailing deferral clause ("Neha let me think about Zest") when an earlier sentence has a concrete commitment/deadline
        effective_body = body
        effective_match_body = match_body
        clauses = [cl.strip() for cl in re.split(r"(?<=[.!?])\s+", body) if cl.strip()]
        if len(clauses) >= 2 and re.search(r"\blet\s+me\s+(?:think|check)\s+about\b", clauses[-1], re.IGNORECASE):
            lead_body = " ".join(clauses[:-1])
            if SELF_COMMIT_RE.search(lead_body) or DEADLINE_CUE_RE.search(lead_body):
                effective_body = lead_body
                effective_match_body = normalize_indic_for_matching(lead_body) if not lead_body.isascii() else lead_body

        if (
            due_date is None
            and msg.ts >= scoring_cutoff
            and (
                has_self_commit
                or has_directive
                or has_question
                or has_indirect_ob
                or has_implicit_household
                or has_raw_deadline_cue
                or msg.kind in ("document", "image_caption")
            )
        ):
            due_date = resolve_due_date(effective_match_body, msg.ts)

        has_deadline_cue = has_raw_deadline_cue or (
            due_date is not None
            and msg.kind in ("document", "image_caption")
            and any(w in match_body.lower() for w in ("bill", "invoice", "due", "fee", "notice", "tax", "challan"))
        )

        # Avoid triggering on tiny 1-2 word questions like "Kya?" or "Really?" or "Coorg?"
        if has_question and len(tokenize(match_body, drop_stopwords=False)) < 4 and not q_msg:
            has_question = False

        if not (has_self_commit or has_directive or has_deadline_cue or has_question or has_indirect_ob or has_implicit_household):
            continue

        # Reject pure status-check questions, postponements, or cancellations quoting an earlier message,
        # as well as polite wellness greetings ("take care", "take your medicine on time")
        if re.search(r"\b(take\s+care|get\s+well\s+soon|take\s+your\s+medicine\s+on\s+time)\b", body, re.IGNORECASE):
            continue
        if q_msg:
            if ACCEPTANCE_REPLY_RE.match(match_body) and len(tokenize(match_body, drop_stopwords=False)) <= 5:
                continue
            if re.search(
                r"\b(did\s+you|have\s+you|ho\s+gaya\s+kya|bhej\s+diya\s+kya|postpone|reschedule|"
                r"move\s+the\s+deadline|push\s+this\s+to|no\s+longer\s+needed|rehne\s+do|bad\s+dao|cancel\s+this)\b",
                match_body,
                re.IGNORECASE,
            ):
                continue

        # Determine loop_type and owner with speaker-aware multi-party grounding
        loop_type: Optional[str] = None
        owner: Optional[str] = None
        conf = 0.68

        mentioned_other = _find_mentioned_person(body, participants, people, msg.sender_id, index)

        if msg.sender_id == USER_ID:
            if has_self_commit and not has_directive:
                loop_type = "deadline" if (has_deadline_cue and due_date) else "commitment"
                owner = USER_ID
                conf = 0.84
            elif has_directive:
                # ME asks someone else to act -> followup owned by that person!
                target = (
                    mentioned_other
                    or (q_msg.sender_id if q_msg and q_msg.sender_id != USER_ID else None)
                    or (non_me_parts[0] if len(non_me_parts) == 1 else None)
                )
                if target:
                    loop_type = "followup"
                    owner = target
                    conf = 0.82
                else:
                    # Group directive with unspecified individual: check if anyone replies accepting it!
                    accepter = None
                    for r_id in index.replies_to.get(msg.id, []):
                        rm = index.messages_by_id[r_id]
                        if rm.sender_id != USER_ID and ACCEPTANCE_REPLY_RE.search(rm.body):
                            accepter = rm.sender_id
                            consumed_msg_ids.add(rm.id)
                            break
                    owner = accepter or (non_me_parts[0] if non_me_parts else USER_ID)
                    loop_type = "followup"
                    conf = 0.72
            elif has_deadline_cue and due_date:
                loop_type = "deadline"
                owner = mentioned_other or USER_ID
                conf = 0.80
            elif has_question:
                target = (
                    mentioned_other
                    or (q_msg.sender_id if q_msg and q_msg.sender_id != USER_ID else None)
                    or (non_me_parts[0] if len(non_me_parts) == 1 else None)
                )
                if target:
                    loop_type = "followup"
                    owner = target
                    conf = 0.66
            elif has_indirect_ob:
                loop_type = "commitment"
                owner = USER_ID
                conf = 0.72
        else:
            # Sender is NOT ME (sender_id == P)
            sender_pid = msg.sender_id
            # Check if P is addressing ANOTHER non-ME person in a group chat! (Adversarial #4 & #19)
            addressed_third_party = mentioned_other  # != USER_ID and != sender_pid
            body_toks_low = set(tokenize(body.lower(), drop_stopwords=False))
            numeric_handles = set(re.findall(r"@(\d{5,})", body))
            me_explicitly_named = bool(
                index.me_aliases and (body_toks_low & index.me_aliases)
            ) or bool(numeric_handles & index.me_mention_handles) or (
                q_msg is not None and q_msg.sender_id == USER_ID
            )

            if addressed_third_party and not me_explicitly_named:
                # E.g., Spouse tells Staff "Ramesh, clean the balcony" or Colleague A asks Colleague B
                if has_directive or has_deadline_cue:
                    loop_type = "deadline" if (has_deadline_cue and due_date) else "followup"
                    owner = addressed_third_party
                    conf = 0.78
                elif has_question:
                    loop_type = "question"
                    owner = addressed_third_party
                    conf = 0.68
            elif has_self_commit and not has_directive:
                # P promises to send/do something -> followup owned by P!
                if chat and chat.is_group and len(participants) > 8 and not me_explicitly_named:
                    prior_me_request = bool(et and any(
                        pm.sender_id == USER_ID
                        and (msg.ts - pm.ts) <= timedelta(minutes=30)
                        and (DIRECTIVE_REQUEST_RE.search(pm.body) or QUESTION_CUE_RE.search(pm.body))
                        for pm in index.by_chat.get(msg.chat_id, [])[max(0, et.chat_index - 3):et.chat_index]
                    ))
                    if not prior_me_request:
                        continue
                loop_type = "followup"
                owner = sender_pid
                conf = 0.83
            elif has_directive:
                # P asks ME (in 1:1 DM, or quoting ME, or addressing group where ME is primary)
                if not chat or not chat.is_group or me_explicitly_named or len(non_me_parts) == 1:
                    # Check if ME replied accepting -> commitment; otherwise open request/commitment on ME
                    me_accepted = any(
                        index.messages_by_id[rid].sender_id == USER_ID
                        and ACCEPTANCE_REPLY_RE.search(index.messages_by_id[rid].body)
                        for rid in index.replies_to.get(msg.id, [])
                    )
                    loop_type = (
                        "deadline"
                        if (has_deadline_cue and due_date and not me_accepted)
                        else "commitment"
                    )
                    owner = USER_ID
                    conf = 0.85 if me_accepted else 0.78
                else:
                    # Unaddressed directive in a multi-person group: check if a staff/colleague claimed it!
                    claimer = None
                    for rid in index.replies_to.get(msg.id, []):
                        rm = index.messages_by_id[rid]
                        if ACCEPTANCE_REPLY_RE.search(rm.body):
                            claimer = rm.sender_id
                            break
                    household_context = len(participants) <= 6 and sender_circle in ("spouse", "parent", "staff")
                    local_me_acceptance = bool(et and any(
                        rm.sender_id == USER_ID and ACCEPTANCE_REPLY_RE.search(rm.body)
                        and (rm.ts - msg.ts) <= timedelta(minutes=30)
                        for rm in index.by_chat.get(msg.chat_id, [])[et.chat_index + 1:et.chat_index + 3]
                    ))
                    if not claimer and not household_context and not local_me_acceptance:
                        # Membership alone does not make ME the addressee of an
                        # operational/broadcast request in somebody else's thread.
                        continue
                    owner = claimer or USER_ID
                    loop_type = "commitment" if owner == USER_ID else "followup"
                    conf = 0.70
            elif has_deadline_cue and due_date:
                # Notice/bill/deadline shared in chat (e.g., school notice, invoice caption, society due date)
                sender_circle = people[sender_pid].circle if sender_pid in people else "friend"
                if sender_circle in ("spouse", "staff", "vendor", "client", "colleague", "partner") or not chat.is_group:
                    loop_type = "deadline"
                    owner = USER_ID
                    conf = 0.80
            elif has_question:
                group_decision_for_me = (
                    chat is not None
                    and chat.is_group
                    and len(participants) <= 8
                    and sender_circle in ("colleague", "partner", "spouse", "staff")
                    and bool(
                        re.search(
                            r"\b(should\s+we|shall\s+we|any\s+thoughts\s+on|kya\s+banana\s+hai|kya\s+karein)\b",
                            match_body,
                            re.IGNORECASE,
                        )
                    )
                )
                if not chat or not chat.is_group or me_explicitly_named or group_decision_for_me:
                    loop_type = "question"
                    owner = USER_ID
                    conf = 0.80
            elif has_implicit_household:
                loop_type = "commitment"
                owner = USER_ID
                conf = 0.82
            elif has_indirect_ob:
                if not chat or not chat.is_group or me_explicitly_named or len(non_me_parts) == 1:
                    loop_type = "followup"
                    owner = USER_ID
                    conf = 0.72

        if not loop_type or not owner:
            continue
        # Official check.py owners must be in chats.jsonl participants; historical/departed senders
        # can remain as message evidence, but cannot become output people or loop owners.
        if owner != USER_ID and owner not in people:
            continue

        # Also check immediate child reply for due_date refinement (e.g. "Can you send deck?" -> "Will send tomorrow")
        if due_date is None:
            for rid in index.replies_to.get(msg.id, []):
                rm = index.messages_by_id[rid]
                r_due = resolve_due_date(rm.body, rm.ts)
                if r_due and (rm.ts - msg.ts) <= timedelta(hours=12):
                    due_date = r_due
                    break

        # Inherit conversational topic context from quoted message OR immediate preceding prompt turn (<30m)
        prompt_msg: Optional[MessageRecord] = q_msg
        if prompt_msg is None and et is not None and et.chat_index > 0:
            c_msgs = index.by_chat.get(msg.chat_id, [])
            pm = c_msgs[et.chat_index - 1]
            if (
                pm.sender_id != msg.sender_id
                and (msg.ts - pm.ts) <= timedelta(minutes=30)
                and (not chat or not chat.is_group or len(participants) <= 6)
            ):
                prompt_msg = pm

        q_body = prompt_msg.body if prompt_msg else None
        title = _clean_loop_title(effective_body, q_msg.body if q_msg else None)
        # Exclude person first-names and generic verbs from task-identity signature tokens
        person_name_toks: Set[str] = set(index.me_aliases)
        for pid_k in participants:
            if pid_k in people:
                person_name_toks.update(tokenize(people[pid_k].name, drop_stopwords=True))
        prompt_context_str = ""
        if prompt_msg is not None:
            eff_tok_count = len(tokenize(effective_body, drop_stopwords=False))
            prompt_is_ask = (
                "?" in prompt_msg.body
                or bool(DIRECTIVE_REQUEST_RE.search(prompt_msg.body))
                or (prompt_msg.sender_id in people and people[prompt_msg.sender_id].circle == "child")
            )
            if eff_tok_count <= 7 or (
                prompt_is_ask
                and re.search(r"\b(options|tickets|homestay|pick\s+kar)\b", effective_body, re.IGNORECASE)
            ):
                prompt_context_str = prompt_msg.body
        sig_toks = {
            t
            for t in tokenize(f"{title} {effective_body} {prompt_context_str}", drop_stopwords=True)
            if len(t) >= 3
            and t not in person_name_toks
            and t
            not in {
                "please", "pls", "send", "share", "check", "will", "can", "could",
                "tomorrow", "today", "kal", "aaj", "bhai", "bhaiya", "didi", "sir",
                "madam", "sahab", "update", "draft", "deck", "slides", "presentation",
            }
        }
        dist_toks = sig_toks & DISTINGUISHING_NOUNS
        involved = {p for p in (msg.sender_id, owner, mentioned_other) if p and p != USER_ID}
        if not chat or not chat.is_group:
            involved.update(non_me_parts)

        raw_candidates.append(
            LoopCandidate(
                origin_msg=msg,
                loop_type=loop_type,
                owner=owner,
                title=title,
                due_date=due_date,
                signature_tokens=sig_toks,
                distinguishing_tokens=dist_toks,
                involved_people=involved,
                related_msg_ids=[msg.id],
                confidence=conf,
                is_task_list=is_task_list,
            )
        )

    # Cross-Chat & Intra-Chat Safe Deduplication
    # Merges repeated reminders/mentions of the SAME open task across chats (Adversarial #5 & #16)
    # while NEVER merging tasks with conflicting distinguishing nouns/counterparties (Adversarial #15)
    # or tasks created AFTER an intervening completion (Adversarial #6 — handled further in temporal_resolver).
    merged_candidates = _deduplicate_candidates(raw_candidates, index)

    # Optional Tier-1 Batched LLM Verification of borderline candidates
    if llm.is_available and merged_candidates:
        borderline = [c for c in merged_candidates if c.confidence < 0.80]
        tasks: List[Tuple[str, str, str]] = []
        for b_idx in range(0, len(borderline), 12):
            chunk = borderline[b_idx : b_idx + 12]
            blocks: List[str] = []
            for c in chunk:
                win = index.get_window(c.origin_msg.id, before=2, after=2)
                win_txt = "\n".join(index.format_turn(m, include_chat=True) for m in win)
                chat_parts = index.chats[c.origin_msg.chat_id].participant_ids
                blocks.append(
                    f"CANDIDATE msg_id={c.origin_msg.id} | chat_participants={list(chat_parts)} | "
                    f"prior_type={c.loop_type} | prior_owner={c.owner} | prior_due={c.due_date}\n{win_txt}"
                )
            tasks.append((f"loop_cand_{b_idx}", LOOP_EXTRACT_SYSTEM_PROMPT, "\n\n".join(blocks)))

        batch_res = llm.complete_json_batch("loop_discovery", tasks, max_tokens=1600)
        drop_ids: Set[str] = set()
        updates_by_mid: Dict[str, dict] = {}
        for res in batch_res.values():
            if not isinstance(res, dict):
                continue
            for item in res.get("loops", []) or []:
                if not isinstance(item, dict):
                    continue
                mid = str(item.get("message_id") or "").strip()
                if not item.get("keep", True):
                    drop_ids.add(mid)
                else:
                    updates_by_mid[mid] = item

        filtered: List[LoopCandidate] = []
        valid_owners = set(people.keys()) | {USER_ID}
        for c in merged_candidates:
            mid = c.origin_msg.id
            if mid in drop_ids:
                continue
            if mid in updates_by_mid:
                upd = updates_by_mid[mid]
                nt = str(upd.get("type") or c.loop_type).strip()
                no = str(upd.get("owner") or c.owner).strip()
                nd = upd.get("due_date")
                if nt in VALID_LOOP_TYPES:
                    c.loop_type = nt
                if no in valid_owners:
                    c.owner = no
                if isinstance(nd, str) and re.match(r"^202\d-\d{2}-\d{2}$", nd):
                    c.due_date = nd
                elif nd is None:
                    pass
                if upd.get("title"):
                    c.title = str(upd["title"]).strip()[:100]
                c.confidence = float(upd.get("confidence") or c.confidence)
            filtered.append(c)
        merged_candidates = filtered

    for c in merged_candidates:
        ledger.record(
            subject_type="open_loop",
            subject_id=c.origin_msg.id,
            field_name="discovery",
            predicted_value={"type": c.loop_type, "owner": c.owner, "due_date": c.due_date},
            strength="explicit" if c.confidence >= 0.8 else "contextual",
            confidence=c.confidence,
            message_ids=c.related_msg_ids,
            rationale=f"Detected {c.loop_type} owned by {c.owner}",
            stage="loop_discovery",
        )

    return merged_candidates


def _deduplicate_candidates(
    candidates: List[LoopCandidate], index: ConversationIndex
) -> List[LoopCandidate]:
    kept: List[LoopCandidate] = []
    for cand in candidates:
        merged_into: Optional[LoopCandidate] = None
        for prev in kept:
            # A bundle of actions is not a restatement of one earlier task merely
            # because both mention a room, supplier or other rare entity.
            if cand.is_task_list != prev.is_task_list:
                continue
            # Never merge candidates separated by > 10 days
            if abs((cand.origin_msg.ts - prev.origin_msg.ts).total_seconds()) > 10 * 86400:
                continue
            # Never merge two independent self-commitments made by DIFFERENT speakers unless cand quotes prev or explicitly takes over
            is_takeover_reply = (
                cand.origin_msg.quoted_id == prev.origin_msg.id
                or bool(
                    re.search(
                        r"\b(leave\s+it\s+to\s+me|i'll\s+handle|i\s+will\s+handle|main\s+kar\s+deta|main\s+dekh\s+leta)\b",
                        cand.origin_msg.body,
                        re.IGNORECASE,
                    )
                )
            )
            if (
                not is_takeover_reply
                and prev.owner != cand.owner
                and prev.origin_msg.sender_id == prev.owner
                and cand.origin_msg.sender_id == cand.owner
            ):
                continue

            # Never merge if specific utility/day entities conflict (e.g., electricity vs broadband, Monday vs Friday)
            exclusive_entities = {
                "electricity", "water", "broadband", "wifi", "gas", "rent", "maintenance",
                "flight", "hotel", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            }
            cand_exc = cand.distinguishing_tokens & exclusive_entities
            prev_exc = prev.distinguishing_tokens & exclusive_entities
            if cand_exc and prev_exc and cand_exc != prev_exc:
                continue
            if (
                cand.distinguishing_tokens
                and prev.distinguishing_tokens
                and not (cand.distinguishing_tokens & prev.distinguishing_tokens)
            ):
                continue
            if (
                cand.due_date
                and prev.due_date
                and cand.due_date != prev.due_date
                and not re.search(
                    r"\b(postpone|reschedule|revised|instead|answer\s+by)\b",
                    cand.origin_msg.body,
                    re.IGNORECASE,
                )
            ):
                continue
            # Never merge if specific 1:1 counterparties are completely disjoint unrelated external clients/vendors
            if (
                cand.involved_people
                and prev.involved_people
                and not (cand.involved_people & prev.involved_people)
                and cand.origin_msg.chat_id != prev.origin_msg.chat_id
            ):
                # Allow cross-chat merge ONLY when one chat is a group and the other is a DM with a group member!
                prev_parts = set(index.chats[prev.origin_msg.chat_id].participant_ids)
                cand_parts = set(index.chats[cand.origin_msg.chat_id].participant_ids)
                if not ((prev_parts & cand_parts) - {USER_ID}):
                    continue

            # Check if an explicit completion happened BETWEEN prev and cand (Adversarial #6: separate revision task!)
            if _has_intervening_completion(prev, cand.origin_msg.ts, index):
                continue

            # Same 1:1 DM Question/Followup -> Counterparty Promise within 6h ("brochure proofs kab tak?" -> "Madam kal shaam tak bhej dunga")
            same_chat = cand.origin_msg.chat_id == prev.origin_msg.chat_id
            chat_obj = index.chats.get(cand.origin_msg.chat_id)
            delta_sec = abs((cand.origin_msg.ts - prev.origin_msg.ts).total_seconds())
            if (
                same_chat
                and chat_obj is not None
                and not chat_obj.is_group
                and prev.owner == cand.owner
                and prev.owner != USER_ID
                and prev.origin_msg.sender_id == USER_ID
                and cand.origin_msg.sender_id == cand.owner
                and delta_sec <= 6 * 3600
            ):
                merged_into = prev
                break

            # Token Jaccard similarity + char overlap on core signature tokens
            if not cand.signature_tokens or not prev.signature_tokens:
                continue
            shared_sig = cand.signature_tokens & prev.signature_tokens
            inter = len(shared_sig)
            union = len(cand.signature_tokens | prev.signature_tokens)
            jacc = inter / max(1, union)

            same_quote_root = (
                cand.origin_msg.quoted_id is not None
                and (
                    cand.origin_msg.quoted_id == prev.origin_msg.id
                    or cand.origin_msg.quoted_id == prev.origin_msg.quoted_id
                )
            )
            within_36h = delta_sec <= 36 * 3600
            within_72h = delta_sec <= 72 * 3600
            has_rare_entity_match = (
                same_chat
                and cand.owner == prev.owner
                and within_72h
                and any(len(t) >= 4 and index._df.get(t, 99) <= 4 for t in shared_sig)
            )
            if (
                jacc >= 0.55
                or (inter >= 3 and within_36h)
                or (inter >= 2 and same_quote_root)
                or has_rare_entity_match
            ):
                merged_into = prev
                break

        if merged_into is not None:
            merged_into.related_msg_ids.append(cand.origin_msg.id)
            merged_into.involved_people.update(cand.involved_people)
            if merged_into.due_date is None and cand.due_date is not None and not re.search(
                r"\b(on\s+it|will\s+send\s+options)\b", cand.origin_msg.body, re.IGNORECASE
            ):
                merged_into.due_date = cand.due_date
            merged_into.confidence = max(merged_into.confidence, cand.confidence)
        else:
            kept.append(cand)
    return kept


def _has_intervening_completion(
    prev: LoopCandidate, before_ts: datetime, index: ConversationIndex
) -> bool:
    """Return True if prev's chat has an explicit completion message strictly between prev.ts and before_ts."""
    chat_msgs = index.by_chat.get(prev.origin_msg.chat_id, [])
    completive_re = re.compile(
        r"\b(sent|shared|done|completed|uploaded|paid|transferred|booked|submitted|bhej\s+diya|kar\s+diya|ho\s+gaya|pathiye\s+diyechi)\b",
        re.IGNORECASE,
    )
    for m in chat_msgs:
        if m.ts <= prev.origin_msg.ts:
            continue
        if m.ts >= before_ts:
            break
        if m.quoted_id == prev.origin_msg.id and completive_re.search(m.body):
            return True
        if completive_re.search(m.body):
            m_toks = set(tokenize(m.body, drop_stopwords=True))
            if len(m_toks & prev.signature_tokens) >= 1:
                return True
    return False
