"""Speaker-grounded Contact Dossier builder and multilingual Circle + Staff-Support inference."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .conversation_index import ConversationIndex, tokenize
from .indic_normalizer import normalize_indic_for_matching
from .llm_client import LLMClient
from .models import (
    USER_ID,
    VALID_CIRCLES,
    EvidenceLedger,
    MessageRecord,
    PersonPrediction,
)

# Multilingual lexical cue dictionaries (English, Romanized Hindi/Hinglish, Romanized Bengali, Devanagari, Bengali)
STAFF_ROLE_WORDS: Set[str] = {
    "driver", "chauffeur", "cook", "chef", "maharaj", "maid", "househelp",
    "cleaner", "sweeper", "gardener", "mali", "watchman", "guard", "security",
    "darwan", "nanny", "ayah", "babysitter", "caretaker", "attendant", "peon",
    "helper", "dhobi", "ironing", "bai", "kaamwali", " ड्राइवर", "रसोइया", "माली",
}

STAFF_STRONG_WORDS: Set[str] = {
    "salary", "pagar", "pagaar", "beton", "chutti", "chhuti", "advance",
    "sahab", "bhabhiji", "malik", "didimoni",
    "jhadu", "pocha", "bartan", "kacha", "safai", "porishkar", "rannar",
    "maid", "driver", "cook", "nanny", "ayah", "caretaker", "mali", "watchman",
}

STAFF_CHORE_WORDS: Set[str] = {
    "bazar", "sabzi", "tarkari", "groceries", "laundry", "ironing", "bagan", "pickup", "nasta",
    "nashta", "khana", "cylinder", "gas", "paani",
}

STAFF_PHRASES_RE = re.compile(
    r"\b(kya\s+banana|khana\s+banana|guests\s+ke\s+liye|gas\s+cylinder|cylinder\s+khatam|"
    r"paani\s+gir|ghar\s+chhod|pick\s+kar|chhod\s+diya|safai\s+ho|jhadu\s+pocha|"
    r"bartan\s+dhona|kaam\s+pe|chhutti\s+chahiye|theek\s+hai\s+didi|haan\s+bhaiya|ok\s+baby)\b",
    re.IGNORECASE,
)

VENDOR_ROLE_WORDS: Set[str] = {
    "plumber", "plumbing", "electrician", "electricals", "carpenter", "mistri", "painter", "mechanic",
    "technician", "pest", "laundry", "cleaners", "drycleaners", "chemist", "pharmacy", "medicals",
    "florist", "florists", "caterer", "caterers", "catering", "travels", "cab", "courier",
    "printer", "printers", "printing", "press", "graphics", "hardware",
    "agency", "vendor", "supplier", "contractor", "builders", "architect", "interior", "decor", "decorators",
    "chartered", "accountant", "advocate", "lawyer", "legal", "notary", "broker", "consultants", "associates",
    "realtor", "properties", "insurance", "store", "shop", "mart", "traders", "enterprises", "repair",
    "motors", "movers", "packers", "bakery", "bakers", "clinic", "diagnostics",
    "tutor", "coaching", "academy",
}

VENDOR_CONTEXT_WORDS: Set[str] = {
    "quotation", "quote", "estimate", "invoice", "bill", "gst", "payment",
    "delivery", "installation", "warranty", "service", "charges", "rate",
    "visiting", "inspection", "booking", "challan", "receipt", "retainer",
    "filing", "audit", "itr", "tds", "tuition", "fee", "qr",
}

CLIENT_CONTEXT_WORDS: Set[str] = {
    "proposal", "sow", "contract", "onboarding", "retainer", "deliverable",
    "milestone", "kickoff", "demo", "deck", "client", "engagement", "scope",
    "feedback", "signoff", "renewal", "poc", "pilot", "walkthrough", "pricing",
}

PARTNER_CONTEXT_WORDS: Set[str] = {
    "cofounder", "co-founder", "equity", "investor", "runway",
    "fundraise", "incorporation", "shareholder",
}

COLLEAGUE_CONTEXT_WORDS: Set[str] = {
    "standup", "sprint", "deploy", "staging", "prod", "merge", "repo",
    "jira", "roadmap", "candidate", "payroll", "okr", "kpi", "release",
}

SPOUSE_WORDS: Set[str] = {
    "anniversary", "hubby", "wifey", "jaan", "sweetheart", "darling",
    "love", "dinner tonight", "our kids", "tum kab", "ghar kab",
}

PARENT_CONTACT_WORDS: Set[str] = {
    "maa", "mom", "mummy", "amma", "ammaji", "aai", "mother", "mataji",
    "baba", "dad", "papa", "pappa", "appa", "nanna", "pitaji", "bapuji",
    "bapi", "father", "माँ", "मम्मी", "पापा", "बाबा", "মা", "বাবা",
}

INLAW_WORDS: Set[str] = {
    "mummyji", "papaji", "sasur", "sasurji", "saas", "sasu", "sasuma", "shashuri", "shoshur",
    "in-law", "inlaw", "jiju", "jijaji", "nanad", "devar", "jethani",
    "sala", "sali", "bhabhi", "boudi", "vahini", "attige", "bava", "mamiyar", "mamanaar", "attha", "mamagaru",
    "jamai", "damad", "bahurani",
    "mother-in-law", "father-in-law", "brother-in-law", "sister-in-law",
}

SIBLING_WORDS: Set[str] = {
    "sis", "sister", "brother", "behen", "bahan", "chota bhai", "bada bhai", "mota bhai",
    "bon", "dada", "didi", "tai", "anna", "akka", "thambi", "thangachi", "tammudu", "chelli",
    "chettan", "chechi", "rakhi", "bhai dooj",
}

RELATIVE_WORDS: Set[str] = {
    "chacha", "chachi", "chachaji", "mama", "mami", "mamaji", "mausi", "mausa", "masi", "mausiji", "mavshi",
    "bua", "fufa", "fufi", "fui", "fua", "tau", "kaka", "kaki", "kaku", "atya", "mashi", "mesho", "pishi", "pishemoshai",
    "periyappa", "chithappa", "chitti", "pinni", "babai", "peddananna", "thatha", "paati", "ajja", "ajji", "ammamma", "nanamma",
    "dadu", "dida", "nana", "nani", "cousin", "uncle", "aunty", "nephew", "niece",
}

CHILD_WORDS: Set[str] = {
    "homework", "tuition", "tutor", "exam", "school", "class", "ptm",
    "pocket money", "ipad", "cartoon", "coach", "practice", "beta", "betu", "bachcha",
}

FRIEND_WORDS: Set[str] = {
    "yaar", "bro", "dude", "machha", "bondhu", "party", "drinks", "beer",
    "pub", "match", "cricket", "football", "poker", "reunion", "trip", "weekend",
    "hangout", "chill",
}

CARE_ACTION_WORDS: Set[str] = {
    "pickup", "pick", "drop", "school", "bus", "tuition", "class", "tennis",
    "swimming", "football", "medicine", "dawai", "doctor", "clinic", "walk",
    "lunch", "tiffin", "milk", "feed", "bath", "homework", "park", "le aana",
    "chod dena", "chhor dena", "niye esho", "diye esho",
}

CONTACT_TAG_RE = re.compile(
    r"(?:[-–—|(/]\s*([A-Za-z0-9 &._]{2,25})\s*[)/]?$)|(?:@([A-Za-z0-9._-]{2,25})$)"
)
BRIDGE_GROUP_RE = re.compile(r"\b([a-z0-9]+)\s*(?:<>|x|×|\+|↔)\s*([a-z0-9]+)\b", re.IGNORECASE)


@dataclass
class ContactDossier:
    person_id: str
    display_name: str
    contact_tag: Optional[str]
    seeded_circle: Optional[str]
    dm_chat_id: Optional[str]
    dm_msg_count: int
    group_chat_ids: List[str]
    group_names: List[str]
    total_msgs_from_person: int
    total_msgs_from_me_in_dm: int
    seeded_co_circles: Counter[str] = field(default_factory=Counter)
    sample_turns: List[MessageRecord] = field(default_factory=list)
    heuristic_scores: Dict[str, float] = field(default_factory=dict)


def extract_contact_tag(display_name: str) -> Optional[str]:
    if not display_name:
        return None
    m = CONTACT_TAG_RE.search(display_name.strip())
    if m:
        tag = (m.group(1) or m.group(2) or "").strip()
        if "." in tag:
            tag = tag.split(".")[0]
        if len(tag) >= 2 and tag.lower() not in (
            "home", "work", "mobile", "cell", "new", "old", "india", "gmail", "yahoo", "outlook"
        ):
            return tag
    return None


def build_contact_dossiers(index: ConversationIndex) -> Dict[str, ContactDossier]:
    seed_people = index.bundle.seed.people
    dossiers: Dict[str, ContactDossier] = {}

    for pid in index.all_person_ids():
        disp = index.best_name(pid)
        tag = extract_contact_tag(disp)
        seeded = seed_people[pid].circle if pid in seed_people else None
        dm_cid = index.dm_chat_by_person.get(pid)
        dm_msgs = index.by_chat.get(dm_cid, []) if dm_cid else []
        me_in_dm = sum(1 for m in dm_msgs if m.sender_id == USER_ID)
        from_p = len(index.person_messages.get(pid, []))

        grp_ids = sorted(
            cid
            for cid in index.person_chats.get(pid, set())
            if cid in index.chats and index.chats[cid].is_group
        )
        grp_names = [index.chats[cid].name for cid in grp_ids]

        co_circles: Counter[str] = Counter()
        for cid in grp_ids:
            chat = index.chats[cid]
            # Weight smaller groups much higher than huge broadcast/housing groups
            size = max(2, len(chat.participant_ids))
            if size > 25:
                continue
            weight = 3 if size <= 6 else 1
            for other in chat.participant_ids:
                if other != pid and other in seed_people:
                    co_circles[seed_people[other].circle] += weight

        # Select up to 8 highest-signal sample turns (DM turns first, then group turns by/mentioning person)
        candidate_msgs: List[MessageRecord] = []
        if dm_msgs:
            candidate_msgs.extend(dm_msgs[:4])
            if len(dm_msgs) > 4:
                candidate_msgs.extend(dm_msgs[-4:])
        p_grp_msgs = [
            m for m in index.person_messages.get(pid, []) if m.chat_id != dm_cid
        ]
        candidate_msgs.extend(p_grp_msgs[:5])
        dedup_samples: List[MessageRecord] = []
        seen_m: Set[str] = set()
        for m in candidate_msgs:
            if m.id not in seen_m and m.body:
                seen_m.add(m.id)
                dedup_samples.append(m)
                if len(dedup_samples) >= 8:
                    break

        dossiers[pid] = ContactDossier(
            person_id=pid,
            display_name=disp,
            contact_tag=tag,
            seeded_circle=seeded,
            dm_chat_id=dm_cid,
            dm_msg_count=len(dm_msgs),
            group_chat_ids=grp_ids,
            group_names=grp_names,
            total_msgs_from_person=from_p,
            total_msgs_from_me_in_dm=me_in_dm,
            seeded_co_circles=co_circles,
            sample_turns=dedup_samples,
        )
    return dossiers


def _score_dossier_tier0(
    dossier: ContactDossier,
    index: ConversationIndex,
    seed_org_tokens: Set[str],
) -> Tuple[Dict[str, float], List[str]]:
    scores: Dict[str, float] = {c: 0.05 for c in VALID_CIRCLES}
    scores["friend"] = 0.25  # Default social acquaintance base prior over rare circles (spouse/child/in-law) when zero cues fire
    evidence_ids: List[str] = []

    raw_disp = dossier.display_name
    disp_norm = normalize_indic_for_matching(raw_disp) if not raw_disp.isascii() else raw_disp
    name_low = disp_norm.lower()
    name_tokens = set(tokenize(name_low, drop_stopwords=False))
    grp_text_low = " ".join(dossier.group_names).lower()
    grp_tokens = set(tokenize(grp_text_low, drop_stopwords=False))

    # Collect text from DM + person's group turns
    relevant_msgs: List[MessageRecord] = []
    if dossier.dm_chat_id:
        relevant_msgs.extend(index.by_chat.get(dossier.dm_chat_id, []))
    relevant_msgs.extend(index.person_messages.get(dossier.person_id, [])[:30])

    body_tokens: Counter[str] = Counter()
    full_text_chunks: List[str] = []
    for m in relevant_msgs:
        toks = tokenize(m.body, drop_stopwords=False)
        body_tokens.update(toks)
        m_norm = normalize_indic_for_matching(m.body) if not m.body.isascii() else m.body
        full_text_chunks.append(m_norm.lower())

    joined_body = " \n ".join(full_text_chunks)

    def _collect_ev(keywords: Set[str], limit: int = 3) -> None:
        for m in relevant_msgs:
            if len(evidence_ids) >= limit:
                break
            ml = m.body.lower()
            if any(kw in ml for kw in keywords) and m.id not in evidence_ids:
                evidence_ids.append(m.id)

    # 1. Phonebook contact name cues (very strong signal in WhatsApp)
    if name_tokens & STAFF_ROLE_WORDS or any(w in name_low for w in ("househelp", "watchman", "maid", "driver", "cook", "nanny", "ayah", "mali", "guard")):
        scores["staff"] += 6.0
        _collect_ev(STAFF_STRONG_WORDS | STAFF_CHORE_WORDS | STAFF_ROLE_WORDS)
    if name_tokens & VENDOR_ROLE_WORDS or re.search(r"\bca\b|\badv\.?\b|\bdr\.?\b", name_low):
        scores["vendor"] += 5.5
        _collect_ev(VENDOR_CONTEXT_WORDS | VENDOR_ROLE_WORDS)
    if name_tokens & PARENT_CONTACT_WORDS and not (name_tokens & INLAW_WORDS):
        scores["parent"] += 6.5
    if name_tokens & INLAW_WORDS or any(w in name_low for w in ("in-law", "mummyji", "papaji", "jiju", "bhabhi", "boudi")):
        scores["in-law"] += 6.5
    if name_tokens & RELATIVE_WORDS:
        scores["relative"] += 5.5
    if any(w in name_low for w in ("partner", "cofounder", "co-founder")):
        scores["partner"] += 5.0
    if "client" in name_tokens:
        scores["client"] += 5.5

    # Disambiguate Indian "<FirstName> didi / bhaiya / tai" domestic staff vs real sibling!
    # A real older sister "Didi" never calls ME "didi"/"madam"/"sir" back while asking what to cook or reporting chores.
    p_sent_text = " \n ".join(
        (normalize_indic_for_matching(m.body) if not m.body.isascii() else m.body).lower()
        for m in relevant_msgs
        if m.sender_id == dossier.person_id
    )
    has_staff_idioms = bool(STAFF_PHRASES_RE.search(p_sent_text))
    calls_me_boss_or_didi = bool(
        re.search(r"\b(didi|bhaiya|madam|ma'am|maam|sir|sahab|bhabhi|bhabhiji|malik|didimoni)\b", p_sent_text)
    )
    if has_staff_idioms and (
        calls_me_boss_or_didi or (name_tokens & {"didi", "bhaiya", "dada", "tai", "bai", "mausi"})
    ):
        scores["staff"] += 6.2
        _collect_ev(STAFF_STRONG_WORDS | STAFF_CHORE_WORDS)
    elif (name_tokens & SIBLING_WORDS) and scores["staff"] < 3.0 and not has_staff_idioms:
        scores["sibling"] += 5.2
        _collect_ev(SIBLING_WORDS)

    # Seed organisation match vs external tag in bridge groups ("ClientCo x MyOrg")
    external_bridge_client = False
    if dossier.contact_tag:
        tag_toks = set(tokenize(dossier.contact_tag, drop_stopwords=True))
        if (
            seed_org_tokens
            and tag_toks
            and not (tag_toks & seed_org_tokens)
            and not (tag_toks & (VENDOR_ROLE_WORDS | STAFF_ROLE_WORDS))
        ):
            for gname in dossier.group_names:
                bm = BRIDGE_GROUP_RE.search(gname)
                if bm:
                    sides = {bm.group(1).lower(), bm.group(2).lower()}
                    if sides & seed_org_tokens:
                        external_bridge_client = True
                        scores["client"] += 5.2
                        break

    if seed_org_tokens and (name_tokens & seed_org_tokens) and not external_bridge_client:
        scores["colleague"] += 3.8
        scores["partner"] += 1.8

    # 1b. Group-domain context for contacts saved by first name only (e.g. silent colleagues or extended family)
    work_group_cues = {"team", "core", "ops", "eng", "product", "tech", "studio", "labs", "internal"}
    family_group_cues = {"family", "parivar", "pariwar", "khandan", "cousins", "utsav", "bari"}
    in_work_group = False
    in_family_group = False
    for cid in dossier.group_chat_ids:
        chat = index.chats[cid]
        if len(chat.participant_ids) > 18:
            continue
        gtoks = set(tokenize(chat.name, drop_stopwords=True))
        if (seed_org_tokens and (gtoks & seed_org_tokens)) or (gtoks & work_group_cues):
            if not external_bridge_client:
                in_work_group = True
        if gtoks & family_group_cues:
            in_family_group = True

    if in_work_group and not in_family_group:
        scores["colleague"] += 3.2
    elif in_family_group and not in_work_group:
        scores["relative"] += 3.2

    # 2. Conversational lexical & speech-act signals (distinct-term gated so 8 weeks of small-talk never drifts!)
    distinct_strong_staff = {w for w in STAFF_STRONG_WORDS if w in body_tokens}
    distinct_chore_staff = {w for w in STAFF_CHORE_WORDS if w in body_tokens}
    if len(distinct_strong_staff) >= 2 or (
        len(distinct_strong_staff) >= 1 and len(distinct_chore_staff) >= 1
    ):
        scores["staff"] += min(
            5.2, 1.8 * len(distinct_strong_staff) + 0.9 * len(distinct_chore_staff)
        )
        if any(w in body_tokens for w in ("sahab", "bhabhiji", "madam", "malik", "dada", "didimoni")):
            scores["staff"] += 2.0
        _collect_ev(STAFF_STRONG_WORDS | STAFF_CHORE_WORDS)

    vendor_hits = sum(body_tokens[w] for w in VENDOR_CONTEXT_WORDS if w in body_tokens)
    if vendor_hits >= 2:
        # Internal work-group teammates discussing client invoices/GST are colleagues, NEVER external vendors!
        if in_work_group and not (name_tokens & VENDOR_ROLE_WORDS) and not external_bridge_client:
            scores["colleague"] += 1.6
        else:
            # Distinguish vendor (sends bill/quote TO ME) vs client (ME sends proposal/invoice TO them, or client asks ME for proposal/deck)
            p_vendor_msgs = sum(
                1
                for m in relevant_msgs
                if m.sender_id == dossier.person_id
                and any(w in m.body.lower() for w in ("invoice", "bill", "quotation", "charges", "estimate", "filing", "itr", "gst", "proofs", "copies"))
            )
            client_flow_msgs = sum(
                1
                for m in relevant_msgs
                if (
                    m.sender_id == USER_ID
                    and any(w in m.body.lower() for w in ("our proposal", "sending the proposal", "sow", "our team", "retainer invoice", "deliverable", "demo"))
                )
                or (
                    m.sender_id == dossier.person_id
                    and any(w in m.body.lower() for w in ("send the revised", "send the proposal", "questionnaire", "share the api", "benchmark deck"))
                )
            )
            if client_flow_msgs > p_vendor_msgs:
                scores["client"] += 4.2 + 0.8 * client_flow_msgs
                _collect_ev(CLIENT_CONTEXT_WORDS)
            else:
                scores["vendor"] += 3.5 + 0.8 * p_vendor_msgs
                _collect_ev(VENDOR_CONTEXT_WORDS)
    elif (
        not in_work_group
        and dossier.dm_chat_id
        and any(w in p_sent_text for w in ("proofs attached", "order ready", "copies", "visiting charges"))
        and calls_me_boss_or_didi
    ):
        scores["vendor"] += 4.6

    client_hits = sum(body_tokens[w] for w in CLIENT_CONTEXT_WORDS if w in body_tokens)
    if client_hits >= 1:
        me_or_inbound_client = sum(
            1
            for m in relevant_msgs
            if (
                m.sender_id == USER_ID
                and any(w in m.body.lower() for w in ("proposal", "sow", "deliverable", "deck", "demo", "our team", "onboarding", "retainer"))
            )
            or (
                m.sender_id == dossier.person_id
                and any(w in m.body.lower() for w in ("proposal", "questionnaire", "sow", "pilot", "procurement", "walkthrough", "pricing"))
            )
        )
        if me_or_inbound_client >= 1 and not (seed_org_tokens and (name_tokens & seed_org_tokens)):
            scores["client"] += 3.8 + 0.9 * me_or_inbound_client
            _collect_ev(CLIENT_CONTEXT_WORDS)
        else:
            scores["colleague"] += 1.5

    distinct_partner = {w for w in PARTNER_CONTEXT_WORDS if w in body_tokens}
    if len(distinct_partner) >= 2 or any(
        p in joined_body for p in ("cap table", "term sheet", "board meeting", "board deck")
    ):
        scores["partner"] += min(4.8, 1.8 * max(2, len(distinct_partner)))
        _collect_ev(PARTNER_CONTEXT_WORDS)

    distinct_colleague = {w for w in COLLEAGUE_CONTEXT_WORDS if w in body_tokens}
    if len(distinct_colleague) >= 2:
        scores["colleague"] += min(4.5, 1.4 * len(distinct_colleague))
        _collect_ev(COLLEAGUE_CONTEXT_WORDS)

    # Kinship & family cues
    if any(w in joined_body for w in SPOUSE_WORDS):
        scores["spouse"] += 3.2
        _collect_ev(SPOUSE_WORDS)
    if any(ch in raw_disp for ch in ("❤️", "♥", "💕", "😘", "💍")) and not in_work_group:
        scores["spouse"] += 5.2
    if (
        dossier.dm_chat_id
        and not in_work_group
        and scores["staff"] < 2.5
        and scores["vendor"] < 2.5
        and re.search(r"\b(kids'?\s+holidays|our\s+kids|kids'?\s+school|call\s+the\s+plumber|homestay)\b", joined_body)
    ):
        scores["spouse"] += 4.4
    if any(w in body_tokens for w in ("mummyji", "papaji", "jiju", "jijaji", "sasur", "saas", "boudi")):
        scores["in-law"] += 4.2
        _collect_ev(INLAW_WORDS)
    if any(w in body_tokens for w in ("chacha", "chachi", "mama", "mami", "mausi", "bua", "mashi", "pishi", "cousin", "dadu", "nani", "nana")):
        scores["relative"] += 3.8
        _collect_ev(RELATIVE_WORDS)

    # Direct parent address in 1:1 DM ("Maa ...", "Papa ...") vs sibling discussing parents ("Maa ko doctor...")
    if dossier.dm_chat_id:
        me_dm_texts = [
            m.body.lower()
            for m in index.by_chat.get(dossier.dm_chat_id, [])
            if m.sender_id == USER_ID
        ]
        if any(
            re.search(
                r"\b(pranam\s+(?:maa|ma|baba|papa|mummy|mom|dad)|charan\s+sparsh|love\s+you\s+(?:maa|mom|papa|dad)|take\s+your\s+medicine\s+maa|take\s+rest\s+papa)\b",
                t,
            )
            for t in me_dm_texts
        ):
            scores["parent"] += 4.5
        if any(re.search(r"\b(maa ko|papa ko|mummy ne|baba bol rahe|mom said|dad said|rakhi)\b", t) for t in me_dm_texts):
            scores["sibling"] += 3.6
            _collect_ev(SIBLING_WORDS | PARENT_CONTACT_WORDS)

    # Child signals (direct parental cues OR home-group child speaking about tuition/school/homework or cared for by staff)
    child_hits = sum(body_tokens[w] for w in CHILD_WORDS if w in body_tokens)
    if child_hits >= 2 and any(
        m.sender_id == USER_ID and any(k in m.body.lower() for k in ("beta", "betu", "finish your homework", "pack your school bag", "proud of you"))
        for m in relevant_msgs
    ):
        scores["child"] += 5.2
        _collect_ev(CHILD_WORDS)
    elif scores["staff"] < 3.0 and scores["vendor"] < 3.0 and not in_work_group:
        in_small_home_grp = any(
            len(index.chats[cid].participant_ids) <= 7
            and (set(tokenize(index.chats[cid].name.lower(), drop_stopwords=False)) & {"home", "ghar", "house", "family"})
            for cid in dossier.group_chat_ids
        )
        if in_small_home_grp:
            p_aliases = {t.lower() for t in tokenize(raw_disp, drop_stopwords=False) if len(t) >= 3}
            own_school_talk = any(
                w in p_sent_text for w in ("tuition", "homework", "school", "exam", "coaching")
            )
            cared_for_in_home_grp = False
            has_heart_emoji = any(ch in raw_disp for ch in ("❤️", "♥", "💕", "😘", "💍"))
            if not has_heart_emoji and scores["spouse"] < 3.5:
                for cid in dossier.group_chat_ids:
                    g_msgs = index.by_chat.get(cid, [])
                    for idx_gm, gm in enumerate(g_msgs):
                        glow = gm.body.lower()
                        prev_is_this_person = idx_gm > 0 and g_msgs[idx_gm - 1].sender_id == dossier.person_id
                        if gm.sender_id != dossier.person_id and (
                            (p_aliases and any(a in glow for a in p_aliases) and any(cw in glow for cw in ("ghar chhod", "chhod diya", "pick", "drop", "school", "tuition")))
                            or ("ok baby" in glow and prev_is_this_person)
                        ):
                            cared_for_in_home_grp = True
                            break
            if own_school_talk or cared_for_in_home_grp:
                scores["child"] += 5.6
                _collect_ev(CHILD_WORDS)

    # Shared-surname family ambiguity signal (never override company/vendor/staff tags; escalate generational ambiguity to LLM)
    if (
        index.me_surnames
        and (name_tokens & index.me_surnames)
        and not in_work_group
        and scores["staff"] < 2.5
        and scores["vendor"] < 2.5
        and scores["client"] < 2.5
    ):
        scores["sibling"] += 1.6
        scores["parent"] += 1.4
        scores["relative"] += 1.5

    # Friend signals
    friend_hits = sum(body_tokens[w] for w in FRIEND_WORDS if w in body_tokens)
    if friend_hits >= 2:
        scores["friend"] += min(4.5, 1.0 * friend_hits)
        _collect_ev(FRIEND_WORDS)
    if any(w in grp_tokens for w in ("boys", "gang", "college", "batch", "friends", "poker", "football", "cricket", "trip", "goa")):
        if scores["staff"] < 3.0 and scores["colleague"] < 3.0:
            scores["friend"] += 2.6

    # 3. Conditioned Bipartite Seed Co-Participation Propagation
    # Guarded: housing society & school-parents groups do NOT make co-members colleagues/relatives!
    is_society_only = bool(grp_tokens & {"society", "residents", "rwa", "apartment", "tower", "owners"}) and not dossier.dm_chat_id
    is_school_parents_only = bool(grp_tokens & {"parents", "grade", "class", "pta", "section"}) and not dossier.dm_chat_id

    if not is_society_only and not is_school_parents_only:
        if dossier.seeded_co_circles["colleague"] > 0 or dossier.seeded_co_circles["partner"] > 0:
            work_w = dossier.seeded_co_circles["colleague"] + dossier.seeded_co_circles["partner"]
            if scores["client"] < 3.5 and scores["staff"] < 3.5 and scores["vendor"] < 3.5:
                scores["colleague"] += min(3.2, 0.7 * work_w)
        if dossier.seeded_co_circles["friend"] > 0 and scores["colleague"] < 3.0:
            scores["friend"] += min(2.8, 0.8 * dossier.seeded_co_circles["friend"])
        fam_w = (
            dossier.seeded_co_circles["parent"]
            + dossier.seeded_co_circles["sibling"]
            + dossier.seeded_co_circles["relative"]
        )
        if fam_w > 0 and scores["staff"] < 3.5:
            scores["relative"] += min(2.4, 0.5 * fam_w)
            scores["sibling"] += min(1.8, 0.35 * fam_w)
    elif is_society_only:
        # Neighbours in housing society without strong work/family link default toward friend/vendor/staff based on role
        if scores["staff"] < 2.0 and scores["vendor"] < 2.0:
            scores["friend"] += 1.5
    elif is_school_parents_only:
        if scores["staff"] < 2.0 and scores["vendor"] < 2.0:
            scores["friend"] += 1.4

    # Fallback evidence IDs if none matched specific keywords
    if not evidence_ids and dossier.sample_turns:
        evidence_ids = [m.id for m in dossier.sample_turns[:2]]

    dossier.heuristic_scores = scores
    return scores, evidence_ids


def _infer_staff_supported_people(
    staff_pid: str,
    people_by_id: Dict[str, str],
    index: ConversationIndex,
) -> Tuple[List[str], List[str]]:
    """Identify specific family individuals (e.g., child, parent, spouse) looked after by a staff member."""
    family_targets: Dict[str, Set[str]] = {}
    for pid, circ in people_by_id.items():
        if pid == staff_pid:
            continue
        if circ in ("child", "parent", "spouse", "in-law"):
            nm = index.best_name(pid)
            aliases = {t.lower() for t in tokenize(nm, drop_stopwords=False) if len(t) >= 3}
            if aliases:
                family_targets[pid] = aliases

    if not family_targets:
        return [], []

    # Check DM with staff + household groups containing both staff and ME/spouse
    candidate_chats: Set[str] = set(index.person_chats.get(staff_pid, set()))
    supported_scores: Counter[str] = Counter()
    ev_ids: List[str] = []
    this_staff_aliases = {
        t.lower() for t in tokenize(index.best_name(staff_pid), drop_stopwords=False)
        if len(t) >= 3 and t.lower() not in STAFF_ROLE_WORDS
    }
    other_staff_aliases: Dict[str, Set[str]] = {
        opid: {
            t.lower() for t in tokenize(index.best_name(opid), drop_stopwords=False)
            if len(t) >= 3 and t.lower() not in STAFF_ROLE_WORDS
        }
        for opid, ocirc in people_by_id.items()
        if opid != staff_pid and ocirc == "staff"
    }
    child_pids = [pid for pid, circ in people_by_id.items() if circ == "child"]

    for cid in candidate_chats:
        chat = index.chats.get(cid)
        if chat and len(chat.participant_ids) > 10:
            continue
        c_msgs = index.by_chat.get(cid, [])
        is_group = bool(chat and chat.is_group)
        for idx_m, m in enumerate(c_msgs):
            sender_is_family_target = m.sender_id in family_targets and people_by_id.get(m.sender_id) != "spouse"
            if (
                m.sender_id not in (USER_ID, staff_pid)
                and people_by_id.get(m.sender_id) != "spouse"
                and not sender_is_family_target
            ):
                continue
            low = m.body.lower()
            toks = set(tokenize(low, drop_stopwords=False))
            # In multi-staff group chats, do not credit staff A when a message addresses staff B!
            if is_group and m.sender_id != staff_pid:
                names_this_staff = bool(this_staff_aliases and (toks & this_staff_aliases))
                names_other_staff = any(
                    opid in chat.participant_ids and bool(oal and (toks & oal))
                    for opid, oal in other_staff_aliases.items()
                )
                staff_replies_next = (
                    idx_m + 1 < len(c_msgs) and c_msgs[idx_m + 1].sender_id == staff_pid
                )
                if names_other_staff and not names_this_staff:
                    continue
                if not names_this_staff and not staff_replies_next:
                    continue
            has_care = bool(toks & CARE_ACTION_WORDS) or any(
                phrase in low
                for phrase in (
                    "le aana", "chod dena", "chhor dena", "chhod diya", "ghar chhod",
                    "pick kar", "pick up", "drop ", "niye esho",
                )
            )
            if not has_care:
                continue
            if sender_is_family_target:
                supported_scores[m.sender_id] += 1
                if m.id not in ev_ids:
                    ev_ids.append(m.id)
                continue
            matched_named = False
            for fam_pid, aliases in family_targets.items():
                # Do not treat spouse sending an instruction to staff about a child as staff looking after spouse
                if m.sender_id == fam_pid and people_by_id.get(fam_pid) == "spouse":
                    continue
                if toks & aliases:
                    supported_scores[fam_pid] += 1
                    matched_named = True
                    if m.id not in ev_ids:
                        ev_ids.append(m.id)
            # Collective child reference ("picked up the kids", "bachchon ko school chhod diya")
            if not matched_named and child_pids and (
                toks & {"kids", "children", "bachche", "bachchon", "babies"}
            ):
                for cpid in child_pids:
                    supported_scores[cpid] += 1
                if m.id not in ev_ids:
                    ev_ids.append(m.id)

    supported = sorted(pid for pid, cnt in supported_scores.items() if cnt >= 1)
    return supported, ev_ids[:4]


CIRCLE_SYSTEM_PROMPT = """You classify contacts in a user's (ME) WhatsApp history into exact relationship circles.
Allowed circles (choose EXACTLY ONE per person):
spouse, child, parent, in-law, sibling, relative, friend, colleague, client, vendor, partner, staff.

Critical rules:
1. Household workers (cook, driver, maid, nanny, watchman) AND office support cleaners/drivers are `staff`.
2. External service providers (CA, lawyer, plumber, travel agent, broker, repairman) billing ME are `vendor`.
3. External buyers/customers receiving proposals, deliverables, or retainers from ME's company are `client`.
4. Co-founders/co-owners running a company with ME are `partner`; employees/teammates in ME's company are `colleague`.
5. Sharing a housing society or school parents' group does NOT make someone a `colleague` or `relative`.
6. Also list `supports_people` (array of person_ids) ONLY when `circle == "staff"` and messages show they look after a specific individual (such as picking up a child or caring for an elderly parent).

Return JSON object:
{"predictions": [{"id": "<person_id>", "circle": "<valid_circle>", "supports_people": [], "confidence": 0.0, "evidence": ["<msg_id>"]}]}"""


def _discover_unseeded_work_tokens(
    index: ConversationIndex, dossiers: Dict[str, ContactDossier]
) -> Set[str]:
    """Self-discover ME's internal workplace brand tokens even when seed.organisations is empty (0% seed)."""
    work_markers = {
        "team", "core", "ops", "operations", "eng", "engineering", "product",
        "tech", "founders", "internal", "leadership",
    }
    generic_ignore = work_markers | {
        "advisory", "studio", "labs", "design", "group", "work", "official", "india", "pilot", "project",
    }
    discovered: Set[str] = set()
    for chat in index.chats.values():
        if not chat.is_group or len(chat.participant_ids) > 18:
            continue
        if BRIDGE_GROUP_RE.search(chat.name):
            continue
        ctoks = [t.lower() for t in tokenize(chat.name, drop_stopwords=True)]
        if set(ctoks) & work_markers:
            for t in ctoks:
                if len(t) >= 3 and t not in generic_ignore:
                    discovered.add(t)

    # Recurring @domain handle shared by >=2 contacts where at least one is in an internal work group
    at_domain_counts: Counter[str] = Counter()
    for dos in dossiers.values():
        if "@" in dos.display_name and dos.contact_tag:
            for t in tokenize(dos.contact_tag, drop_stopwords=True):
                if len(t) >= 3:
                    at_domain_counts[t.lower()] += 1
    for dom, cnt in at_domain_counts.items():
        if cnt >= 2:
            discovered.add(dom)
    return discovered


def infer_people_circles(
    index: ConversationIndex,
    llm: LLMClient,
    ledger: EvidenceLedger,
    batch_size: int = 18,
) -> Dict[str, PersonPrediction]:
    """Infer circle and staff person-support links for every non-ME person."""
    dossiers = build_contact_dossiers(index)
    seed_people = index.bundle.seed.people

    seed_org_tokens: Set[str] = _discover_unseeded_work_tokens(index, dossiers)
    for org in index.bundle.seed.organisations:
        for tok in tokenize(org.name, drop_stopwords=True):
            if len(tok) >= 3:
                seed_org_tokens.add(tok)

    predictions: Dict[str, PersonPrediction] = {}
    uncertain_dossiers: List[ContactDossier] = []

    # Step 1: Lock seed ground truth and run Tier-0 scoring on unseeded contacts
    for pid, dos in sorted(dossiers.items()):
        if pid in seed_people:
            sp = seed_people[pid]
            ev = [m.id for m in dos.sample_turns[:2]]
            pred = PersonPrediction(
                id=pid,
                name=dos.display_name,
                circle=sp.circle,
                supports_people=list(sp.supports_people),
                evidence=ev,
                confidence=1.0,
                source="seed",
            )
            predictions[pid] = pred
            ledger.record(
                subject_type="person",
                subject_id=pid,
                field_name="circle",
                predicted_value=sp.circle,
                strength="seed",
                confidence=1.0,
                message_ids=ev,
                rationale="Provided in seed.json",
                stage="relationship_seed",
            )
            continue

        scores, ev_ids = _score_dossier_tier0(dos, index, seed_org_tokens)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        best_circle, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        conf = min(0.95, 0.40 + 0.10 * margin)

        pred = PersonPrediction(
            id=pid,
            name=dos.display_name,
            circle=best_circle,
            evidence=ev_ids,
            confidence=round(conf, 3),
            source="inferred",
        )
        predictions[pid] = pred

        # Escalate ONLY genuinely ambiguous or cue-bearing contacts (avoids sending 250+ pure greeting
        # acquaintances in a 300-chat export when no competing non-friend circle or substantive turn exists!)
        has_competing_circle = second_score >= 0.45
        has_substantive_turn = any(len(m.body.split()) >= 10 for m in dos.sample_turns)
        has_domain_or_tag_signal = bool(
            best_score >= 0.50
            or has_substantive_turn
            or "@" in dos.display_name
            or " - " in dos.display_name
            or any(c != "friend" for c in dos.seeded_co_circles)
        )
        if margin < 3.2 and (has_competing_circle or has_domain_or_tag_signal):
            uncertain_dossiers.append(dos)
        else:
            ledger.record(
                subject_type="person",
                subject_id=pid,
                field_name="circle",
                predicted_value=best_circle,
                strength="explicit" if margin >= 4.5 else "contextual",
                confidence=conf,
                message_ids=ev_ids,
                rationale=f"Tier-0 contact/speech-act margin={margin:.2f}",
                stage="relationship_tier0",
            )

    # Step 2: Tier-1 Batched LLM Dossier Classification for uncertain contacts
    if llm.is_available and uncertain_dossiers:
        known_family_summary = [
            f"{p.id}({p.name}): {p.circle}"
            for p in predictions.values()
            if p.source == "seed" or p.confidence >= 0.80
        ][:25]
        seed_org_names = [o.name for o in index.bundle.seed.organisations]

        tasks: List[Tuple[str, str, str]] = []
        for b_idx in range(0, len(uncertain_dossiers), batch_size):
            chunk = uncertain_dossiers[b_idx : b_idx + batch_size]
            lines = [
                f"Seeded/Known Anchors: {', '.join(known_family_summary) or 'None'}",
                f"ME's Known Organisations: {', '.join(seed_org_names) or 'Unknown'}",
                "Classify each contact below:",
            ]
            for d in chunk:
                top_hints = sorted(
                    d.heuristic_scores.items(), key=lambda kv: -kv[1]
                )[:3]
                hint_str = ", ".join(f"{c}:{s:.1f}" for c, s in top_hints)
                snippets = [
                    index.format_turn(m, include_chat=True) for m in d.sample_turns[:5]
                ]
                lines.append(
                    f"\n--- PERSON {d.person_id} | saved_name=\"{d.display_name}\" | "
                    f"dm_msgs={d.dm_msg_count} | groups={d.group_names[:4]} | "
                    f"seed_neighbors={dict(d.seeded_co_circles)} | prior=[{hint_str}] ---\n"
                    + ("\n".join(snippets) if snippets else "(No direct messages; group participant only)")
                )
            tasks.append((f"batch_{b_idx}", CIRCLE_SYSTEM_PROMPT, "\n".join(lines)))

        batch_outputs = llm.complete_json_batch(
            "relationship_inference", tasks, max_tokens=1800
        )
        valid_msg_ids = set(index.messages_by_id.keys())
        for res in batch_outputs.values():
            items = res.get("predictions", []) if isinstance(res, dict) else []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("id") or "").strip()
                circ = str(item.get("circle") or "").strip().lower()
                if pid not in predictions or pid in seed_people or circ not in VALID_CIRCLES:
                    continue
                llm_conf = float(item.get("confidence") or 0.78)
                ev = [
                    str(m)
                    for m in (item.get("evidence") or [])
                    if str(m) in valid_msg_ids
                ]
                if not ev:
                    ev = predictions[pid].evidence
                sup_p = [
                    str(sp_id)
                    for sp_id in (item.get("supports_people") or [])
                    if str(sp_id) in predictions and str(sp_id) != pid
                ]
                predictions[pid].circle = circ
                predictions[pid].confidence = round(max(0.5, min(0.98, llm_conf)), 3)
                predictions[pid].evidence = ev
                if circ == "staff" and sup_p:
                    predictions[pid].supports_people = sorted(set(sup_p))
                ledger.record(
                    subject_type="person",
                    subject_id=pid,
                    field_name="circle",
                    predicted_value=circ,
                    strength="explicit" if llm_conf >= 0.8 else "contextual",
                    confidence=llm_conf,
                    message_ids=ev,
                    rationale="Batched LLM Contact Dossier classification",
                    stage="relationship_llm",
                )
    else:
        for d in uncertain_dossiers:
            p = predictions[d.person_id]
            ledger.record(
                subject_type="person",
                subject_id=p.id,
                field_name="circle",
                predicted_value=p.circle,
                strength="contextual" if p.confidence >= 0.55 else "insufficient",
                confidence=p.confidence,
                message_ids=p.evidence,
                rationale="Tier-0 fallback (no LLM endpoint configured)",
                stage="relationship_tier0_fallback",
            )

    # Step 3: Global Consistency Constraints
    # Constraint A: At most ONE spouse. If multiple unseeded candidates have `spouse`, keep highest-confidence/highest-DM one.
    seeded_spouses = [p.id for p in predictions.values() if p.source == "seed" and p.circle == "spouse"]
    inferred_spouses = [
        p for p in predictions.values() if p.source != "seed" and p.circle == "spouse"
    ]
    if seeded_spouses and inferred_spouses:
        for p in inferred_spouses:
            p.circle = "relative"
    elif len(inferred_spouses) > 1:
        inferred_spouses.sort(
            key=lambda p: (p.confidence, dossiers[p.id].dm_msg_count), reverse=True
        )
        for p in inferred_spouses[1:]:
            p.circle = "friend"

    # Step 4: Deterministic + Dossier Staff `supports_people` enrichment
    circle_map = {pid: p.circle for pid, p in predictions.items()}
    for pid, pred in predictions.items():
        if pred.circle == "staff":
            det_sup, sup_ev = _infer_staff_supported_people(pid, circle_map, index)
            merged_sup = sorted(set(pred.supports_people) | set(det_sup))
            pred.supports_people = merged_sup
            for eid in sup_ev:
                if eid not in pred.evidence:
                    pred.evidence.append(eid)
            if merged_sup:
                ledger.record(
                    subject_type="person",
                    subject_id=pid,
                    field_name="supports_people",
                    predicted_value=merged_sup,
                    strength="explicit",
                    confidence=0.88,
                    message_ids=sup_ev,
                    rationale="Staff care/pickup/drop co-mentions with family members",
                    stage="staff_support_links",
                )
        else:
            pred.supports_people = []

    return predictions
