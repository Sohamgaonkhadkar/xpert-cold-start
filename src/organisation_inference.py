"""Joint Organisation discovery, Person-Organisation link inference, and Chat ownership resolution."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .conversation_index import ConversationIndex, tokenize
from .llm_client import LLMClient
from .models import (
    USER_ID,
    VALID_ORG_RELATIONS,
    VALID_ORG_ROLES,
    VALID_ORG_SIDES,
    VALID_ORG_TYPES,
    ChatPrediction,
    ChatRecord,
    EvidenceLedger,
    OrgLink,
    OrganisationPrediction,
    PersonPrediction,
)
from .relationship_inference import BRIDGE_GROUP_RE, extract_contact_tag

SOCIETY_KEYWORDS: Set[str] = {
    "society", "residents", "rwa", "apartment", "apartments", "tower", "towers",
    "enclave", "condo", "heights", "greens", "maintenance", "committee", "flat",
}

HOLIDAY_HOME_KEYWORDS: Set[str] = {
    "villa", "cottage", "farmhouse", "holiday home", "vacation home", "beach house", "estate",
    "homestay", "alibaug", "goa house", "hill house", "caretaker",
}

WORKSPACE_KEYWORDS: Set[str] = {
    "workspace", "coworking", "wework", "awfis", "indiqube", "workafella",
    "91springboard", "regus", "innov8", "hotdesk",
}

HOME_GROUP_KEYWORDS: Set[str] = {
    "home", "ghar", "house", "household", "kitchen", "groceries", "bari",
}

SCHOOL_KEYWORDS: Set[str] = {
    "school", "academy", "international", "vidyalaya", "high", "montessori",
    "nursery", "kindergarten", "pta",
}

CLASS_GROUP_KEYWORDS: Set[str] = {
    "class", "grade", "section", "std", "parents", "bus route", "tuition",
}

WORK_TEAM_MARKERS: Set[str] = {
    "team", "core", "ops", "operations", "eng", "engineering", "product",
    "tech", "founders", "internal", "leadership",
}


def _slugify(name: str, prefix: str = "org") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{prefix}_{slug[:28]}" if slug else f"{prefix}_1"


def _infer_org_archetype_from_name(
    name: str,
) -> Tuple[str, str, str]:
    """Return (type, side, default_role) for a seeded or discovered organisation name."""
    low = name.lower()
    toks = set(tokenize(low, drop_stopwords=False))
    if toks & SOCIETY_KEYWORDS:
        return ("society", "home", "resident")
    if any(k in low for k in HOLIDAY_HOME_KEYWORDS):
        return ("vacation home", "home", "owner")
    if toks & WORKSPACE_KEYWORDS:
        return ("workspace", "work", "member")
    if toks & SCHOOL_KEYWORDS:
        return ("school", "home", "parent")
    if toks & HOME_GROUP_KEYWORDS or low in ("home", "our home", "household"):
        return ("residence", "home", "owner")
    return ("company", "work", "owner")


ORG_CHAT_SYSTEM_PROMPT = """You map group chats in ME's WhatsApp to either one of ME's organisations OR a specific person.
Critical rules:
1. Every chat ties to EXACTLY ONE owner: `{"owner_type": "organisation", "owner_id": "<org_id>"}` OR `{"owner_type": "person", "owner_id": "<person_id>"}`.
2. A client's company is NEVER an organisation of ME: a group chat dedicated to a single client belongs to `person` (<client_person_id>) or ME's delivery company if multiple clients aren't separated, preferring `person` (<client_person_id>) when the chat is named after the client/project with that client.
3. A child's class parents' group or child's tutor group belongs to `person` (<child_person_id>) unless a multi-chat school organisation is explicitly listed.
4. A friend's birthday group or care group for a specific parent belongs to `person` (<that_person_id>).

Return JSON:
{"chat_assignments": [{"chat_id": "<id>", "owner_type": "organisation|person", "owner_id": "<id>"}]}"""


def infer_organisations_and_chats(
    index: ConversationIndex,
    people: Dict[str, PersonPrediction],
    llm: LLMClient,
    ledger: EvidenceLedger,
) -> Tuple[List[OrganisationPrediction], List[ChatPrediction]]:
    """Jointly construct ME's organisations, attach every chat, and populate person org links."""
    orgs: Dict[str, OrganisationPrediction] = {}
    org_keywords: Dict[str, Set[str]] = {}

    # 1. Anchor organisations from seed.json
    for idx, s_org in enumerate(index.bundle.seed.organisations):
        oid = s_org.id or _slugify(s_org.name, prefix="org")
        if oid in orgs:
            oid = f"{oid}_{idx + 1}"
        inferred_type, inferred_side, inferred_role = _infer_org_archetype_from_name(
            s_org.name
        )
        otype = s_org.type or inferred_type
        oside = s_org.side or inferred_side
        orole = s_org.role or inferred_role
        orgs[oid] = OrganisationPrediction(
            id=oid,
            name=s_org.name,
            type=otype,
            side=oside,
            role=orole,
            source="seed",
        )
        kws = {
            t for t in tokenize(s_org.name, drop_stopwords=True) if len(t) >= 3
        }
        org_keywords[oid] = kws
        ledger.record(
            subject_type="organisation",
            subject_id=oid,
            field_name="organisation",
            predicted_value=s_org.name,
            strength="seed",
            confidence=1.0,
            rationale="Seeded organisation",
            stage="org_seed",
        )

    # Helper lookups by circle
    by_circle: Dict[str, List[str]] = defaultdict(list)
    for pid, p in people.items():
        by_circle[p.circle].append(pid)

    # 2. Discover unseeded user organisations from group chats & contact tags
    # Guard: NEVER promote a `client` or `vendor` tag into a user organisation!
    work_orgs = [oid for oid, o in orgs.items() if o.side == "work"]
    has_home_org = any(o.type == "residence" for o in orgs.values())
    has_society_org = any(o.type == "society" for o in orgs.values())
    has_holiday_org = any(o.type == "vacation home" for o in orgs.values())
    has_workspace_org = any(o.type == "workspace" for o in orgs.values())

    # Discover additional work company if unseeded colleague/partner cluster shares a recurring tag/prefix
    tag_supporters: Dict[str, List[str]] = defaultdict(list)
    for pid, p in people.items():
        if p.circle in ("colleague", "partner"):
            tag = extract_contact_tag(p.name)
            if tag:
                norm_tag = tag.strip()
                # Check if already covered by a seeded org
                tag_toks = set(tokenize(norm_tag, drop_stopwords=True))
                if not any(tag_toks & kws for kws in org_keywords.values()):
                    tag_supporters[norm_tag].append(pid)

    for tag_name, members in sorted(tag_supporters.items()):
        if len(members) >= 2:
            oid = _slugify(tag_name, prefix="org")
            if oid not in orgs:
                has_partner = any(people[m].circle == "partner" for m in members)
                orgs[oid] = OrganisationPrediction(
                    id=oid,
                    name=tag_name,
                    type="company",
                    side="work",
                    role="partner" if has_partner else "owner",
                    source="inferred",
                )
                org_keywords[oid] = {
                    t for t in tokenize(tag_name, drop_stopwords=True) if len(t) >= 2
                }
                work_orgs.append(oid)

    # Discover Home, Housing Society, Holiday Home from group structure & household staff
    group_chats = [c for c in index.chats.values() if c.is_group]

    # Check Home ecosystem: home ops group OR >=1 household staff/spouse, plus Workspace & unseeded Work groups
    home_group_candidates: List[str] = []
    society_group_candidates: List[str] = []
    holiday_group_candidates: List[str] = []
    workspace_group_candidates: List[str] = []
    unseeded_work_group_candidates: List[ChatRecord] = []

    for gc in group_chats:
        low_name = gc.name.lower()
        gtoks = set(tokenize(low_name, drop_stopwords=False))
        part_circles = Counter(
            people[p].circle for p in gc.participant_ids if p in people
        )
        # Skip if already matching a seeded/tag-discovered org name
        if any(gtoks & kws for kws in org_keywords.values() if kws):
            continue

        if any(k in low_name for k in HOLIDAY_HOME_KEYWORDS):
            holiday_group_candidates.append(gc.chat_id)
        elif gtoks & SOCIETY_KEYWORDS:
            society_group_candidates.append(gc.chat_id)
        elif gtoks & WORKSPACE_KEYWORDS:
            workspace_group_candidates.append(gc.chat_id)
        elif (gtoks & HOME_GROUP_KEYWORDS) or (
            part_circles["staff"] >= 1
            and (part_circles["spouse"] >= 1 or part_circles["parent"] >= 1 or len(gc.participant_ids) <= 6)
            and part_circles["colleague"] == 0
            and part_circles["client"] == 0
        ):
            home_group_candidates.append(gc.chat_id)
        elif (
            (gtoks & WORK_TEAM_MARKERS)
            and not BRIDGE_GROUP_RE.search(gc.name)
            and part_circles["client"] == 0
            and (part_circles["colleague"] + part_circles["partner"] >= 1)
        ):
            unseeded_work_group_candidates.append(gc)

    # Identify household staff vs office staff
    household_staff: List[str] = []
    office_staff: List[str] = []
    for spid in by_circle.get("staff", []):
        s_msgs = " ".join(m.body.lower() for m in index.person_messages.get(spid, [])[:20])
        if dm_cid := index.dm_chat_by_person.get(spid):
            s_msgs += " " + " ".join(m.body.lower() for m in index.by_chat.get(dm_cid, [])[:20])
        s_name = people[spid].name.lower()
        if any(w in s_name or w in s_msgs for w in ("office", "pantry", "reception", "desk", "cabin")) and not people[spid].supports_people:
            office_staff.append(spid)
        else:
            household_staff.append(spid)

    if not has_home_org and (home_group_candidates or household_staff):
        oid = "org_home"
        orgs[oid] = OrganisationPrediction(
            id=oid,
            name="Home",
            type="residence",
            side="home",
            role="owner",
            source="inferred",
        )
        org_keywords[oid] = {"home", "ghar", "house", "household", "kitchen"}

    if not has_society_org and society_group_candidates:
        first_soc_name = index.chats[society_group_candidates[0]].name
        oid = _slugify(first_soc_name, prefix="org_soc")
        orgs[oid] = OrganisationPrediction(
            id=oid,
            name=first_soc_name,
            type="society",
            side="home",
            role="resident",
            source="inferred",
        )
        org_keywords[oid] = {
            t for t in tokenize(first_soc_name, drop_stopwords=True) if len(t) >= 3
        } | {"society", "rwa", "maintenance", "tower"}

    if not has_holiday_org and holiday_group_candidates:
        first_hol_name = index.chats[holiday_group_candidates[0]].name
        oid = _slugify(first_hol_name, prefix="org_hol")
        orgs[oid] = OrganisationPrediction(
            id=oid,
            name=first_hol_name,
            type="vacation home",
            side="home",
            role="owner",
            source="inferred",
        )
        org_keywords[oid] = {
            t for t in tokenize(first_hol_name, drop_stopwords=True) if len(t) >= 3
        }

    if not has_workspace_org and workspace_group_candidates:
        first_ws_name = index.chats[workspace_group_candidates[0]].name
        oid = _slugify(first_ws_name, prefix="org_ws")
        orgs[oid] = OrganisationPrediction(
            id=oid,
            name=first_ws_name,
            type="workspace",
            side="work",
            role="member",
            source="inferred",
        )
        org_keywords[oid] = {
            t for t in tokenize(first_ws_name, drop_stopwords=True) if len(t) >= 3
        }

    # Spawn distinct unseeded work companies from internal team/ops groups ("Zephyr Core Team", "Kite Advisory Ops")
    for wgc in unseeded_work_group_candidates:
        gtoks_clean = [
            t for t in tokenize(wgc.name, drop_stopwords=True)
            if t.lower() not in WORK_TEAM_MARKERS and len(t) >= 3
        ]
        if not gtoks_clean or any(set(gtoks_clean) & kws for kws in org_keywords.values()):
            continue
        brand_words = [
            w for w in wgc.name.split()
            if re.sub(r"[^a-z0-9]", "", w.lower()) not in WORK_TEAM_MARKERS
        ]
        brand_name = " ".join(brand_words).strip(" -–—|") or wgc.name
        oid = _slugify(brand_name, prefix="org")
        if oid not in orgs:
            has_partner_in_grp = any(
                p in people and people[p].circle == "partner" for p in wgc.participant_ids
            )
            orgs[oid] = OrganisationPrediction(
                id=oid,
                name=brand_name,
                type="company",
                side="work",
                role="partner" if has_partner_in_grp else "owner",
                source="inferred",
            )
            org_keywords[oid] = set(gtoks_clean)
            work_orgs.append(oid)

    # Ensure at least one work company exists if ME has colleagues or partners
    work_orgs = [oid for oid, o in orgs.items() if o.side == "work"]
    if not work_orgs and (by_circle.get("colleague") or by_circle.get("partner")):
        # Derive name from most common internal work group name if possible
        work_gc_names = [
            gc.name
            for gc in group_chats
            if sum(1 for p in gc.participant_ids if p in people and people[p].circle in ("colleague", "partner")) >= 2
            and sum(1 for p in gc.participant_ids if p in people and people[p].circle == "client") == 0
        ]
        inferred_co_name = work_gc_names[0].split("-")[0].strip() if work_gc_names else "Work"
        oid = _slugify(inferred_co_name, prefix="org_work")
        has_partner = bool(by_circle.get("partner"))
        orgs[oid] = OrganisationPrediction(
            id=oid,
            name=inferred_co_name,
            type="company",
            side="work",
            role="partner" if has_partner else "owner",
            source="inferred",
        )
        org_keywords[oid] = {
            t for t in tokenize(inferred_co_name, drop_stopwords=True) if len(t) >= 3
        }
        work_orgs.append(oid)

    # Update company role to `partner` if ME has `partner` circle co-founders in that org
    for oid in work_orgs:
        kws = org_keywords.get(oid, set())
        for part_pid in by_circle.get("partner", []):
            ptoks = set(tokenize(people[part_pid].name, drop_stopwords=True))
            if (kws and (ptoks & kws)) or len(work_orgs) == 1:
                if orgs[oid].source != "seed" or not any(
                    so.name == orgs[oid].name and so.role for so in index.bundle.seed.organisations
                ):
                    orgs[oid].role = "partner"

    # 3. Resolve Chat Ownership (`person` vs `organisation`) for EVERY chat
    chat_predictions: Dict[str, ChatPrediction] = {}
    ambiguous_group_chats: List[ChatRecord] = []

    for cid, chat in sorted(index.chats.items()):
        non_me = [p for p in chat.participant_ids if p != USER_ID]

        # Rule A: 1-to-1 chats tie to that person, UNLESS the chat is an explicit institutional desk/office of one of ME's organisations
        if not chat.is_group and len(non_me) == 1:
            target_pid = non_me[0]
            dm_toks = set(tokenize(chat.name.lower(), drop_stopwords=False))
            inst_org_id: Optional[str] = None
            if dm_toks & {"desk", "reception", "admin", "helpdesk", "frontdesk", "office"}:
                for oid, kws in org_keywords.items():
                    if kws and (dm_toks & kws):
                        inst_org_id = oid
                        break
                # Spawn an unseeded school/society/workspace org when an institutional desk DM explicitly names one
                if inst_org_id is None and (dm_toks & (SCHOOL_KEYWORDS | SOCIETY_KEYWORDS | WORKSPACE_KEYWORDS)):
                    inst_name = re.sub(
                        r"\b(desk|reception|admin|helpdesk|frontdesk|office)\b",
                        "",
                        chat.name,
                        flags=re.IGNORECASE,
                    ).strip(" -–:")
                    if inst_name:
                        if dm_toks & SCHOOL_KEYWORDS:
                            itype, irole, iside = "school", "parent", "life"
                        elif dm_toks & SOCIETY_KEYWORDS:
                            itype, irole, iside = "society", "resident", "life"
                        else:
                            itype, irole, iside = "workspace", "member", "work"
                        inst_org_id = _slugify(inst_name, prefix=f"org_{itype.split()[0]}")
                        if inst_org_id not in orgs:
                            orgs[inst_org_id] = OrganisationPrediction(
                                id=inst_org_id,
                                name=inst_name,
                                type=itype,
                                side=iside,
                                role=irole,
                                source="inferred",
                            )
                            org_keywords[inst_org_id] = {
                                t for t in tokenize(inst_name, drop_stopwords=True) if len(t) >= 3
                            }
            if inst_org_id:
                chat_predictions[cid] = ChatPrediction(
                    chat_id=cid,
                    owner_type="organisation",
                    owner_id=inst_org_id,
                    confidence=0.88,
                    evidence=[m.id for m in index.by_chat.get(cid, [])[:1]],
                    source="inferred",
                )
            else:
                chat_predictions[cid] = ChatPrediction(
                    chat_id=cid,
                    owner_type="person",
                    owner_id=target_pid,
                    confidence=1.0,
                    evidence=[m.id for m in index.by_chat.get(cid, [])[:1]],
                    source="inferred",
                )
            continue

        low_cname = chat.name.lower()
        ctoks = set(tokenize(low_cname, drop_stopwords=False))
        body_preview = " ".join(m.body.lower() for m in index.by_chat.get(cid, [])[:15])
        btoks = set(tokenize(body_preview, drop_stopwords=False))

        part_circles = Counter(people[p].circle for p in non_me if p in people)

        # Rule B: Child's class parents' group / child's activity or tutor group -> belongs to the child!
        # Guard: NEVER hijack a household ("Ghar"/"Home") or extended-family group just because the child is a participant!
        children = by_circle.get("child", [])
        matched_child: Optional[str] = None
        is_child_centric_group = bool(
            ctoks & (CLASS_GROUP_KEYWORDS | SCHOOL_KEYWORDS | {"tutor", "coaching", "activity", "swimming", "tennis", "piano"})
        ) and not bool(ctoks & (HOME_GROUP_KEYWORDS | {"family", "parivar", "pariwar", "cousins"}))
        if children and is_child_centric_group and not (
            (ctoks & SCHOOL_KEYWORDS) and any(o.type == "school" for o in orgs.values())
        ):
            best_ch = children[0]
            best_hits = -1
            for ch in children:
                ch_aliases = [
                    t for t in tokenize(people[ch].name, drop_stopwords=False) if len(t) >= 3
                ]
                hits = sum(2 for a in ch_aliases if a in low_cname) + sum(
                    1 for a in ch_aliases if a in body_preview
                )
                if ch in non_me:
                    hits += 4
                if hits > best_hits:
                    best_hits = hits
                    best_ch = ch
            matched_child = best_ch

        if matched_child is not None:
            chat_predictions[cid] = ChatPrediction(
                chat_id=cid,
                owner_type="person",
                owner_id=matched_child,
                confidence=0.90,
                evidence=[m.id for m in index.by_chat.get(cid, [])[:2]],
            )
            continue

        # Rule C: Dedicated external client group (`client` present, e.g. "ClientCo x MyOrg") ->
        # tie to `client` person FIRST before internal work org keyword match!
        clients_in_chat = [p for p in non_me if p in people and people[p].circle == "client"]
        if len(clients_in_chat) == 1:
            cl_pid = clients_in_chat[0]
            cl_tag = extract_contact_tag(people[cl_pid].name)
            if (cl_tag and cl_tag.lower() in low_cname) or "pilot" in ctoks or "client" in ctoks or "<> " in low_cname or " x " in low_cname:
                chat_predictions[cid] = ChatPrediction(
                    chat_id=cid,
                    owner_type="person",
                    owner_id=cl_pid,
                    confidence=0.90,
                    evidence=[m.id for m in index.by_chat.get(cid, [])[:2]],
                )
                continue

        # Rule D: Explicit match to a known user organisation by name keywords or archetype
        best_org_id: Optional[str] = None
        best_org_score = 0.0
        for oid, org in orgs.items():
            score = 0.0
            kws = org_keywords.get(oid, set())
            overlap = len(ctoks & kws)
            if overlap > 0:
                score += 4.5 * overlap
            if any( k in btoks for k in kws if len(k) >= 4 ):
                score += 1.5
            if org.type == "society" and (ctoks & SOCIETY_KEYWORDS):
                score += 5.0
            if org.type == "vacation home" and any(k in low_cname for k in HOLIDAY_HOME_KEYWORDS):
                score += 5.5
            if org.type == "workspace" and (ctoks & WORKSPACE_KEYWORDS):
                score += 5.5
            if org.type == "residence" and (
                (ctoks & HOME_GROUP_KEYWORDS)
                or (
                    part_circles["staff"] >= 1
                    and part_circles["colleague"] == 0
                    and part_circles["client"] == 0
                    and not (ctoks & {"parivar", "utsav", "cousins", "college", "squad", "friends", "batch"})
                )
            ):
                score += 4.2
            if org.side == "work" and part_circles["colleague"] + part_circles["partner"] >= 1 and part_circles["client"] == 0:
                # Score shared colleague tags
                org_member_overlap = sum(
                    1
                    for p in non_me
                    if p in people
                    and people[p].circle in ("colleague", "partner")
                    and (set(tokenize(people[p].name, drop_stopwords=True)) & kws)
                )
                score += 1.5 + 1.8 * org_member_overlap
            if score > best_org_score:
                best_org_score = score
                best_org_id = oid

        if len(clients_in_chat) == 1 and best_org_score < 4.5:
            chat_predictions[cid] = ChatPrediction(
                chat_id=cid,
                owner_type="person",
                owner_id=clients_in_chat[0],
                confidence=0.84,
                evidence=[m.id for m in index.by_chat.get(cid, [])[:2]],
            )
            continue

        if best_org_id is not None and best_org_score >= 3.5:
            chat_predictions[cid] = ChatPrediction(
                chat_id=cid,
                owner_type="organisation",
                owner_id=best_org_id,
                confidence=min(0.95, 0.55 + 0.08 * best_org_score),
                evidence=[m.id for m in index.by_chat.get(cid, [])[:2]],
            )
            continue

        # Rule E: Person-centric family/friend/vendor group (e.g., parent care group, friend trip/birthday group, spouse + vendor renovation group)
        home_org_id = next((oid for oid, o in orgs.items() if o.type == "residence"), None)
        if part_circles["vendor"] >= 1 and part_circles["spouse"] >= 1 and home_org_id:
            chat_predictions[cid] = ChatPrediction(
                chat_id=cid,
                owner_type="organisation",
                owner_id=home_org_id,
                confidence=0.76,
                evidence=[m.id for m in index.by_chat.get(cid, [])[:2]],
            )
            continue

        if work_orgs and (part_circles["colleague"] + part_circles["partner"] >= 2) and part_circles["client"] == 0:
            chat_predictions[cid] = ChatPrediction(
                chat_id=cid,
                owner_type="organisation",
                owner_id=work_orgs[0],
                confidence=0.72,
                evidence=[m.id for m in index.by_chat.get(cid, [])[:2]],
            )
            ambiguous_group_chats.append(chat)
            continue

        # Fallback to primary participant in the group (named person, family elder in extended-family group, or most active speaker)
        named_person: Optional[str] = None
        for p in non_me:
            if p in people:
                p_toks = [t for t in tokenize(people[p].name, drop_stopwords=True) if len(t) >= 3]
                if any(t in low_cname for t in p_toks):
                    named_person = p
                    break
        if named_person is None and part_circles["parent"] >= 1 and (
            part_circles["relative"] + part_circles["sibling"] >= 2
        ):
            named_person = next(
                p for p in non_me if p in people and people[p].circle == "parent"
            )
        if named_person is None and non_me:
            msg_counts = Counter(
                m.sender_id for m in index.by_chat.get(cid, []) if m.sender_id != USER_ID
            )
            named_person = (
                msg_counts.most_common(1)[0][0] if msg_counts else non_me[0]
            )

        if named_person:
            chat_predictions[cid] = ChatPrediction(
                chat_id=cid,
                owner_type="person",
                owner_id=named_person,
                confidence=0.58,
                evidence=[m.id for m in index.by_chat.get(cid, [])[:2]],
            )
            ambiguous_group_chats.append(chat)
        elif home_org_id:
            chat_predictions[cid] = ChatPrediction(
                chat_id=cid,
                owner_type="organisation",
                owner_id=home_org_id,
                confidence=0.50,
            )

    # Optional LLM refinement on ambiguous group chats
    if llm.is_available and ambiguous_group_chats and orgs:
        org_desc = [
            f"{o.id}: name='{o.name}', type={o.type}, side={o.side}, role={o.role}"
            for o in orgs.values()
        ]
        chat_blocks: List[str] = []
        for gc in ambiguous_group_chats[:25]:
            parts_desc = [
                f"{p}({people[p].name}:{people[p].circle})"
                for p in gc.participant_ids
                if p != USER_ID and p in people
            ][:10]
            samples = [
                index.format_turn(m) for m in index.by_chat.get(gc.chat_id, [])[:5]
            ]
            chat_blocks.append(
                f"CHAT {gc.chat_id} | name=\"{gc.name}\" | participants={parts_desc}\n"
                + "\n".join(samples)
            )
        usr_prompt = (
            "ME's Organisations:\n"
            + "\n".join(org_desc)
            + "\n\nGroup Chats to Assign:\n"
            + "\n\n".join(chat_blocks)
        )
        res = llm.complete_json("chat_ownership", ORG_CHAT_SYSTEM_PROMPT, usr_prompt)
        if isinstance(res, dict):
            for item in res.get("chat_assignments", []) or []:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("chat_id") or "").strip()
                otype = str(item.get("owner_type") or "").strip()
                oid = str(item.get("owner_id") or "").strip()
                if cid in chat_predictions:
                    if otype == "organisation" and oid in orgs:
                        chat_predictions[cid].owner_type = "organisation"
                        chat_predictions[cid].owner_id = oid
                        chat_predictions[cid].confidence = 0.85
                    elif otype == "person" and oid in people:
                        chat_predictions[cid].owner_type = "person"
                        chat_predictions[cid].owner_id = oid
                        chat_predictions[cid].confidence = 0.85

    # 4. Populate Person <-> Organisation Links (`runs` vs `supports`) & Organisation `.chats` / `.people`
    for oid in orgs:
        orgs[oid].chats = []
        orgs[oid].people = []

    for cid, cpred in sorted(chat_predictions.items()):
        ledger.record(
            subject_type="chat",
            subject_id=cid,
            field_name="owner",
            predicted_value=f"{cpred.owner_type}:{cpred.owner_id}",
            strength="explicit" if cpred.confidence >= 0.85 else "contextual",
            confidence=cpred.confidence,
            message_ids=cpred.evidence,
            stage="chat_ownership",
        )
        if cpred.owner_type == "organisation" and cpred.owner_id in orgs:
            orgs[cpred.owner_id].chats.append(cid)

    home_org_id = next((oid for oid, o in orgs.items() if o.type == "residence"), None)
    holiday_org_ids = [oid for oid, o in orgs.items() if o.type == "vacation home"]
    society_org_ids = [oid for oid, o in orgs.items() if o.type == "society"]

    for pid, person in sorted(people.items()):
        links_map: Dict[str, str] = {}

        # A. Direct participation in organisation-owned chats
        for cid in index.person_chats.get(pid, set()):
            cpred = chat_predictions.get(cid)
            if cpred and cpred.owner_type == "organisation" and cpred.owner_id in orgs:
                org = orgs[cpred.owner_id]
                if person.circle in ("partner", "colleague") and org.side == "work":
                    links_map[org.id] = "runs"
                elif person.circle == "staff":
                    links_map[org.id] = "supports"
                elif person.circle == "vendor" and org.side == "home":
                    links_map[org.id] = "supports"

        # B. 1-to-1 contacts linked by role/tag even if not in a group chat
        ptoks = set(tokenize(person.name, drop_stopwords=True))
        if person.circle in ("colleague", "partner") and not any(
            orgs[oid].side == "work" for oid in links_map
        ):
            matched_wo = None
            for woid in work_orgs:
                if ptoks & org_keywords.get(woid, set()):
                    matched_wo = woid
                    break
            if matched_wo is None and len(work_orgs) == 1:
                matched_wo = work_orgs[0]
            if matched_wo:
                links_map[matched_wo] = "runs"

        if person.circle == "staff":
            # Match holiday-home or society staff first if already in a holiday/society group chat or name matches
            matched_hol = any(
                oid in holiday_org_ids or oid in society_org_ids for oid in links_map
            )
            for hoid in holiday_org_ids:
                if ptoks & org_keywords.get(hoid, set()):
                    links_map[hoid] = "supports"
                    matched_hol = True
            if not matched_hol:
                if pid in office_staff and work_orgs:
                    links_map.setdefault(work_orgs[0], "supports")
                elif pid in household_staff and home_org_id:
                    links_map.setdefault(home_org_id, "supports")
                elif society_org_ids and any(
                    cid in orgs[society_org_ids[0]].chats for cid in index.person_chats.get(pid, set())
                ):
                    links_map.setdefault(society_org_ids[0], "supports")

        person.organisation_links = [
            OrgLink(organisation_id=oid, relation=rel)
            for oid, rel in sorted(links_map.items())
            if rel in VALID_ORG_RELATIONS
        ]
        for link in person.organisation_links:
            if pid not in orgs[link.organisation_id].people:
                orgs[link.organisation_id].people.append(pid)

    # Prune empty unseeded organisations that ended up with 0 chats and 0 people, and refine partner role
    final_orgs: List[OrganisationPrediction] = []
    for oid, org in sorted(orgs.items()):
        org.chats = sorted(set(org.chats))
        org.people = sorted(set(org.people))
        if (
            org.side == "work"
            and org.type == "company"
            and any(p in people and people[p].circle == "partner" for p in org.people)
            and not any(so.name == org.name and so.role for so in index.bundle.seed.organisations)
        ):
            org.role = "partner"
        if org.source == "seed" or org.chats or org.people:
            final_orgs.append(org)

    return final_orgs, [chat_predictions[cid] for cid in sorted(chat_predictions)]
