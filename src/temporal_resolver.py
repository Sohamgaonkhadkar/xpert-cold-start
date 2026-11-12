"""ARR (Anchor-Retrieve-Reconcile) Cross-Chat Temporal Resolution with Asymmetric Closure Guards."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .conversation_index import ConversationIndex, tokenize
from .indic_normalizer import is_pure_indic_bare_ack, normalize_indic_for_matching
from .llm_client import LLMClient
from .loop_extraction import LoopCandidate, resolve_due_date
from .models import (
    USER_ID,
    VALID_LOOP_STATUSES,
    EvidenceLedger,
    MessageRecord,
    OpenLoopPrediction,
    PersonPrediction,
)


def _asr_phonetic_key(tok: str) -> str:
    """Normalize common Indian-English / Hinglish Whisper ASR spelling variations (e.g. challan/chalan, zephyr/zefir)."""
    if len(tok) < 4 or not tok.isascii():
        return tok
    s = (
        tok.lower()
        .replace("ph", "f")
        .replace("chh", "ch")
        .replace("sh", "s")
        .replace("z", "s")
        .replace("v", "w")
    )
    dedup: List[str] = []
    for ch in s:
        if not dedup or dedup[-1] != ch:
            dedup.append(ch)
    s2 = "".join(dedup)
    if len(s2) >= 5:
        s2 = s2[0] + re.sub(r"[aeiou]", "", s2[1:])
    return s2


def _compute_token_overlap(
    msg_tokens: Sequence[str], sig_tokens: Set[str], allow_asr_fuzzy: bool = False
) -> int:
    m_set = set(msg_tokens)
    exact = m_set & sig_tokens
    if not allow_asr_fuzzy or len(exact) >= len(sig_tokens):
        return len(exact)
    rem_sig = sig_tokens - exact
    rem_msg = m_set - exact
    bigrams = {
        f"{msg_tokens[i]}{msg_tokens[i + 1]}" for i in range(len(msg_tokens) - 1)
    }
    compound_hits = rem_sig & bigrams
    rem_sig -= compound_hits
    if not rem_sig:
        return len(exact) + len(compound_hits)
    msg_phon = {_asr_phonetic_key(t) for t in rem_msg if len(t) >= 4}
    phon_hits = sum(
        1 for st in rem_sig if len(st) >= 4 and _asr_phonetic_key(st) in msg_phon
    )
    return len(exact) + len(compound_hits) + phon_hits

# Guard against negated, conditional, or future-tense "done/sent/paid" false positives!
NEGATED_OR_FUTURE_COMPLETION_RE = re.compile(
    r"\b(not\s+(?:yet\s+)?(?:done|sent|paid|shared|completed|booked|received|filed)|"
    r"haven't|havent|hasn't|hasnt|didn't|didnt|couldn't|couldnt|"
    r"(?:will|to|should|must|can|get|getting)\s+(?:be\s+)?(?:done|sent|paid|shared|completed|booked|filed)|"
    r"(?:if|once|when|after)\s+(?:it\s+is\s+)?(?:done|sent|paid|shared|completed)|"
    r"almost\s+done|nearly\s+done|half\s+done|"
    r"abhi\s+(?:tak\s+)?nahi|nahi\s+(?:hua|bheja|kiya|mila)|baaki\s+hai|pending|"
    r"ekhono\s+hoyni|pathaini|kal\s+kar\s+dunga|kal\s+bhej\s+dunga)\b",
    re.IGNORECASE,
)

# Explicit past-completive markers (English, Hinglish, Hindi, Bengali, Marathi, Tamil, Telugu, Kannada, Gujarati)
EXPLICIT_COMPLETION_RE = re.compile(
    r"\b(sent\s+(?:it|the|your|our|over|across|on|via|you|with)|have\s+sent|just\s+sent|"
    r"shared\s+(?:the|it|your|our|on|over|link|doc|deck|slides)|uploaded|submitted|completed|filed\s+(?:the|it|your)|"
    r"paid\s+(?:it|the|your|our|him|her|them|via|by|upi|\d)|payment\s+done|transferred|"
    r"booked(?:\s+(?:it|the|our|your|tickets|flight|hotel|slot))?|"
    r"proofs\s+attached|attached\s+(?:madam|sir|here|the|please)|"
    r"fixed\s+(?:it|the)|resolved|signed\s+and\s+sent|all\s+done|task\s+done|done\s+from\s+my\s+side|"
    r"bhej\s+diya|bhj\s+diya|kar\s+diya|kr\s+diya|de\s+diya|jama\s+kar\s+diya|le\s+li(?:\s+hai)?|le\s+liya|"
    r"chhod\s+diya|ghar\s+chhod\s+diya|pick\s+kar\s+liya|drop\s+kar\s+diya|"
    r"ho\s+gaya(?:\s+hai)?|hogaya|book\s+kar\s+diya|pay\s+kar\s+diya|"
    r"pathiye\s+diyechi|kore\s+diyechi|hoe\s+geche|hoye\s+geche|deye\s+diyechi|"
    r"pathavle|zale|anuppitten|mudinjathu|pampanu|ayipoyindi|kaliside|aaytu|mokli\s+didhu|thai\s+gayu)\b",
    re.IGNORECASE,
)

# Completion-check question ("Did you send it?", "Ho gaya kya?", "Done?")
COMPLETION_CHECK_QUESTION_RE = re.compile(
    r"(\b(?:did\s+you|have\s+you)\s+(?:send|share|pay|transfer|book|complete|finish|submit|upload|call|check)|"
    r"\b(?:is\s+it|was\s+it)\s+(?:done|sent|paid|booked|completed)\s*\?|"
    r"\b(?:ho\s+gaya\s+kya|hogaya\s+kya|bhej\s+diya\s+kya|kar\s+diya\s+kya|kiya\s+kya|pathiyecho\s+ki|hoye\s+geche\s+ki)\b|"
    r"^(?:done|sent|paid|booked|updated)\s*\?+$)",
    re.IGNORECASE,
)

# Affirmative answer to a completion-check question ("Ok", "Yes", "Haan", "Done")
AFFIRMATIVE_CONFIRM_RE = re.compile(
    r"^(?:yes|yep|yeah|yup|ok|okay|done|sent|paid|haan|han|haanji|ji\s+haan|ji\s+ho\s+gaya|ho\s+gaya|bhej\s+diya|hoye\s+geche|ha\s+pathiyechi)\b",
    re.IGNORECASE,
)

# Bare acknowledgement ("Ok", "Sure", "Theek hai", "👍") that does NOT mean completion when replying to a request!
BARE_ACK_RE = re.compile(
    r"^(?:ok|okay|k|kk|sure|noted|roger|will\s+do|on\s+it|haan|han|ji|ji\s+sahab|ji\s+bhaiya|"
    r"theek\s+hai|thik\s+hai|acha|accha|dekhta\s+hoon|dekh\s+leta\s+hoon|kar\s+dunga|👍|🙏|👌)[!.,\s👍🙏👌]*$",
    re.IGNORECASE,
)

# Explicit cancellation / drop markers
CANCELLATION_RE = re.compile(
    r"\b(cancel\s+(?:this|the\s+(?:entire\s+)?(?:task\s+)?list)|cancelled|canceled|no\s+longer\s+needed|not\s+needed\s+anymore|"
    r"drop\s+this|dropping\s+this|forget\s+it|never\s+mind|nevermind|ignore\s+previous|hold\s+off\s+indefinitely|"
    r"skip\s+this|don't\s+send|dont\s+send|don't\s+book|dont\s+book|"
    r"rehne\s+do|rehnedo|mat\s+karo|mat\s+bhejo|zaroorat\s+nahi|jarurat\s+nahi|cancel\s+kar\s+do|"
    r"bad\s+dao|ar\s+lagbe\s+na|pathate\s+hobe\s+na)\b",
    re.IGNORECASE,
)

# Postponement / deadline revision markers
POSTPONE_RE = re.compile(
    r"\b(postpone|postponed|reschedule|rescheduled|push\s+(?:it|this)\s+to|move\s+(?:it|this|the\s+deadline)\s+to|"
    r"extend(?:ed)?\s+(?:to|till|until)|instead\s+of\s+today|kal\s+nahi|aaj\s+nahi|shift\s+to)\b",
    re.IGNORECASE,
)

# Ownership takeover / delegation markers
TAKEOVER_SELF_RE = re.compile(
    r"\b(i'll\s+handle\s+(?:it|this)|i\s+will\s+handle\s+(?:it|this)|i'm\s+taking\s+this\s+over|"
    r"leave\s+it\s+to\s+me|i'll\s+take\s+care\s+of\s+(?:it|this)|main\s+kar\s+lunga|main\s+dekh\s+lunga|"
    r"main\s+handle\s+kar\s+leta\s+hoon|main\s+ja\s+raha\s+hoon|ami\s+dekhe\s+nichhi|ami\s+kore\s+nichhi)\b",
    re.IGNORECASE,
)

DELEGATE_DIRECTIVE_RE = re.compile(
    r"\b(you\s+handle\s+this|please\s+take\s+over|can\s+you\s+take\s+this\s+up|"
    r"tum\s+dekh\s+lo|aap\s+dekh\s+lo|tum\s+kar\s+do|tum\s+handle\s+kar\s+lo|tumi\s+dekhe\s+nao)\b",
    re.IGNORECASE,
)

# Requester receipt confirmation ("Got the report", "Received, thanks", "Mil gaya", "Peye gechi")
RECEIPT_CONFIRM_RE = re.compile(
    r"\b(received\s+(?:it|the|your)|got\s+(?:the|your)|payment\s+received|money\s+received|"
    r"mil\s+gaya|mil\s+gayi|paisa\s+mil\s+gaya|peye\s+gechi|peyechi)\b",
    re.IGNORECASE,
)

# Banking/payment rail colloquial completion ("Sorted from HDFC", "Done via GPay", "IMPS done")
PAYMENT_RAIL_DONE_RE = re.compile(
    r"\b(?:sorted|done|sent|transferred|cleared)\s+(?:from|via|on|through|by)\s+"
    r"(?:upi|gpay|phonepe|paytm|neft|imps|rtgs|hdfc|icici|sbi|axis|kotak|bank)\b|"
    r"\b(?:upi|gpay|phonepe|neft|imps)\s+(?:done|sent|kar\s+diya|kore\s+diyechi)\b",
    re.IGNORECASE,
)

PAYMENT_LOOP_WORDS: Set[str] = {
    "pay", "payment", "paid", "fee", "fees", "bill", "salary", "pagar", "pagaar",
    "beton", "invoice", "rent", "emi", "transfer", "upi", "cheque", "tuition", "tax", "gst",
}

# Short colloquial 1:1 DM completion ("All sorted!", "Taken care of!", "Done and dusted", "Sab set hai")
SHORT_DM_COMPLETION_RE = re.compile(
    r"^(?:all\s+sorted|sorted|taken\s+care\s+of|done\s+and\s+dusted|sab\s+set\s+hai|kaam\s+ho\s+gaya)[!.,\s👍🙏👌]*$",
    re.IGNORECASE,
)

# Reopen / failed completion markers
FAILURE_REOPEN_RE = re.compile(
    r"\b(didn't\s+receive|didnt\s+receive|not\s+received|attachment\s+missing|broken\s+link|"
    r"payment\s+failed|bounced|wrong\s+file|still\s+waiting|abhi\s+tak\s+nahi\s+aaya|nahi\s+mila)\b",
    re.IGNORECASE,
)

DEFER_ANSWER_RE = re.compile(
    r"\b(let\s+me\s+(?:check|think)|will\s+(?:check|send|share|book|do)|give\s+me\s+a\s+day|on\s+it\b|"
    r"tell\s+you\s+later|get\s+back\s+to\s+you|dekhta\s+hoon|soch\s+ke\s+batata|puch\s+ke\s+batata|pore\s+bolchi)\b",
    re.IGNORECASE,
)

PARTIAL_PROGRESS_GUARD_RE = re.compile(
    r"\b(\d+\s*(?:of|out\s+of|/)\s*\d+|part\s*1|slides?\s+\d+\s*[-–]\s*\d+|finishing\s+.*?\btomorrow\b|rest\s+tomorrow|baaki\s+kal)\b",
    re.IGNORECASE,
)


SLOW_VERIFY_SYSTEM_PROMPT = """You perform strict asymmetric temporal verification on a WhatsApp open loop.
Given an originating task message and its chronological cross-chat evidence trail:
1. NEVER mark `done` just because a due_date passed!
2. A bare "Ok", "Sure", "Haan", "Theek hai", or "👍" replying to a request is ACKNOWLEDGEMENT (`open`), NOT `done`!
   "Ok" / "Yes" / "Haan" ONLY means `done` if it directly answers a later completion-check question ("Did you send it?", "Ho gaya kya?") or states past completion ("Sent", "Bhej diya", "Paid").
3. If explicitly cancelled ("Cancel this", "Rehne do, zaroorat nahi"), status is `dropped` with `closure_message_id`.
4. If someone else explicitly took over responsibility, update `owner` to their ID (`ME` or `<person_id>`).
5. If deadline was postponed, keep `open` and return updated `due_date` (YYYY-MM-DD).

Return JSON:
{"status": "open|done|dropped", "closure_message_id": "<msg_id>|null", "owner": "ME|<person_id>", "due_date": "YYYY-MM-DD|null", "confidence": 0.0}"""


def _delivers_promised_reply(cand: LoopCandidate, message: MessageRecord, index: ConversationIndex) -> bool:
    """A substantive on-topic answer can fulfil a promise to get back to someone.

    This resolves the communication promise, not any underlying repair or decision.
    Restrict it to an adjacent owner response in a DM with an identifiable preceding
    information request. Unrelated chatter and another deferral do not suffice.
    """
    if cand.is_task_list or message.sender_id != cand.owner or message.chat_id != cand.origin_msg.chat_id:
        return False
    if message.quoted_id and message.quoted_id not in {cand.origin_msg.id, *cand.related_msg_ids}:
        return False
    if index.chats[message.chat_id].is_group or not re.match(
        r"^(?:reverting\b|(?:i(?:'ll| will)|we(?:'ll| will))\s+(?:revert\b|respond\b|get back to you\b))",
        cand.origin_msg.body.strip(), re.IGNORECASE,
    ):
        return False
    origin = index.enriched_by_id.get(cand.origin_msg.id)
    response = index.enriched_by_id.get(message.id)
    if not origin or not response or origin.chat_index < 1 or response.chat_index != origin.chat_index + 1:
        return False
    if not timedelta(0) <= message.ts - cand.origin_msg.ts <= timedelta(hours=48):
        return False
    request = index.by_chat[message.chat_id][origin.chat_index - 1]
    if request.sender_id == cand.owner or cand.origin_msg.ts - request.ts > timedelta(hours=24):
        return False
    if not ("?" in request.body or re.search(r"\b(?:any (?:initial )?(?:thoughts|update|feedback|news|read)|how|what|when|whether)\b", request.body, re.IGNORECASE)):
        return False
    body = message.body.strip()
    if "?" in body or DEFER_ANSWER_RE.search(body) or NEGATED_OR_FUTURE_COMPLETION_RE.search(body) or PARTIAL_PROGRESS_GUARD_RE.search(body) or re.match(r"^(?:i(?:'ll| will)|we(?:'ll| will)|let me)\b", body, re.IGNORECASE):
        return False
    if re.search(r"\b(?:no(?:\s+\w+){0,2}\s+(?:update|news|answer)|nothing to report|not sure|don't know|do not know|cannot say|can't say|still checking|need more time|waiting for)\b", body, re.IGNORECASE):
        return False
    if not re.match(r"^(?:(?:hi|hello|hey)[,!\s]+)?(?:here(?:'s| is)\b|update\s*[:\-]|we (?:found|measured|observed)\b|(?:the|our|your)\b.{0,90}?\b(?:is|are|has|have|now)\b)", body, re.IGNORECASE):
        return False
    if len(tokenize(body, drop_stopwords=False)) < 16:
        return False
    generic = {"any", "initial", "read", "how", "things", "going", "what", "when", "whether", "update", "thoughts", "please", "can", "could", "need", "know", "feedback"}
    topics = set(tokenize(request.body)) - generic
    return bool(topics & set(tokenize(body)))


def _covers_task_bundle(cand: LoopCandidate, body: str) -> bool:
    """A statement about one item cannot resolve a bundled list of actions."""
    return not cand.is_task_list or bool(re.search(
        r"\b(?:everything|all\s+(?:of\s+)?(?:the\s+)?(?:tasks?|items?|chores?|work|list)|"
        r"all\s+(?:done|completed|finished|sorted)|every\s+(?:task|item|chore)|"
        r"(?:entire|whole)\s+(?:task\s+)?(?:list|set|job)|(?:this|the)\s+(?:task\s+)?list)\b",
        body, re.IGNORECASE
    ))


def reconcile_open_loops(
    candidates: List[LoopCandidate],
    index: ConversationIndex,
    people: Dict[str, PersonPrediction],
    llm: LLMClient,
    ledger: EvidenceLedger,
) -> List[OpenLoopPrediction]:
    """Assemble cross-chat trails and resolve final status, closure evidence, owner, and due_date."""
    resolved: List[OpenLoopPrediction] = []
    slow_verify_queue: List[Tuple[int, LoopCandidate, List[MessageRecord]]] = []
    chat_cand_counts: Dict[str, int] = {}
    for c in candidates:
        chat_cand_counts[c.origin_msg.chat_id] = chat_cand_counts.get(c.origin_msg.chat_id, 0) + 1

    for idx, cand in enumerate(candidates):
        trail = _retrieve_evidence_trail(cand, index)
        status, closure_id, eff_owner, eff_due, cl_conf = _evaluate_trail_deterministic(
            cand, trail, index, people, chat_loop_count=chat_cand_counts.get(cand.origin_msg.chat_id, 1)
        )

        loop_id = f"loop_{idx + 1:03d}"
        pred = OpenLoopPrediction(
            id=loop_id,
            title=cand.title,
            type=cand.loop_type,
            owner=eff_owner,
            chat_id=cand.origin_msg.chat_id,
            message_id=cand.origin_msg.id,
            due_date=eff_due,
            status=status,
            closure_message_id=closure_id if status in ("done", "dropped") else None,
            evidence_trail=[cand.origin_msg.id] + [m.id for m in trail if m.id != cand.origin_msg.id],
            confidence=round(cand.confidence, 3),
            closure_confidence=round(cl_conf, 3),
            source="inferred",
        )
        resolved.append(pred)

        # PRISM/FrugalGPT Selective Slow Mode: escalate ONLY uncertain boundary trails
        if trail and (0.35 <= cl_conf <= 0.80):
            slow_verify_queue.append((idx, cand, trail))

    if llm.is_available and slow_verify_queue:
        tasks: List[Tuple[str, str, str]] = []
        for idx, cand, trail in slow_verify_queue:
            origin_txt = index.format_turn(cand.origin_msg, include_chat=True)
            trail_txt = "\n".join(
                index.format_turn(m, include_chat=True) for m in trail[:14]
            )
            ref_dt = trail[-1].ts if trail else cand.origin_msg.ts
            cal_strip = ", ".join(
                (ref_dt + timedelta(days=d)).strftime("%a=%Y-%m-%d") for d in range(8)
            )
            usr_p = (
                f"Dataset End Timestamp: {index.anchor_ts.strftime('%Y-%m-%d %H:%M')} | "
                f"Calendar from last turn: {cal_strip}\n"
                f"Originating Loop ({cand.loop_type}, initial_owner={cand.owner}, initial_due={cand.due_date}):\n"
                f"{origin_txt}\n\nSubsequent Chronological Trail:\n{trail_txt}"
            )
            tasks.append((str(idx), SLOW_VERIFY_SYSTEM_PROMPT, usr_p))

        batch_out = llm.complete_json_batch(
            "temporal_slow_verify", tasks, max_tokens=600
        )
        valid_owners = set(people.keys()) | {USER_ID}
        for key_str, res in batch_out.items():
            if not isinstance(res, dict):
                continue
            r_idx = int(key_str)
            pred = resolved[r_idx]
            cand = candidates[r_idx]
            trail_ids = set(pred.evidence_trail) - {cand.origin_msg.id}

            llm_status = str(res.get("status") or pred.status).strip().lower()
            llm_cid = res.get("closure_message_id")
            llm_owner = str(res.get("owner") or pred.owner).strip()
            llm_due = res.get("due_date")
            llm_conf = float(res.get("confidence") or 0.80)

            # Enforce hard deterministic safety guards even over LLM outputs!
            if llm_status in ("done", "dropped"):
                if not llm_cid or str(llm_cid) not in trail_ids:
                    # Refuse ungrounded closure!
                    continue
                c_msg = index.messages_by_id[str(llm_cid)]
                if c_msg.ts < cand.origin_msg.ts:
                    continue
                c_raw = c_msg.body.strip()
                c_norm = normalize_indic_for_matching(c_raw) if not c_raw.isascii() else c_raw
                if not _covers_task_bundle(cand, c_norm):
                    continue
                # Hard post-LLM safety guards: never let a small LLM hallucinate `done` on bare acks, deferrals, or partial progress!
                if llm_status == "done":
                    if BARE_ACK_RE.match(c_raw) or BARE_ACK_RE.match(c_norm) or is_pure_indic_bare_ack(c_raw):
                        quoted = index.messages_by_id.get(c_msg.quoted_id)
                        if not quoted or not COMPLETION_CHECK_QUESTION_RE.search(quoted.body) or c_msg.sender_id != pred.owner:
                            continue
                    if (
                        c_msg.quoted_id == cand.origin_msg.id
                        and (BARE_ACK_RE.match(c_raw) or BARE_ACK_RE.match(c_norm) or is_pure_indic_bare_ack(c_raw))
                        and cand.loop_type != "question"
                    ):
                        continue
                    if (
                        NEGATED_OR_FUTURE_COMPLETION_RE.search(c_norm)
                        or PARTIAL_PROGRESS_GUARD_RE.search(c_norm)
                        or DEFER_ANSWER_RE.search(c_norm)
                    ):
                        continue
                pred.status = llm_status
                pred.closure_message_id = str(llm_cid)
                pred.closure_confidence = round(llm_conf, 3)
            elif llm_status == "open":
                pred.status = "open"
                pred.closure_message_id = None
                pred.closure_confidence = round(llm_conf, 3)

            if llm_owner in valid_owners:
                pred.owner = llm_owner
            if isinstance(llm_due, str) and re.match(r"^202\d-\d{2}-\d{2}$", llm_due):
                pred.due_date = llm_due

    for pred in resolved:
        ledger.record(
            subject_type="open_loop",
            subject_id=pred.id,
            field_name="status",
            predicted_value={
                "status": pred.status,
                "closure_message_id": pred.closure_message_id,
                "owner": pred.owner,
                "due_date": pred.due_date,
            },
            strength="explicit" if pred.closure_confidence >= 0.82 else "contextual",
            confidence=pred.closure_confidence,
            message_ids=pred.evidence_trail[:6],
            rationale=f"ARR resolved status={pred.status} (closure={pred.closure_message_id})",
            stage="temporal_reconciliation",
        )

    return resolved


def _retrieve_evidence_trail(
    cand: LoopCandidate, index: ConversationIndex
) -> List[MessageRecord]:
    """Retrieve chronological messages after origin_msg across originating chat, DMs, and lexical matches."""
    origin = cand.origin_msg
    seen: Dict[str, MessageRecord] = {}

    # 1. Direct reply chain (`quoted_id`)
    queue = list(index.replies_to.get(origin.id, []))
    while queue:
        rid = queue.pop(0)
        rm = index.messages_by_id.get(rid)
        if rm and rm.ts >= origin.ts and rm.id != origin.id:
            seen[rm.id] = rm
            queue.extend(index.replies_to.get(rm.id, []))

    # 2. Local subsequent turns in the same chat (thread-scoped: never pull replies to unrelated threads!)
    thread_msg_ids: Set[str] = {origin.id} | set(cand.related_msg_ids) | set(seen.keys())
    et = index.enriched_by_id.get(origin.id)
    if et:
        chat_msgs = index.by_chat[origin.chat_id]
        is_dm = not index.chats[origin.chat_id].is_group
        local_span = 12 if is_dm else 20
        for m in chat_msgs[et.chat_index + 1 : et.chat_index + 1 + local_span]:
            if (m.ts - origin.ts) > timedelta(days=14):
                break
            m_tok_list = tokenize(m.body, drop_stopwords=True)
            overlap = _compute_token_overlap(
                m_tok_list,
                cand.signature_tokens,
                allow_asr_fuzzy=(origin.kind == "voice_transcript" or m.kind == "voice_transcript"),
            )
            m_toks = set(m_tok_list)
            m_match = normalize_indic_for_matching(m.body) if not m.body.isascii() else m.body
            # If message explicitly quotes a DIFFERENT unrelated message thread, ignore unless >=2 shared task tokens!
            if m.quoted_id and m.quoted_id not in thread_msg_ids and overlap < 2:
                continue
            if (
                (is_dm and (m.ts - origin.ts) <= timedelta(hours=36))
                or (m.sender_id == cand.owner and (m.ts - origin.ts) <= timedelta(hours=24))
                or m.quoted_id in thread_msg_ids
                or (cand.signature_tokens and overlap >= 1)
                or (is_dm and (COMPLETION_CHECK_QUESTION_RE.search(m_match) or CANCELLATION_RE.search(m_match)))
            ):
                other_dist = (m_toks - cand.distinguishing_tokens) & {
                    "electricity", "water", "broadband", "wifi", "rent", "gst", "tds", "flight", "hotel"
                }
                if cand.distinguishing_tokens and other_dist and not (m_toks & cand.distinguishing_tokens):
                    continue
                seen[m.id] = m
                thread_msg_ids.add(m.id)

    # 3. Cross-chat hybrid search RESTRICTED to chats sharing an actual participant ID and >= 2 signature tokens!
    origin_parts = set(index.chats[origin.chat_id].participant_ids) - {USER_ID}
    allowed_people = cand.involved_people | origin_parts
    allowed_cross_chats: Set[str] = {origin.chat_id}
    for pid in allowed_people:
        allowed_cross_chats.update(index.person_chats.get(pid, set()))

    if cand.signature_tokens:
        query_str = " ".join(sorted(cand.signature_tokens))
        hits = index.search(
            query=query_str,
            top_k=15,
            candidate_chat_ids=allowed_cross_chats,
            participant_ids=allowed_people,
            after_ts=origin.ts,
            anchor_message_id=origin.id,
        )
        for m, score in hits:
            if m.id == origin.id or m.ts < origin.ts:
                continue
            m_tok_list = tokenize(m.body, drop_stopwords=True)
            overlap = _compute_token_overlap(
                m_tok_list,
                cand.signature_tokens,
                allow_asr_fuzzy=(origin.kind == "voice_transcript" or m.kind == "voice_transcript"),
            )
            if m.chat_id == origin.chat_id:
                if score >= 2.6 or overlap >= 1:
                    seen[m.id] = m
            else:
                # Strict cross-chat guard: require >= 2 shared topic tokens so shared names/generic words never bleed!
                if overlap >= 2:
                    seen[m.id] = m

    trail = sorted(seen.values(), key=lambda x: (x.ts, x.id))
    return trail


def _evaluate_trail_deterministic(
    cand: LoopCandidate,
    trail: List[MessageRecord],
    index: ConversationIndex,
    people: Dict[str, PersonPrediction],
    chat_loop_count: int = 1,
) -> Tuple[str, Optional[str], str, Optional[str], float]:
    """Apply the 4 Asymmetric Closure Guards chronologically over the retrieved trail."""
    status = "open"
    closure_id: Optional[str] = None
    eff_owner = cand.owner
    eff_due = cand.due_date
    confidence = 0.86 if not trail else 0.72

    pending_completion_check_ids: Set[str] = set()

    thread_ids: Set[str] = {cand.origin_msg.id} | set(cand.related_msg_ids)
    origin_all_toks = cand.signature_tokens | set(tokenize(cand.origin_msg.body, drop_stopwords=True))
    is_payment_loop = bool(origin_all_toks & PAYMENT_LOOP_WORDS)

    for m in trail:
        raw_body = (m.body or "").strip()
        if not raw_body and m.kind != "document":
            continue
        body = normalize_indic_for_matching(raw_body) if not raw_body.isascii() else raw_body

        m_tok_list = tokenize(raw_body, drop_stopwords=True)
        token_overlap = _compute_token_overlap(
            m_tok_list,
            cand.signature_tokens,
            allow_asr_fuzzy=(cand.origin_msg.kind == "voice_transcript" or m.kind == "voice_transcript"),
        )
        m_toks = set(m_tok_list)
        covers_bundle = _covers_task_bundle(cand, body)
        is_direct_reply = m.quoted_id is not None and m.quoted_id in thread_ids
        # Never treat a reply to an unrelated message as `same_chat_close`!
        quotes_unrelated = m.quoted_id is not None and m.quoted_id not in thread_ids
        if quotes_unrelated and token_overlap < 2:
            continue

        m_is_dm = not index.chats[m.chat_id].is_group if m.chat_id in index.chats else False
        same_chat_close = (
            m.chat_id == cand.origin_msg.chat_id
            and not quotes_unrelated
            and (m.ts - cand.origin_msg.ts) <= timedelta(hours=36)
            and (m_is_dm or is_direct_reply or token_overlap >= 1 or m.sender_id == eff_owner)
        )

        if not (is_direct_reply or token_overlap >= 1 or same_chat_close):
            continue

        if is_direct_reply:
            thread_ids.add(m.id)

        is_authorized_actor = m.sender_id in {eff_owner, cand.origin_msg.sender_id, USER_ID}

        # Typed delivery: a phone number or map pin can fulfil the corresponding
        # request without the word 'sent'. Require the owner, same DM, prompt
        # response and exact deliverable type; unrelated links remain evidence only.
        if (not cand.is_task_list and same_chat_close and m_is_dm and m.sender_id == eff_owner and not quotes_unrelated
                and (m.ts - cand.origin_msg.ts) <= timedelta(hours=2)
                and not re.search(r"\b(?:not|wrong|incorrect|don't|do not|will|later|tomorrow)\b", body, re.IGNORECASE)):
            phone_delivery = bool(origin_all_toks & {"number", "contact", "phone"}) and bool(re.search(r"(?<!\d)\d{10,13}(?!\d)", body))
            address_delivery = bool(origin_all_toks & {"address", "location"}) and bool(re.search(r"https?://(?:maps\.app\.goo\.gl/|(?:www\.)?google\.[^/]+/maps)", body))
            purchase_delivery = bool(origin_all_toks & {"buy", "purchase", "order"}) and bool(re.match(r"^(?:i (?:have |just )?)?(?:bought|purchased|ordered)\b", body, re.IGNORECASE)) and token_overlap >= 1
            email_delivery = bool(origin_all_toks & {"link", "document", "report", "invoice", "file"}) and bool(re.fullmatch(
                r"(?:it(?:'s| is|s)?|the (?:link|document|report|invoice|file) is)\s+(?:already\s+)?in\s+(?:your|ur)\s+(?:email|inbox)[.!\s]*",
                body, re.IGNORECASE))
            if phone_delivery or address_delivery or purchase_delivery or email_delivery:
                status, closure_id, confidence = "done", m.id, 0.89
                continue

        # 1. Check for failed/reopened completion AFTER a prior tentative done!
        if status == "done" and FAILURE_REOPEN_RE.search(body) and (is_authorized_actor or is_direct_reply):
            status = "open"
            closure_id = None
            confidence = 0.80
            continue

        # Once a loop is already marked `done`, do not let a later unrelated message in the same DM overwrite `closure_id`!
        if status == "done" and not is_direct_reply and token_overlap < 2 and not CANCELLATION_RE.search(body):
            continue

        # 2. Check for Ownership Transfer / Delegation (Adversarial #3, #7)
        if covers_bundle and m.sender_id != USER_ID and TAKEOVER_SELF_RE.search(body):
            if m_is_dm or is_direct_reply or token_overlap >= 1:
                eff_owner = m.sender_id
                confidence = 0.85
        elif covers_bundle and m.sender_id == USER_ID and DELEGATE_DIRECTIVE_RE.search(body):
            chat = index.chats.get(m.chat_id)
            parts = chat.participant_ids if chat else ()
            for p in parts:
                if p != USER_ID and p in people:
                    p_aliases = [
                        t for t in tokenize(people[p].name, drop_stopwords=True) if len(t) >= 3
                    ]
                    if len([x for x in parts if x != USER_ID]) == 1 or any(
                        a in body.lower() for a in p_aliases
                    ):
                        eff_owner = p
                        confidence = 0.84
                        break

        # 3. Check for Explicit Cancellation / Drop (Adversarial #8) — guarded against bystander noise in groups!
        if covers_bundle and CANCELLATION_RE.search(body) and (
            is_direct_reply or m_is_dm or (is_authorized_actor and token_overlap >= 1) or token_overlap >= 2
        ):
            status = "dropped"
            closure_id = m.id
            confidence = 0.90
            continue

        # 4. Check for Postponed / Rescheduled Deadline (Adversarial #7)
        if POSTPONE_RE.search(body) and (
            is_direct_reply or m_is_dm or (is_authorized_actor and token_overlap >= 1)
        ):
            new_due = resolve_due_date(body, m.ts)
            if new_due:
                eff_due = new_due
            if status != "done":
                status = "open"
                closure_id = None
                confidence = 0.86
            continue

        # 5. Track Completion-Check Questions ("Did you send it?", "Ho gaya kya?")
        if COMPLETION_CHECK_QUESTION_RE.search(body):
            if covers_bundle:
                pending_completion_check_ids.add(m.id)
            continue

        # 6. GUARD 1: Bare "Ok" / "Sure" / "Haan" / "👍" (including native Devanagari/Bengali "हाँ", "ठीक है", "হ্যাঁ")
        if BARE_ACK_RE.match(raw_body) or BARE_ACK_RE.match(body) or is_pure_indic_bare_ack(raw_body):
            # ONLY counts as `done` if replying to a completion-check question!
            if m.sender_id == eff_owner and m.quoted_id in pending_completion_check_ids and AFFIRMATIVE_CONFIRM_RE.search(body):
                status = "done"
                closure_id = m.id
                confidence = 0.88
            # Otherwise it is an acknowledgement of the request -> stays OPEN!
            continue

        # 6b. Follow-up deadline inheritance when loop is still open ("Any thoughts on Zest? They want an answer by Monday")
        if (
            status == "open"
            and eff_due is None
            and (m.id in thread_ids or token_overlap >= 1)
            and not DEFER_ANSWER_RE.search(body)
        ):
            inh_due = resolve_due_date(body, m.ts)
            if inh_due:
                eff_due = inh_due

        # 7. Substantive answer to an open `question` loop
        # Completing or delivering one list item does not complete the bundle.
        if not covers_bundle:
            continue
        if status == "open" and _delivers_promised_reply(cand, m, index):
            status, closure_id, confidence = "done", m.id, 0.86
            continue
        origin_turn = index.enriched_by_id.get(cand.origin_msg.id)
        response_turn = index.enriched_by_id.get(m.id)
        adjacent_answer = (
            m.chat_id == cand.origin_msg.chat_id
            and origin_turn is not None and response_turn is not None
            and response_turn.chat_index == origin_turn.chat_index + 1
            and (m.ts - cand.origin_msg.ts) <= timedelta(minutes=10)
        )
        if (
            cand.loop_type == "question"
            and status == "open"
            and m.sender_id == eff_owner
            and (m_is_dm or is_direct_reply or token_overlap >= 1)
            and (m.quoted_id == cand.origin_msg.id or same_chat_close)
            and (is_direct_reply or token_overlap >= 1 or adjacent_answer)
            and not DEFER_ANSWER_RE.search(body)
            and "?" not in body
            and len(m_toks) >= 2
        ):
            status = "done"
            closure_id = m.id
            confidence = 0.84
            continue

        # 8. Answer to a completion-check question with affirmative confirmation
        if (
            m.quoted_id in pending_completion_check_ids
            and AFFIRMATIVE_CONFIRM_RE.search(body)
            and not NEGATED_OR_FUTURE_COMPLETION_RE.search(body)
        ):
            status = "done"
            closure_id = m.id
            confidence = 0.89
            continue

        # 9. Deliverable document attachment (`kind == "document"`) sent by owner
        # Guard: never treat a new future commitment/request caption on a PDF (e.g., forwarding an unpaid bill PDF
        # with "I'll pay the broadband bill before 4 Nov") as closing an earlier task!
        is_future_or_request_caption = bool(
            NEGATED_OR_FUTURE_COMPLETION_RE.search(body)
            or re.search(
                r"\b(i'll|i\s+will|we'll|will\s+pay|will\s+send|please|pls|plz|kindly|can\s+you|need\s+to|have\s+to|before\s+\d)\b|\?",
                body,
                re.IGNORECASE,
            )
        )
        if (
            m.kind == "document"
            and m.sender_id == eff_owner
            and not is_future_or_request_caption
            and (
                is_direct_reply
                or (cand.loop_type in ("commitment", "followup", "deadline") and token_overlap >= 2)
                or (
                    m_is_dm
                    and m.chat_id == cand.origin_msg.chat_id
                    and cand.loop_type in ("commitment", "followup")
                    and chat_loop_count == 1
                    and (m.ts - cand.origin_msg.ts) <= timedelta(hours=48)
                )
            )
        ):
            status = "done"
            closure_id = m.id
            confidence = 0.88
            continue

        # 10. Requester receipt confirmation ("Received the audit report", "Mil gaya")
        if (
            m.sender_id in {cand.origin_msg.sender_id, USER_ID}
            and m.sender_id != eff_owner
            and RECEIPT_CONFIRM_RE.search(body)
            and not NEGATED_OR_FUTURE_COMPLETION_RE.search(body)
            and (is_direct_reply or token_overlap >= 1)
        ):
            status = "done"
            closure_id = m.id
            confidence = 0.88
            continue

        # 11. Banking/UPI rail completion on payment loops ("Sorted from HDFC!", "UPI done")
        if (
            is_payment_loop
            and m.sender_id == eff_owner
            and PAYMENT_RAIL_DONE_RE.search(body)
            and not NEGATED_OR_FUTURE_COMPLETION_RE.search(body)
            and (is_direct_reply or m_is_dm or token_overlap >= 1)
        ):
            status = "done"
            closure_id = m.id
            confidence = 0.87
            continue

        # 12. Short colloquial completion ("All sorted!", "Taken care of!") in an unambiguous 1:1 DM or direct reply
        if (
            m.sender_id == eff_owner
            and SHORT_DM_COMPLETION_RE.match(raw_body)
            and (
                is_direct_reply
                or (
                    m_is_dm
                    and m.chat_id == cand.origin_msg.chat_id
                    and chat_loop_count == 1
                    and (m.ts - cand.origin_msg.ts) <= timedelta(hours=48)
                )
            )
        ):
            status = "done"
            closure_id = m.id
            confidence = 0.84
            continue

        # 13. Explicit past-completive statement (Speaker-Authorized & Negation-Guarded!)
        if (
            EXPLICIT_COMPLETION_RE.search(body)
            and not NEGATED_OR_FUTURE_COMPLETION_RE.search(body)
            and not COMPLETION_CHECK_QUESTION_RE.search(body)
        ):
            if m_is_dm:
                can_close = is_direct_reply or token_overlap >= 1 or m.sender_id == eff_owner
            else:
                # In group chats, third-party bystanders require direct reply or >=2 shared task tokens!
                pickup_semantic_match = (
                    m.sender_id == eff_owner
                    and m.chat_id == cand.origin_msg.chat_id
                    and bool({"pick", "pickup", "tuition", "school"} & cand.signature_tokens)
                    and bool(re.search(r"\b(chhod\s+diya|dropped|picked)\b", body, re.IGNORECASE))
                )
                if is_authorized_actor:
                    can_close = is_direct_reply or token_overlap >= 1 or pickup_semantic_match
                else:
                    can_close = is_direct_reply or token_overlap >= 2
            if can_close:
                status = "done"
                closure_id = m.id
                confidence = 0.90 if (is_direct_reply or token_overlap >= 2) else 0.82

    # GUARD 4: Never infer completion from elapsed due_date!
    if status == "open":
        closure_id = None
        # Escalate to slow-pass LLM ONLY if eff_owner sent an unguarded substantive message after origin!
        has_unguarded_owner_turn = False
        for m in trail:
            if m.sender_id != eff_owner or m.ts <= cand.origin_msg.ts:
                continue
            rb = (m.body or "").strip()
            nb = normalize_indic_for_matching(rb) if not rb.isascii() else rb
            if (
                not rb
                or BARE_ACK_RE.match(rb)
                or BARE_ACK_RE.match(nb)
                or is_pure_indic_bare_ack(rb)
                or NEGATED_OR_FUTURE_COMPLETION_RE.search(nb)
                or PARTIAL_PROGRESS_GUARD_RE.search(nb)
                or DEFER_ANSWER_RE.search(nb)
                or POSTPONE_RE.search(nb)
            ):
                continue
            overlap = _compute_token_overlap(
                tokenize(rb, drop_stopwords=True), cand.signature_tokens,
                allow_asr_fuzzy=(cand.origin_msg.kind == "voice_transcript" or m.kind == "voice_transcript"),
            )
            origin_turn = index.enriched_by_id.get(cand.origin_msg.id)
            response_turn = index.enriched_by_id.get(m.id)
            adjacent = (
                m.chat_id == cand.origin_msg.chat_id
                and origin_turn is not None and response_turn is not None
                and response_turn.chat_index == origin_turn.chat_index + 1
                and (m.ts - cand.origin_msg.ts) <= timedelta(minutes=10)
            )
            if not (m.quoted_id in thread_ids or overlap >= 1 or adjacent or SHORT_DM_COMPLETION_RE.match(rb)):
                continue
            has_unguarded_owner_turn = True
            break
        if has_unguarded_owner_turn:
            confidence = min(confidence, 0.68)
        else:
            confidence = max(confidence, 0.85)

    return status, closure_id, eff_owner, eff_due, confidence
