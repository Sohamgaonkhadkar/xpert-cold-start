"""Immutable verbatim conversation index with quoted-reply trees, BM25, and char n-gram retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .indic_normalizer import normalize_indic_for_matching
from .models import USER_ID, ChatRecord, DatasetBundle, MessageRecord

# Multilingual token regex: Latin words, digits, and major Indian scripts (Devanagari..Malayalam \u0900-\u0d7f)
TOKEN_RE = re.compile(r"[a-z0-9\u0900-\u0d7f]+", re.IGNORECASE)

STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "am", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "at", "for", "with", "and", "or", "but", "so", "if", "it", "this",
    "that", "i", "me", "my", "you", "your", "we", "our", "they", "he", "she",
    "hai", "hain", "ho", "tha", "thi", "the", "ka", "ki", "ke", "ko", "se", "me",
    "mein", "par", "pe", "bhi", "aur", "ya", "toh", "na", "nhi", "nahi", "kya",
    "main", "hum", "tum", "aap", "ye", "yeh", "wo", "woh", "ek", "koi",
    "ar", "o", "ebong", "theke", "kore", "ache", "chilo", "ami", "tumi", "apni",
}


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    norm_text = normalize_indic_for_matching(text) if text and not text.isascii() else (text or "")
    tokens = [t.lower() for t in TOKEN_RE.findall(norm_text)]
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def char_ngrams(text: str, n_values: Tuple[int, ...] = (3, 4)) -> Counter[str]:
    norm = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not norm:
        return Counter()
    padded = f" {norm} "
    counts: Counter[str] = Counter()
    length = len(padded)
    for n in n_values:
        if length >= n:
            for i in range(length - n + 1):
                counts[padded[i : i + n]] += 1
    return counts


@dataclass(frozen=True)
class EnrichedTurn:
    message: MessageRecord
    quoted_message: Optional[MessageRecord]
    reply_children_ids: Tuple[str, ...]
    chat_index: int  # 0-based position inside its chat


class ConversationIndex:
    """Dual-track Track-1 store: lossless speaker-grounded turns + hybrid local search."""

    def __init__(self, bundle: DatasetBundle) -> None:
        self.bundle = bundle
        self.chats: Dict[str, ChatRecord] = bundle.chats
        self.messages: List[MessageRecord] = bundle.messages
        self.messages_by_id: Dict[str, MessageRecord] = bundle.messages_by_id
        self.anchor_ts: datetime = bundle.anchor_ts
        self.min_ts: datetime = bundle.min_ts

        self.by_chat: Dict[str, List[MessageRecord]] = defaultdict(list)
        self.replies_to: Dict[str, List[str]] = defaultdict(list)
        self.person_names: Dict[str, Counter[str]] = defaultdict(Counter)
        self.person_messages: Dict[str, List[MessageRecord]] = defaultdict(list)
        self.person_chats: Dict[str, Set[str]] = defaultdict(set)
        self.dm_chat_by_person: Dict[str, str] = {}
        self.person_by_dm_chat: Dict[str, str] = {}
        self.co_participants: Dict[str, Counter[str]] = defaultdict(Counter)
        self.me_aliases: Set[str] = set()
        self.me_surnames: Set[str] = set()
        self.mention_handle_owner: Dict[str, str] = {}
        self.me_mention_handles: Set[str] = set()
        self._ngram_cache: Dict[str, Tuple[Counter[str], float]] = {}

        self._build_structural_indices()
        self._build_lexical_indices()
        self._resolve_numeric_mentions()
        self._discover_me_aliases()

    def _build_structural_indices(self) -> None:
        for cid, chat in self.chats.items():
            non_me = [p for p in chat.participant_ids if p != USER_ID]
            for p in non_me:
                self.person_chats[p].add(cid)
            if not chat.is_group and len(non_me) == 1:
                p = non_me[0]
                # Prefer chat with more messages if multiple DMs exist
                self.dm_chat_by_person.setdefault(p, cid)
                self.person_by_dm_chat[cid] = p
            for i, p1 in enumerate(non_me):
                for p2 in non_me[i + 1 :]:
                    self.co_participants[p1][p2] += 1
                    self.co_participants[p2][p1] += 1

        self.messages_by_id = {m.id: m for m in self.messages}
        for msg in self.messages:
            self.by_chat[msg.chat_id].append(msg)
            if msg.quoted_id and msg.quoted_id in self.messages_by_id:
                self.replies_to[msg.quoted_id].append(msg.id)
            if msg.sender_id and msg.sender_id != USER_ID:
                self.person_messages[msg.sender_id].append(msg)
                self.person_chats[msg.sender_id].add(msg.chat_id)
                if msg.sender_name and msg.sender_name != msg.sender_id:
                    self.person_names[msg.sender_id][msg.sender_name] += 1

        # Include seeded people even if they have 0 messages/chats
        for pid, sp in self.bundle.seed.people.items():
            if pid != USER_ID:
                _ = self.person_chats[pid]
                if sp.name:
                    self.person_names[pid][sp.name] += 5

        # Re-select primary DM chat by highest message volume
        for cid, chat in self.chats.items():
            non_me = [p for p in chat.participant_ids if p != USER_ID]
            if not chat.is_group and len(non_me) == 1:
                p = non_me[0]
                prev = self.dm_chat_by_person.get(p)
                if prev is None or len(self.by_chat[cid]) > len(self.by_chat[prev]):
                    self.dm_chat_by_person[p] = cid
                # Also infer contact display name from 1:1 chat title when informative
                if chat.name and chat.name != cid and chat.name != p:
                    self.person_names[p][chat.name] += 2

        self.enriched_by_id: Dict[str, EnrichedTurn] = {}
        for cid, chat_msgs in self.by_chat.items():
            for idx, m in enumerate(chat_msgs):
                q_msg = (
                    self.messages_by_id.get(m.quoted_id) if m.quoted_id else None
                )
                self.enriched_by_id[m.id] = EnrichedTurn(
                    message=m,
                    quoted_message=q_msg,
                    reply_children_ids=tuple(self.replies_to.get(m.id, [])),
                    chat_index=idx,
                )

    def _build_lexical_indices(self) -> None:
        self._doc_tokens: Dict[str, Counter[str]] = {}
        self._doc_len: Dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._inverted: Dict[str, List[str]] = defaultdict(list)

        total_len = 0
        for m in self.messages:
            q_body = ""
            if m.quoted_id and m.quoted_id in self.messages_by_id:
                q_body = self.messages_by_id[m.quoted_id].body
            combined = f"{m.sender_name} {m.body} {q_body}".strip()
            toks = tokenize(combined, drop_stopwords=True)
            tf = Counter(toks)
            self._doc_tokens[m.id] = tf
            self._doc_len[m.id] = len(toks)
            total_len += len(toks)
            for term in tf.keys():
                self._df[term] += 1
                self._inverted[term].append(m.id)

        self._num_docs = max(1, len(self.messages))
        self._avg_dl = max(1.0, total_len / self._num_docs)

    def _discover_me_aliases(self) -> None:
        """Dynamically infer ME's first name(s) from 1:1 DM vocatives & direct replies to ME (zero dev-name hardcoding)."""
        non_name_vocatives = {
            "hey", "hi", "hello", "dear", "sir", "madam", "maam", "sahab", "malik",
            "bhai", "bhaiya", "didi", "dada", "beta", "betu", "baba", "maa", "mom",
            "mummy", "papa", "dad", "bro", "yaar", "dude", "jaan", "love", "team",
            "all", "everyone", "guys", "folks", "good", "happy", "shubho", "pranam",
            "please", "pls", "plz", "kindly", "sure", "yes", "yeah", "okay", "noted",
            "thanks", "thank", "paid", "sent", "shared", "done", "check", "update",
            "can", "could", "would", "did", "have", "traffic", "heavy", "match",
            "forwarded", "let", "sorry", "great", "still", "leave", "landlord",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december",
        }
        other_people_toks: Set[str] = set()
        for pid in self.all_person_ids():
            other_people_toks.update(tokenize(self.best_name(pid), drop_stopwords=False))
        for org in self.bundle.seed.organisations:
            other_people_toks.update(tokenize(org.name, drop_stopwords=False))
        for chat in self.chats.values():
            if chat.is_group:
                other_people_toks.update(tokenize(chat.name, drop_stopwords=False))

        # 0. Direct ground-truth sender_name on ME's own messages (e.g. "Meera" or "Arjun Sharma")
        me_sender_names: Counter[str] = Counter()
        for m in self.messages:
            if m.sender_id == USER_ID and m.sender_name and m.sender_name.strip().upper() != USER_ID:
                me_sender_names[m.sender_name.strip()] += 1
        if me_sender_names:
            top_me_name = me_sender_names.most_common(1)[0][0]
            parts = [t.lower() for t in TOKEN_RE.findall(top_me_name) if len(t) >= 2]
            if parts and parts[0] not in non_name_vocatives:
                self.me_aliases.add(parts[0])
            if len(parts) >= 2 and parts[-1] not in non_name_vocatives and len(parts[-1]) >= 3:
                self.me_surnames.add(parts[-1])

        voc_re = re.compile(
            r"^(?:hey|hi|hello|dear|oi|suno)?\s*([A-Z][a-z]{2,14})\s*(?:[,!:]|\b(?:baba|beta|bhai|bhaiya|sir|da|dada|ji|can|could|please|pls|spoke)\b)"
        )
        senders_by_cand: Dict[str, Set[str]] = defaultdict(set)
        for m in self.messages:
            if m.sender_id == USER_ID or not m.body:
                continue
            chat = self.chats.get(m.chat_id)
            is_to_me = (chat is not None and not chat.is_group) or (
                m.quoted_id is not None
                and m.quoted_id in self.messages_by_id
                and self.messages_by_id[m.quoted_id].sender_id == USER_ID
            ) or (chat is not None and chat.is_group)
            if not is_to_me:
                continue
            match = voc_re.search(m.body.strip())
            if match:
                cand = match.group(1).lower()
                if cand not in non_name_vocatives and cand not in other_people_toks:
                    # Weight 1:1 DM or direct reply to ME doubly
                    if (chat and not chat.is_group) or (
                        m.quoted_id in self.messages_by_id
                        and self.messages_by_id[m.quoted_id].sender_id == USER_ID
                    ):
                        senders_by_cand[cand].add(f"dm:{m.sender_id}")
                    senders_by_cand[cand].add(m.sender_id)

        for cand, s_set in senders_by_cand.items():
            # Group-only vocatives need independent witnesses. A DM/reply is
            # stronger addressing evidence, but sentence words are never names.
            witnesses = {s for s in s_set if not s.startswith("dm:")}
            if (len(witnesses) >= 2 or any(s.startswith("dm:") for s in s_set)) and cand not in STOPWORDS and cand not in {
                "you", "they", "also", "oops", "wow", "surely", "congratulations",
                "newspaper", "hahahahah", "this", "that", "these", "those", "there",
                "here", "thanks", "thank", "please", "yes", "okay", "sorry",
            }:
                self.me_aliases.add(cand)

        # Augment family surnames from seeded core family (spouse/parent/sibling/child) & >=3 non-role 1:1 DM contacts
        role_or_generic_last = non_name_vocatives | {
            "driver", "cook", "maid", "nanny", "guard", "watchman", "mali", "cleaner", "helper",
            "caretaker", "plumber", "electrician", "carpenter", "tutor", "teacher", "coach",
            "traders", "trading", "enterprises", "services", "solutions", "agency", "printers",
            "press", "builders", "clinic", "doctor", "dentist", "lawyer", "advocate", "home",
            "ghar", "office", "farm", "farmhouse", "villa", "admin", "desk", "reception",
        }
        def _clean_surname(raw_nm: str) -> Optional[str]:
            # WhatsApp contact IDs / synthetic exports often embed tokens like p_scale_031 or digits;
            # never turn those machine identifiers into a shared family surname.
            if any(ch in raw_nm for ch in ("-", "@", "|", "_", "#")) or re.search(r"\d", raw_nm):
                return None
            ws = [w.lower() for w in re.findall(r"[A-Za-z]+", raw_nm) if len(w) >= 2]
            while ws and ws[-1] in role_or_generic_last:
                ws.pop()
            if len(ws) >= 2 and len(ws[-1]) >= 3 and ws[-1] not in role_or_generic_last:
                return ws[-1]
            return None

        for pid, sp in self.bundle.seed.people.items():
            if sp.circle in ("spouse", "parent", "sibling", "child"):
                sn = _clean_surname(self.best_name(pid))
                if sn:
                    self.me_surnames.add(sn)
        dm_surname_counts: Counter[str] = Counter()
        for pid in self.dm_chat_by_person:
            sn = _clean_surname(self.best_name(pid))
            if sn:
                dm_surname_counts[sn] += 1
        for sn, cnt in dm_surname_counts.items():
            if cnt >= 3:
                self.me_surnames.add(sn)

    def _resolve_numeric_mentions(self) -> None:
        """Map WhatsApp @123456 numeric JID/LID mention handles via cross-chat participant intersection."""
        mention_re = re.compile(r"@(\d{5,})")
        where: Dict[str, Set[str]] = defaultdict(set)
        writers: Dict[str, Set[str]] = defaultdict(set)
        for m in self.messages:
            if not m.body or "@" not in m.body:
                continue
            for h in mention_re.findall(m.body):
                where[h].add(m.chat_id)
                writers[h].add(m.sender_id)
        for h, chat_ids in where.items():
            common: Optional[Set[str]] = None
            for cid in chat_ids:
                if cid in self.chats:
                    parts = set(self.chats[cid].participant_ids)
                    common = parts if common is None else (common & parts)
            candidates = (common or set()) - writers[h]
            if len(candidates) == 1:
                owner = next(iter(candidates))
                self.mention_handle_owner[h] = owner
                if owner == USER_ID:
                    self.me_mention_handles.add(h)

    def all_person_ids(self) -> List[str]:
        """Official check.py only accepts person_ids present in chats.jsonl `participants` (excluding ME)."""
        return sorted(
            {
                p
                for chat in self.chats.values()
                for p in chat.participant_ids
                if p and p != USER_ID
            }
        )

    def best_name(self, person_id: str) -> str:
        if person_id == USER_ID:
            return USER_ID
        counts = self.person_names.get(person_id)
        if counts:
            return counts.most_common(1)[0][0]
        if person_id in self.bundle.seed.people and self.bundle.seed.people[person_id].name:
            return str(self.bundle.seed.people[person_id].name)
        return person_id

    def get_window(
        self, message_id: str, before: int = 3, after: int = 3
    ) -> List[MessageRecord]:
        et = self.enriched_by_id.get(message_id)
        if et is None:
            return []
        chat_msgs = self.by_chat[et.message.chat_id]
        lo = max(0, et.chat_index - before)
        hi = min(len(chat_msgs), et.chat_index + after + 1)
        window = list(chat_msgs[lo:hi])
        # Ensure quoted parent message is included even if far earlier in chat
        if et.quoted_message and et.quoted_message.id not in {m.id for m in window}:
            window.insert(0, et.quoted_message)
        return window

    def format_turn(self, msg: MessageRecord, include_chat: bool = False) -> str:
        prefix = f"[{msg.id} | {msg.ts.strftime('%Y-%m-%d %H:%M')}"
        if include_chat:
            cname = self.chats.get(msg.chat_id).name if msg.chat_id in self.chats else msg.chat_id
            prefix += f" | chat:{msg.chat_id}({cname})"
        speaker = (
            "ME"
            if msg.sender_id == USER_ID
            else f"{msg.sender_id}({msg.sender_name or self.best_name(msg.sender_id)})"
        )
        kind_tag = "" if msg.kind == "text" else f" <{msg.kind}>"
        quote_tag = ""
        if msg.quoted_id and msg.quoted_id in self.messages_by_id:
            qm = self.messages_by_id[msg.quoted_id]
            q_snip = qm.body[:70].replace("\n", " ")
            quote_tag = f" [replying_to {qm.id} by {qm.sender_id}: \"{q_snip}\"]"
        body_clean = (msg.body or "").replace("\n", " ").strip()
        return f"{prefix}] {speaker}{kind_tag}{quote_tag}: {body_clean}"

    def search(
        self,
        query: str,
        top_k: int = 20,
        candidate_chat_ids: Optional[Set[str]] = None,
        participant_ids: Optional[Set[str]] = None,
        after_ts: Optional[datetime] = None,
        before_ts: Optional[datetime] = None,
        anchor_message_id: Optional[str] = None,
    ) -> List[Tuple[MessageRecord, float]]:
        """Hybrid BM25 + char 3/4-gram cosine + structural thread/entity boost."""
        q_tokens = tokenize(query, drop_stopwords=True)
        q_ngrams = char_ngrams(query)
        q_ngram_norm = math.sqrt(sum(v * v for v in q_ngrams.values())) or 1.0

        candidate_ids: Set[str] = set()
        for qt in q_tokens:
            candidate_ids.update(self._inverted.get(qt, []))

        if anchor_message_id:
            for r_id in self.replies_to.get(anchor_message_id, []):
                candidate_ids.add(r_id)
            et = self.enriched_by_id.get(anchor_message_id)
            if et:
                chat_msgs = self.by_chat[et.message.chat_id]
                for m in chat_msgs[et.chat_index + 1 : et.chat_index + 35]:
                    candidate_ids.add(m.id)

        if participant_ids:
            for pid in participant_ids:
                dm_cid = self.dm_chat_by_person.get(pid)
                if dm_cid:
                    for m in self.by_chat.get(dm_cid, []):
                        if after_ts is None or m.ts >= after_ts:
                            candidate_ids.add(m.id)

        if candidate_chat_ids and not candidate_ids:
            for cid in candidate_chat_ids:
                for m in self.by_chat.get(cid, []):
                    candidate_ids.add(m.id)

        k1 = 1.5
        b = 0.75
        phase1: List[Tuple[MessageRecord, float]] = []

        for mid in candidate_ids:
            msg = self.messages_by_id[mid]
            if after_ts is not None and msg.ts < after_ts:
                continue
            if before_ts is not None and msg.ts > before_ts:
                continue
            if candidate_chat_ids is not None and msg.chat_id not in candidate_chat_ids:
                continue

            # Phase 1: Fast O(|Q|) BM25 + structural thread/participant boosts
            bm25 = 0.0
            tf_map = self._doc_tokens.get(mid, Counter())
            dl = self._doc_len.get(mid, 1)
            for qt in q_tokens:
                tf = tf_map.get(qt, 0)
                if tf > 0:
                    df = self._df.get(qt, 1)
                    idf = math.log(1.0 + (self._num_docs - df + 0.5) / (df + 0.5))
                    denom = tf + k1 * (1.0 - b + b * (dl / self._avg_dl))
                    bm25 += idf * ((tf * (k1 + 1.0)) / denom)

            boost = 0.0
            if anchor_message_id and msg.quoted_id == anchor_message_id:
                boost += 4.5
            if participant_ids and msg.sender_id in participant_ids:
                boost += 0.8

            phase1.append((msg, bm25 + boost))

        # Sort Phase 1 so expensive char 3/4-gram cosine runs ONLY on top 80 candidates (20x faster at 20k scale!)
        phase1.sort(key=lambda item: (-item[1], item[0].ts, item[0].id))
        scored: List[Tuple[MessageRecord, float]] = []

        for idx_p, (msg, base_score) in enumerate(phase1):
            ng_sim = 0.0
            if idx_p < 80 and q_ngrams and msg.body:
                cached = self._ngram_cache.get(msg.id)
                if cached is None:
                    m_ng = char_ngrams(msg.body)
                    m_norm = math.sqrt(sum(v * v for v in m_ng.values())) or 1.0
                    cached = (m_ng, m_norm)
                    self._ngram_cache[msg.id] = cached
                m_ng, m_norm = cached
                if m_ng:
                    dot = sum(
                        v * m_ng.get(k, 0) for k, v in q_ngrams.items() if k in m_ng
                    )
                    if dot > 0:
                        ng_sim = dot / (q_ngram_norm * m_norm)

            total = base_score + 2.5 * ng_sim
            if total > 0.05:
                scored.append((msg, total))

        scored.sort(key=lambda item: (-item[1], item[0].ts, item[0].id))
        return scored[:top_k]
