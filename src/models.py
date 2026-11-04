"""Core data structures, official enums, and internal Evidence Ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

USER_ID = "ME"

VALID_CIRCLES: Tuple[str, ...] = (
    "spouse",
    "child",
    "parent",
    "in-law",
    "sibling",
    "relative",
    "friend",
    "colleague",
    "client",
    "vendor",
    "partner",
    "staff",
)

VALID_ORG_SIDES: Tuple[str, ...] = ("home", "work")

VALID_ORG_ROLES: Tuple[str, ...] = (
    "owner",
    "partner",
    "helper",
    "resident",
    "parent",
    "member",
    "other",
)

VALID_ORG_TYPES: Tuple[str, ...] = (
    "company",
    "residence",
    "vacation home",
    "workspace",
    "society",
    "school",
    "other",
)

ORG_TYPE_ALIASES: Dict[str, str] = {
    "home": "residence",
    "residence": "residence",
    "holiday_home": "vacation home",
    "vacation_home": "vacation home",
    "vacation home": "vacation home",
    "housing_society": "society",
    "society": "society",
    "coworking": "workspace",
    "workspace": "workspace",
    "company": "company",
    "school": "school",
    "community": "other",
    "other": "other",
}

VALID_ORG_RELATIONS: Tuple[str, ...] = ("runs", "supports")

VALID_GOAL_AREAS: Tuple[str, ...] = (
    "self",
    "spouse",
    "kids",
    "parents",
    "work",
    "friends",
    "experiences",
    "finances",
    "health",
    "home",
)

VALID_GOAL_TYPES: Tuple[str, ...] = ("habit", "milestone", "experience")

VALID_GOAL_HORIZONS: Tuple[str, ...] = ("3mo", "1yr", "3yr", "5yr")

VALID_LOOP_TYPES: Tuple[str, ...] = (
    "commitment",
    "followup",
    "deadline",
    "question",
)

VALID_LOOP_STATUSES: Tuple[str, ...] = ("open", "done", "dropped")

VALID_INTERVENTION_ACTIONS: Tuple[str, ...] = ("assist", "remind", "leave")

VALID_LEAVE_REASONS: Tuple[str, ...] = (
    "someone_elses",
    "personal",
    "annoying",
)

VALID_EVIDENCE_STRENGTHS: Tuple[str, ...] = (
    "seed",
    "user_correction",
    "explicit",
    "contextual",
    "conflicting",
    "insufficient",
)


@dataclass(frozen=True)
class ChatRecord:
    chat_id: str
    name: str
    is_group: bool
    participant_ids: Tuple[str, ...]


@dataclass(frozen=True)
class MessageRecord:
    id: str
    chat_id: str
    sender_id: str
    sender_name: str
    ts_raw: str
    ts: datetime
    kind: str  # text | voice_transcript | image_caption | document
    body: str
    quoted_id: Optional[str] = None
    line_index: int = 0


@dataclass
class SeedPerson:
    id: str
    circle: str
    name: Optional[str] = None
    supports_people: List[str] = field(default_factory=list)


@dataclass
class SeedOrganisation:
    name: str
    id: Optional[str] = None
    type: Optional[str] = None
    side: Optional[str] = None
    role: Optional[str] = None


@dataclass
class SeedGoal:
    title: str
    id: Optional[str] = None
    area: Optional[str] = None
    type: Optional[str] = None
    horizon: Optional[str] = None
    message_ids: List[str] = field(default_factory=list)


@dataclass
class SeedData:
    people: Dict[str, SeedPerson] = field(default_factory=dict)
    organisations: List[SeedOrganisation] = field(default_factory=list)
    goals: List[SeedGoal] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    chats: Dict[str, ChatRecord]
    messages: List[MessageRecord]
    messages_by_id: Dict[str, MessageRecord]
    seed: SeedData
    anchor_ts: datetime
    min_ts: datetime
    tz_suffix: str = ""


@dataclass
class EvidenceClaim:
    claim_id: str
    subject_type: str  # person | organisation | chat | goal | open_loop | intervention
    subject_id: str
    field_name: str
    predicted_value: Any
    strength: str  # seed | user_correction | explicit | contextual | conflicting | insufficient
    confidence: float
    message_ids: List[str] = field(default_factory=list)
    rationale: str = ""
    stage: str = ""
    conflicting_message_ids: List[str] = field(default_factory=list)


class EvidenceLedger:
    """Append-only internal evidence ledger tracking provenance and certainty."""

    def __init__(self) -> None:
        self._claims: List[EvidenceClaim] = []
        self._by_subject: Dict[Tuple[str, str, str], List[EvidenceClaim]] = {}

    def record(
        self,
        subject_type: str,
        subject_id: str,
        field_name: str,
        predicted_value: Any,
        strength: str,
        confidence: float,
        message_ids: Optional[List[str]] = None,
        rationale: str = "",
        stage: str = "",
        conflicting_message_ids: Optional[List[str]] = None,
    ) -> EvidenceClaim:
        if strength not in VALID_EVIDENCE_STRENGTHS:
            strength = "contextual"
        claim = EvidenceClaim(
            claim_id=f"ev_{len(self._claims) + 1:05d}",
            subject_type=subject_type,
            subject_id=subject_id,
            field_name=field_name,
            predicted_value=predicted_value,
            strength=strength,
            confidence=max(0.0, min(1.0, float(confidence))),
            message_ids=list(dict.fromkeys(m for m in (message_ids or []) if m)),
            rationale=rationale,
            stage=stage,
            conflicting_message_ids=list(
                dict.fromkeys(m for m in (conflicting_message_ids or []) if m)
            ),
        )
        self._claims.append(claim)
        key = (subject_type, subject_id, field_name)
        self._by_subject.setdefault(key, []).append(claim)
        return claim

    def latest(
        self, subject_type: str, subject_id: str, field_name: str
    ) -> Optional[EvidenceClaim]:
        history = self._by_subject.get((subject_type, subject_id, field_name))
        return history[-1] if history else None

    def history(
        self, subject_type: str, subject_id: str, field_name: str
    ) -> List[EvidenceClaim]:
        return list(self._by_subject.get((subject_type, subject_id, field_name), []))

    def all_claims(self) -> List[EvidenceClaim]:
        return list(self._claims)

    def summary_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {k: 0 for k in VALID_EVIDENCE_STRENGTHS}
        for c in self._claims:
            counts[c.strength] = counts.get(c.strength, 0) + 1
        return counts


@dataclass
class OrgLink:
    organisation_id: str
    relation: str  # runs | supports


@dataclass
class PersonPrediction:
    id: str
    circle: str
    name: str = ""
    organisation_links: List[OrgLink] = field(default_factory=list)
    supports_people: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    source: str = "inferred"  # seed | inferred | corrected


@dataclass
class OrganisationPrediction:
    id: str
    name: str
    type: str
    side: str
    role: str
    people: List[str] = field(default_factory=list)
    chats: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    source: str = "inferred"


@dataclass
class ChatPrediction:
    chat_id: str
    owner_type: str  # person | organisation
    owner_id: str
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    source: str = "inferred"


@dataclass
class GoalPrediction:
    id: str
    title: str
    area: str
    type: str
    horizon: str
    message_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    source: str = "inferred"


@dataclass
class InterventionPrediction:
    action: str  # assist | remind | leave
    next_step: Optional[str] = None
    needs_approval: Optional[bool] = None
    remind_at: Optional[str] = None
    leave_reason: Optional[str] = None
    confidence: float = 0.5
    source: str = "inferred"


@dataclass
class OpenLoopPrediction:
    id: str
    title: str
    type: str  # commitment | followup | deadline | question
    owner: str  # ME or person_id
    chat_id: str
    message_id: str
    due_date: Optional[str] = None  # YYYY-MM-DD or None
    status: str = "open"  # open | done | dropped
    closure_message_id: Optional[str] = None
    intervention: Optional[InterventionPrediction] = None
    evidence_trail: List[str] = field(default_factory=list)
    confidence: float = 0.5
    closure_confidence: float = 0.5
    source: str = "inferred"


@dataclass
class PipelineTelemetry:
    stage_timings_sec: Dict[str, float] = field(default_factory=dict)
    stage_tokens: Dict[str, Dict[str, int]] = field(default_factory=dict)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    dollars_per_1k_messages: float = 0.0
    total_messages: int = 0
    model_used: str = "deterministic-tier0"
    evidence_summary: Dict[str, int] = field(default_factory=dict)
    validation_repairs: List[str] = field(default_factory=list)


@dataclass
class ColdStartOutput:
    people: List[PersonPrediction]
    organisations: List[OrganisationPrediction]
    chats: List[ChatPrediction]
    goals: List[GoalPrediction]
    open_loops: List[OpenLoopPrediction]
    telemetry: PipelineTelemetry = field(default_factory=PipelineTelemetry)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
