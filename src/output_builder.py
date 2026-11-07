"""End-to-end Pipeline Orchestrator, Cost/Timing Telemetry, and Deterministic JSON Output Builder."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .conversation_index import ConversationIndex
from .corrections import apply_corrections, load_corrections_file
from .data_loader import load_dataset
from .evidence_validator import validate_and_repair
from .goal_extraction import extract_goals
from .intervention_policy import apply_intervention_policy
from .llm_client import LLMClient
from .loop_extraction import extract_open_loop_candidates
from .models import ColdStartOutput, EvidenceLedger, PipelineTelemetry
from .organisation_inference import infer_organisations_and_chats
from .relationship_inference import infer_people_circles
from .schema_adapter import (
    load_schema,
    serialize_output_for_schema,
    validate_json_against_schema,
)
from .temporal_resolver import reconcile_open_loops
from .submission_validator import validate_references


def run_pipeline(
    input_folder: str | Path,
    output_path: Optional[str | Path] = None,
    corrections_path: Optional[str | Path] = None,
    corrections_dict: Optional[Dict[str, Any]] = None,
    schema_path: Optional[str | Path] = None,
    telemetry_path: Optional[str | Path] = None,
    llm_client: Optional[LLMClient] = None,
    mock_responder: Optional[Callable[[str, str, str], Dict[str, Any]]] = None,
) -> Tuple[ColdStartOutput, Dict[str, Any]]:
    """Run the complete Dossier-ARR-PRISM pipeline and optionally write output.json."""
    timings: Dict[str, float] = {}
    t_start = time.perf_counter()

    # Stage 0: Load & Index
    t0 = time.perf_counter()
    bundle = load_dataset(input_folder)
    index = ConversationIndex(bundle)
    timings["01_ingest_and_index"] = round(time.perf_counter() - t0, 4)

    llm = llm_client or LLMClient(mock_responder=mock_responder)
    ledger = EvidenceLedger()

    # Stage 1: People Circles & Staff Support
    t1 = time.perf_counter()
    people_map = infer_people_circles(index, llm, ledger)
    timings["02_relationship_inference"] = round(time.perf_counter() - t1, 4)

    # Stage 2: Joint Organisations & Chat Ownership
    t2 = time.perf_counter()
    organisations, chats = infer_organisations_and_chats(
        index, people_map, llm, ledger
    )
    timings["03_organisation_and_chat_inference"] = round(
        time.perf_counter() - t2, 4
    )

    # Stage 3: Precision-First Goal Extraction
    t3 = time.perf_counter()
    goals = extract_goals(index, people_map, llm, ledger, max_goals=10)
    timings["04_goal_extraction"] = round(time.perf_counter() - t3, 4)

    # Stage 4: Open-Loop Discovery & ARR Temporal Reconciliation
    t4 = time.perf_counter()
    candidates = extract_open_loop_candidates(index, people_map, llm, ledger)
    open_loops = reconcile_open_loops(candidates, index, people_map, llm, ledger)
    timings["05_loop_discovery_and_reconciliation"] = round(
        time.perf_counter() - t4, 4
    )

    # Stage 5: PRISM Risk-Sensitive Intervention Policy
    t5 = time.perf_counter()
    apply_intervention_policy(open_loops, index, people_map, llm, ledger)
    timings["06_intervention_policy"] = round(time.perf_counter() - t5, 4)

    output = ColdStartOutput(
        people=[people_map[pid] for pid in sorted(people_map.keys())],
        organisations=organisations,
        chats=chats,
        goals=goals,
        open_loops=open_loops,
        ledger=ledger,
    )

    # Stage 6: Optional Live-Interview Corrections
    t6 = time.perf_counter()
    corr_data: Dict[str, Any] = {}
    if corrections_path:
        corr_data.update(load_corrections_file(corrections_path))
    if corrections_dict:
        for k, v in corrections_dict.items():
            if isinstance(v, dict) and isinstance(corr_data.get(k), dict):
                corr_data[k].update(v)
            else:
                corr_data[k] = v
    if corr_data:
        apply_corrections(output, corr_data, index)
    timings["07_corrections"] = round(time.perf_counter() - t6, 4)

    # Stage 7: Deterministic Validation & Schema Adaptation
    t7 = time.perf_counter()
    repairs = validate_and_repair(output, index)
    schema = load_schema(schema_path)
    serialized = serialize_output_for_schema(output, schema)
    schema_errors = validate_json_against_schema(serialized, schema)
    # Check exactly the JSON representation that will be written, after repairs.
    serialized = json.loads(json.dumps(serialized, ensure_ascii=False, allow_nan=False))
    canonical = serialize_output_for_schema(output, load_schema())
    reference_errors = validate_references(canonical, index)
    if schema_errors or reference_errors:
        raise ValueError("Output validation failed: " + "; ".join((schema_errors + reference_errors)[:20]))
    timings["08_validation_and_serialization"] = round(time.perf_counter() - t7, 4)
    timings["total_sec"] = round(time.perf_counter() - t_start, 4)

    usage = llm.usage_snapshot()
    msg_count = max(1, len(bundle.messages))
    cost_usd = float(usage["estimated_cost_usd"])
    dollars_per_1k = (cost_usd / msg_count) * 1000.0

    output.telemetry = PipelineTelemetry(
        stage_timings_sec=timings,
        stage_tokens=usage["stages"],
        llm_calls=int(usage["calls"]),
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
        total_tokens=int(usage["total_tokens"]),
        estimated_cost_usd=round(cost_usd, 6),
        dollars_per_1k_messages=round(dollars_per_1k, 6),
        total_messages=len(bundle.messages),
        model_used=str(usage["model"]),
        evidence_summary=ledger.summary_counts(),
        validation_repairs=repairs,
    )

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # A failed write must not leave a truncated submission or replace a valid file.
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out_p.parent,
                                             prefix=".xpert-", suffix=".tmp", delete=False) as fh:
                temp_name = fh.name
                json.dump(serialized, fh, indent=2, ensure_ascii=False, allow_nan=False)
                fh.write("\n")
            os.replace(temp_name, out_p)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    if telemetry_path is not None:
        tel_p = Path(telemetry_path)
        tel_p.parent.mkdir(parents=True, exist_ok=True)
        tel_p.write_text(
            json.dumps(
                {
                    "model_used": output.telemetry.model_used,
                    "total_messages": output.telemetry.total_messages,
                    "llm_calls": output.telemetry.llm_calls,
                    "prompt_tokens": output.telemetry.prompt_tokens,
                    "completion_tokens": output.telemetry.completion_tokens,
                    "total_tokens": output.telemetry.total_tokens,
                    "estimated_cost_usd": output.telemetry.estimated_cost_usd,
                    "dollars_per_1k_messages": output.telemetry.dollars_per_1k_messages,
                    "stage_timings_sec": output.telemetry.stage_timings_sec,
                    "stage_tokens": output.telemetry.stage_tokens,
                    "usage_details": usage,
                    "evidence_summary": output.telemetry.evidence_summary,
                    "validation_repairs": output.telemetry.validation_repairs,
                    "counts": {
                        "people": len(output.people),
                        "organisations": len(output.organisations),
                        "chats": len(output.chats),
                        "goals": len(output.goals),
                        "open_loops": len(output.open_loops),
                        "open_loops_open": sum(
                            1 for l in output.open_loops if l.status == "open"
                        ),
                        "open_loops_done": sum(
                            1 for l in output.open_loops if l.status == "done"
                        ),
                        "open_loops_dropped": sum(
                            1 for l in output.open_loops if l.status == "dropped"
                        ),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    return output, serialized
