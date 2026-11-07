"""Regressions for actual audit failures; no private challenge content."""
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from src import schema_adapter
from src.conversation_index import ConversationIndex
from src.corrections import apply_corrections, parse_natural_language_correction
from src.data_loader import load_dataset
from src.llm_client import LLMClient
from src.output_builder import run_pipeline
from src.submission_validator import validate_references


@pytest.fixture(autouse=True)
def no_external_model(monkeypatch):
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


def fixture(root, bodies, *, dates=None, group=False):
    root.mkdir(parents=True, exist_ok=True)
    chats = [{"chat_id": "c", "name": "Project" if group else "Mira", "is_group": group,
              "participants": ["ME", "p"]}]
    messages = [dict(id=f"s{i}", chat_id="c", sender_id=sender,
                     sender_name="Mira" if sender == "p" else "ME",
                     ts=dates[i] if dates else f"2026-10-02T10:{i:02d}:00+05:30",
                     body=body, kind="text", quoted_id=None)
                for i, (sender, body) in enumerate(bodies)]
    for name, rows in (("chats", chats), ("messages", messages)):
        (root / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    return root


def test_schema_failure_preserves_existing_output(tmp_path, monkeypatch):
    import src.output_builder as builder
    dest = tmp_path / "old.json"
    dest.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(builder, "validate_json_against_schema", lambda *args: ["invalid nested field"])
    with pytest.raises(ValueError, match="Output validation failed"):
        run_pipeline("example", dest)
    assert dest.read_text() == "preserve"


def test_missing_schema_and_input_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        schema_adapter.load_schema(tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="Missing required input"):
        load_dataset(tmp_path)


def test_stdlib_schema_checks_nested_enums_oneof_and_evidence(monkeypatch):
    _, payload = run_pipeline("example")
    monkeypatch.setattr(schema_adapter, "_HAS_JSONSCHEMA", False)
    schema = schema_adapter.load_schema()
    assert schema_adapter.validate_json_against_schema(payload, schema) == []
    for mutate in (lambda p: p["people"][0].update(circle="unknown"),
                   lambda p: p["chats"][0].update(person="p", organisation="o"),
                   lambda p: p["goals"][0].update(evidence=[]),
                   lambda p: p["people"][0].update(unknown=True)):
        broken = copy.deepcopy(payload)
        mutate(broken)
        assert schema_adapter.validate_json_against_schema(broken, schema)


def test_serialized_references_catch_cross_chat_and_missing_people():
    _, payload = run_pipeline("example")
    index = ConversationIndex(load_dataset("example"))
    assert validate_references(payload, index) == []
    payload["people"].pop()
    payload["open_loops"][0]["source_msg"] = "absent"
    errors = validate_references(payload, index)
    assert any("each non-ME" in e for e in errors)
    assert any("originating chat" in e for e in errors)


def test_all_history_loops_remain_available(tmp_path):
    data = fixture(tmp_path, [("ME", "I will send the invoice tomorrow."), ("p", "Good morning")],
                   dates=["2026-08-01T10:00:00+05:30", "2026-10-02T10:00:00+05:30"])
    _, payload = run_pipeline(data)
    assert any(l["source_msg"] == "s0" for l in payload["open_loops"])


@pytest.mark.parametrize("body", ["Can u send me the furniture link?", "Could u share the receipt?",
    "Need your help to update the registration.", "We will send the contract tomorrow.",
    "Will complete the accounts by Saturday."])
def test_informal_requests_and_collective_promises(tmp_path, body):
    data = fixture(tmp_path, [("p", body)])
    _, payload = run_pipeline(data)
    assert any(l["source_msg"] == "s0" for l in payload["open_loops"])


def client_with_transport(monkeypatch, tmp_path, handler, **kwargs):
    monkeypatch.setenv("LLM_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    client = LLMClient(base_url="https://endpoint.invalid/api/v1", api_key="test-only", model="test-model", **kwargs)
    client._http_client.close()
    client._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def completion(content='{"ok":true}', usage=True):
    payload = {"choices": [{"message": {"content": content}}]}
    if usage:
        payload["usage"] = {"prompt_tokens": 13, "completion_tokens": 7}
    return httpx.Response(200, json=payload)


def test_http_auth_url_usage_cache_and_capability_fallback(monkeypatch, tmp_path):
    requests = []
    def handler(req):
        assert req.url == "https://endpoint.invalid/api/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer test-only"
        payload = json.loads(req.content)
        requests.append(payload)
        if "response_format" in payload:
            return httpx.Response(400, json={"error": "response_format unsupported"})
        return completion('```json\n{"ok":true}\n```')
    client = client_with_transport(monkeypatch, tmp_path, handler)
    assert client.complete_json("smoke", "JSON", "hello") == {"ok": True}
    assert client.complete_json("smoke", "JSON", "hello") == {"ok": True}
    snap = client.usage_snapshot()
    assert len(requests) == 2 and snap["calls"] == 1 and snap["cache_hits"] == 1
    assert snap["prompt_tokens"] == 13 and snap["completion_tokens"] == 7
    assert len((tmp_path / "usage.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_http_failures_fallback_and_retry_count(monkeypatch, tmp_path, status):
    requests = []
    def handler(req):
        requests.append(req)
        return httpx.Response(status, json={"error": "test failure"})
    monkeypatch.setattr("src.llm_client.time.sleep", lambda seconds: None)
    client = client_with_transport(monkeypatch, tmp_path, handler, max_retries=1)
    assert client.complete_json("stage", "JSON", "hello") is None
    assert len(requests) == (2 if status in (429, 500) else 1)


def test_deadline_rechecked_after_capability_failure(monkeypatch, tmp_path):
    requests = []
    def handler(req):
        requests.append(req)
        client._created_at -= 2000
        return httpx.Response(400, json={"error": "response_format unsupported"})
    client = client_with_transport(monkeypatch, tmp_path, handler)
    assert client.complete_json("stage", "JSON", "hello") is None
    assert len(requests) == 1


def test_concurrent_budget_reservations_and_missing_usage(monkeypatch, tmp_path):
    requests = []
    def handler(req):
        requests.append(req)
        return completion(usage=False)
    client = client_with_transport(monkeypatch, tmp_path, handler)
    client.max_cost_usd = 0.00001
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: client.complete_json("stage", "JSON", str(i), max_tokens=100), range(4)))
    assert results == [None] * 4 and not requests
    client.max_cost_usd = 1
    assert client.complete_json("stage", "JSON", "normal") == {"ok": True}
    assert client.usage_snapshot()["estimated_usage_calls"] == 1


def test_redirects_are_not_followed(monkeypatch, tmp_path):
    from src.llm_client import _NoRedirect
    requests = []
    def handler(req):
        requests.append(req)
        return httpx.Response(302, headers={"Location": "https://different.invalid"})
    client = client_with_transport(monkeypatch, tmp_path, handler, max_retries=0)
    assert client.complete_json("stage", "JSON", "hello") is None
    assert len(requests) == 1
    assert _NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://different.invalid") is None


def test_transient_inflight_402_retries_but_permanent_402_does_not(monkeypatch, tmp_path):
    requests = []
    def handler(req):
        requests.append(req)
        if len(requests) == 1:
            return httpx.Response(402, headers={"Retry-After": "0"}, json={"error": {
                "message": "temporary credit hold", "metadata": {"limit_source": "openrouter_in_flight_budget"}}})
        return completion()
    client = client_with_transport(monkeypatch, tmp_path, handler, max_retries=1)
    assert client.complete_json("stage", "JSON", "hello") == {"ok": True}
    assert len(requests) == 2 and client._provider_limited
    client._http_client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(
        402, json={"error": {"metadata": {"limit_source": "openrouter_key_limit"}}})))
    assert client.complete_json("stage", "JSON", "other") is None
    assert client.usage_snapshot()["http_requests"] == 3


def test_no_completion_correction_without_evidence(tmp_path):
    data = fixture(tmp_path, [("ME", "I will send the invoice tomorrow."), ("p", "Ok")])
    output, _ = run_pipeline(data)
    with pytest.raises(ValueError, match="closure_message_id"):
        apply_corrections(output, {"open_loops": {"s0": {"status": "done"}}}, ConversationIndex(load_dataset(data)))


def test_unrelated_later_dm_response_cannot_answer_old_question(tmp_path):
    data = fixture(tmp_path, [("p", "Should we repair the balcony railing?"),
                             ("p", "Let me know when you decide."), ("ME", "Train ticket.")],
                   dates=["2026-10-02T10:00:00+05:30", "2026-10-02T10:01:00+05:30", "2026-10-02T11:00:00+05:30"])
    _, payload = run_pipeline(data)
    assert next(l for l in payload["open_loops"] if l["source_msg"] == "s0")["status"] == "open"
    stages = []
    def mock(stage, system, user):
        stages.append((stage, user))
        return {}
    run_pipeline(data, mock_responder=mock)
    assert not any(stage == "temporal_slow_verify" and "[s0 |" in user for stage, user in stages)


def test_llm_cannot_close_on_unquoted_bare_ack(tmp_path):
    data = fixture(tmp_path, [("p", "Can you send the contract tomorrow?"), ("ME", "Ok"),
                             ("ME", "I checked my calendar.")])
    output, payload = run_pipeline(data, mock_responder=lambda stage, system, user:
        {"status": "done", "closure_message_id": "s1", "owner": "ME", "confidence": 0.99}
        if stage == "temporal_slow_verify" else {})
    assert next(l for l in payload["open_loops"] if l["source_msg"] == "s0")["status"] == "open"


def test_saved_name_alias_preserves_kinship_and_trading_role(tmp_path):
    from src.relationship_inference import build_contact_dossiers, _score_dossier_tier0
    data = fixture(tmp_path, [("p", "Hello"), ("p", "Good morning"), ("p", "Hi")])
    index = ConversationIndex(load_dataset(data))
    index.person_names["p"]["Mira Chachi"] = 1
    dossier = build_contact_dossiers(index)["p"]
    scores, _ = _score_dossier_tier0(dossier, index, set())
    assert max(scores, key=scores.get) == "relative"
    index.person_names["p"].clear()
    index.person_names["p"]["Cedar Trading Company"] = 3
    dossier = build_contact_dossiers(index)["p"]
    scores, _ = _score_dossier_tier0(dossier, index, set())
    assert max(scores, key=scores.get) == "vendor"


def test_cli_invalid_schema_nonzero(tmp_path):
    result = subprocess.run([sys.executable, "-m", "src.cli", "example", str(tmp_path / "out.json"),
                             "--schema", str(tmp_path / "absent.json")], capture_output=True, text=True)
    assert result.returncode == 1 and "Failed:" in result.stderr
    assert not (tmp_path / "out.json").exists()


def test_live_benchmark_rejects_fixture_alias_before_network(tmp_path):
    from scripts.live_evaluation import validate_official_input
    data = fixture(tmp_path, [("ME", "Hello")])
    validate_official_input(data)
    chat = json.loads((data / "chats.jsonl").read_text())
    chat["participant_ids"] = chat.pop("participants")
    (data / "chats.jsonl").write_text(json.dumps(chat), encoding="utf-8")
    with pytest.raises(ValueError, match="participants field"):
        validate_official_input(data)
