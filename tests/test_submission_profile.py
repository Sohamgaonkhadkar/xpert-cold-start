"""The graded command must reproduce the verified profile with grader credentials present."""
import json
from datetime import datetime, timedelta

import pytest

from src.cli import main
from src.intervention_policy import _estimate_action_posteriors, _select_bayes_optimal_action
from src.llm_client import LLMClient
from src.models import OpenLoopPrediction


def configured_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-local-only")
    monkeypatch.setenv("LLM_MODEL", "test-model")


def test_graded_default_is_network_free_even_with_credentials(tmp_path, monkeypatch):
    configured_endpoint(monkeypatch)
    def refuse_network(*args, **kwargs):
        pytest.fail("Deterministic submission tried to call the model")
    monkeypatch.setattr(LLMClient, "_post_http", refuse_network)
    output, telemetry = tmp_path / "result.json", tmp_path / "telemetry.json"
    assert main(["example", str(output), "--telemetry", str(telemetry), "--quiet"]) == 0
    report = json.loads(telemetry.read_text())
    assert report["model_used"] == "deterministic-tier0"
    assert report["llm_calls"] == report["total_tokens"] == 0
    assert output.exists()


def test_hybrid_mode_explicitly_uses_configured_transport(tmp_path, monkeypatch):
    configured_endpoint(monkeypatch)
    calls = []
    def local_transport(self, payload):
        calls.append(payload["model"])
        return {"choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    monkeypatch.setattr(LLMClient, "_post_http", local_transport)
    assert main(["example", str(tmp_path / "hybrid.json"), "--mode", "hybrid", "--quiet"]) == 0
    assert calls and set(calls) == {"test-model"}


@pytest.mark.parametrize("circle,due,waiting,body,expected", [
    ("vendor", None, True, "Send the invoice", "leave"),
    ("vendor", "2026-11-10", True, "Send the invoice", "remind"),
    ("vendor", "2026-11-10", False, "Send the invoice", "leave"),
    ("colleague", "2026-11-14", True, "Send the report", "leave"),
    ("staff", "2026-11-10", True, "Collect the receipt", "remind"),
    ("staff", None, True, "Collect the receipt", "leave"),
    ("spouse", "2026-11-10", True, "Handle my own plans", "leave"),
    ("vendor", "2026-11-10", True, "Send the invoice; don't remind", "leave"),
])
def test_other_owner_intervention_follows_whose_move_it_is(circle, due, waiting, body, expected):
    anchor = datetime(2026, 11, 12, 12)
    loop = OpenLoopPrediction(id="test", title=body, type="followup", owner="p", chat_id="c", message_id="m", due_date=due)
    weights = _estimate_action_posteriors(loop, body, circle, anchor - timedelta(days=20), anchor, waiting_for_me=waiting)
    assert _select_bayes_optimal_action(weights) == expected


def test_polite_apology_does_not_make_me_owed_spouse_task_private():
    anchor = datetime(2026, 11, 12, 12)
    text = "Sorry, I will call the plumber tomorrow"
    loop = OpenLoopPrediction(id="test", title=text, type="commitment", owner="ME", chat_id="c", message_id="m")
    assert _select_bayes_optimal_action(_estimate_action_posteriors(loop, text, "spouse", anchor, anchor)) == "remind"
