"""The unit/regression suite must never spend real API credits."""
import pytest


@pytest.fixture(autouse=True)
def isolate_model_environment(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
