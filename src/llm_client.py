"""Compatibility-friendly OpenAI-compatible LLM client with retries, bounded concurrency, JSON repair, and stage-level token/cost telemetry."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import httpx  # type: ignore

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    httpx = None
    _HAS_HTTPX = False


# Approximate public pricing ($ per 1M tokens: (input_usd, output_usd)) for cost telemetry
MODEL_PRICING_PER_1M: List[Tuple[str, Tuple[float, float]]] = [
    ("gpt-5-nano", (0.05, 0.40)),
    ("gpt-5-mini", (0.25, 2.00)),
    ("gpt-5", (1.25, 10.00)),
    ("o4-mini", (1.10, 4.40)),
    ("o3-mini", (1.10, 4.40)),
    ("gpt-4o-mini", (0.15, 0.60)),
    ("gpt-4.1-mini", (0.40, 1.60)),
    ("gpt-4.1-nano", (0.10, 0.40)),
    ("gpt-4.1", (2.00, 8.00)),
    ("gpt-4o", (2.50, 10.00)),
    ("gemini-2.5-flash-lite", (0.10, 0.40)),
    ("gemini-2.5-flash", (0.15, 0.60)),
    ("gemini-2.0-flash", (0.10, 0.40)),
    ("claude-3-5-haiku", (0.80, 4.00)),
    ("claude-haiku-4-5", (1.00, 5.00)),
    ("claude-sonnet", (3.00, 15.00)),
    ("deepseek-chat", (0.27, 1.10)),
    ("llama-3.3-70b", (0.59, 0.79)),
    ("qwen", (0.20, 0.60)),
]

REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5", "deepseek-r1", "qwq")

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    env_in = os.environ.get("LLM_INPUT_COST_PER_1M")
    env_out = os.environ.get("LLM_OUTPUT_COST_PER_1M")
    if env_in and env_out:
        try:
            in_rate, out_rate = float(env_in), float(env_out)
            return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0
        except ValueError:
            pass
    m_low = (model or "").lower()
    in_rate, out_rate = (0.30, 1.20)  # conservative midrange default
    for key, rates in MODEL_PRICING_PER_1M:
        if key in m_low:
            in_rate, out_rate = rates
            break
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0


def extract_json_payload(raw_text: str) -> Any:
    """Safely parse JSON from LLM output even with <think> tags, fences, or prose."""
    if not raw_text:
        raise ValueError("Empty LLM response")
    cleaned = THINK_BLOCK_RE.sub("", raw_text).strip()

    fence_match = CODE_FENCE_RE.search(cleaned)
    candidates: List[str] = []
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    candidates.append(cleaned)

    # Extract balanced outer {...} or [...]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        if start != -1:
            depth = 0
            in_str = False
            esc = False
            for idx in range(start, len(cleaned)):
                ch = cleaned[idx]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                        if depth == 0:
                            candidates.append(cleaned[start : idx + 1])
                            break

    last_err: Optional[Exception] = None
    for cand in candidates:
        for attempt in (cand, TRAILING_COMMA_RE.sub(r"\1", cand)):
            try:
                return json.loads(attempt)
            except Exception as exc:
                last_err = exc
    raise ValueError(f"Could not parse JSON from response: {last_err}")


@dataclass
class StageUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    """Thread-safe OpenAI-compatible chat completion client with stage token accounting."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_concurrency: int = 6,
        timeout_sec: float = 120.0,
        max_retries: int = 3,
        mock_responder: Optional[Callable[[str, str, str], Dict[str, Any]]] = None,
    ) -> None:
        env_llm_base = os.environ.get("LLM_BASE_URL", "").strip()
        env_oai_base = os.environ.get("OPENAI_BASE_URL", "").strip()
        env_llm_key = os.environ.get("LLM_API_KEY", "").strip()
        env_oai_key = os.environ.get("OPENAI_API_KEY", "").strip()

        raw_base = (
            base_url
            if base_url is not None
            else (env_llm_base or env_oai_base)
        )
        self.api_key = (
            api_key
            if api_key is not None
            else (env_llm_key or env_oai_key)
        )
        self.model = (
            model
            if model is not None
            else (
                os.environ.get("LLM_MODEL", "").strip()
                or os.environ.get("OPENAI_MODEL", "").strip()
                or "gpt-4o-mini"
            )
        )
        # Never send LLM_API_KEY anywhere except an explicit LLM_BASE_URL; only default for OPENAI_API_KEY
        if not raw_base and env_oai_key and not env_llm_key:
            raw_base = "https://api.openai.com/v1"
        self.base_url = raw_base.rstrip("/")
        self.max_concurrency = max(1, int(max_concurrency))
        self.timeout_sec = float(timeout_sec)
        self.max_retries = max(1, int(max_retries))
        self.mock_responder = mock_responder
        self.max_wall_sec = float(os.environ.get("LLM_MAX_WALL_SEC", "1380"))
        self.max_cost_usd = float(os.environ.get("LLM_MAX_COST_USD", "2.50"))
        self._created_at = time.monotonic()

        m_leaf = self.model.lower().split("/")[-1]
        is_reasoning = any(m_leaf.startswith(p) for p in REASONING_MODEL_PREFIXES)

        self._lock = threading.Lock()
        self._cache: Dict[str, Any] = {}
        self._stage_usage: Dict[str, StageUsage] = {}
        self._supports_json_mode: bool = True
        self._supports_temperature: bool = not is_reasoning
        self._use_max_completion_tokens: bool = is_reasoning
        self._is_reasoning_model: bool = is_reasoning
        self._http_client: Any = (
            httpx.Client(timeout=self.timeout_sec)
            if (_HAS_HTTPX and httpx is not None and self.api_key and self.base_url)
            else None
        )

    @property
    def is_available(self) -> bool:
        return self.mock_responder is not None or bool(self.api_key and self.base_url)

    def _endpoint_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _record_usage(self, stage: str, prompt_toks: int, completion_toks: int) -> None:
        with self._lock:
            su = self._stage_usage.setdefault(stage, StageUsage())
            su.calls += 1
            su.prompt_tokens += max(0, int(prompt_toks))
            su.completion_tokens += max(0, int(completion_toks))

    def usage_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            stages = {
                k: {
                    "calls": v.calls,
                    "prompt_tokens": v.prompt_tokens,
                    "completion_tokens": v.completion_tokens,
                    "total_tokens": v.prompt_tokens + v.completion_tokens,
                }
                for k, v in self._stage_usage.items()
            }
            p_tot = sum(v.prompt_tokens for v in self._stage_usage.values())
            c_tot = sum(v.completion_tokens for v in self._stage_usage.values())
            calls = sum(v.calls for v in self._stage_usage.values())
            chits = getattr(self, "_cache_hits", 0)
        cost = estimate_cost_usd(self.model, p_tot, c_tot) if calls > 0 else 0.0
        return {
            "model": self.model if self.is_available else "deterministic-tier0",
            "calls": calls,
            "cache_hits": chits,
            "prompt_tokens": p_tot,
            "completion_tokens": c_tot,
            "total_tokens": p_tot + c_tot,
            "estimated_cost_usd": round(cost, 6),
            "stages": stages,
        }

    def _append_usage_log(self, model_name: str, data: Dict[str, Any], elapsed_sec: float) -> None:
        usage = data.get("usage") or {}
        log_path = os.environ.get("LLM_USAGE_LOG", "usage.jsonl")
        try:
            with self._lock:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "model": model_name,
                                "prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens"),
                                "seconds": round(elapsed_sec, 2),
                            }
                        )
                        + "\n"
                    )
        except OSError:
            pass

    def _post_http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._endpoint_url()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body_bytes = json.dumps(payload).encode("utf-8")
        started = time.time()

        if self._http_client is not None:
            resp = self._http_client.post(url, headers=headers, content=body_bytes)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP_{resp.status_code}: {resp.text[:400]}")
            data = resp.json()
            self._append_usage_log(str(payload.get("model") or self.model), data, time.time() - started)
            return data

        if _HAS_HTTPX and httpx is not None:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(url, headers=headers, content=body_bytes)
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP_{resp.status_code}: {resp.text[:400]}")
                data = resp.json()
                self._append_usage_log(str(payload.get("model") or self.model), data, time.time() - started)
                return data

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                self._append_usage_log(str(payload.get("model") or self.model), data, time.time() - started)
                return data
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"HTTP_{exc.code}: {err_body}") from exc

    def complete_json(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1600,
    ) -> Optional[Any]:
        """Execute a single JSON-mode chat completion with caching, fallback, and repair."""
        if not self.is_available:
            return None

        cache_key = hashlib.sha256(
            f"{self.model}|{stage}|{system_prompt}|{user_prompt}".encode("utf-8")
        ).hexdigest()
        with self._lock:
            if cache_key in self._cache:
                self._cache_hits = getattr(self, "_cache_hits", 0) + 1
                return self._cache[cache_key]

        if self.mock_responder is not None:
            t_mock = time.time()
            result = self.mock_responder(stage, system_prompt, user_prompt)
            est_in = max(1, (len(system_prompt) + len(user_prompt)) // 4)
            est_out = max(1, len(json.dumps(result)) // 4)
            self._record_usage(stage, est_in, est_out)
            if os.environ.get("LLM_USAGE_LOG"):
                self._append_usage_log(
                    self.model,
                    {"usage": {"prompt_tokens": est_in, "completion_tokens": est_out}},
                    time.time() - t_mock,
                )
            with self._lock:
                self._cache[cache_key] = result
            return result

        # Enforce wall-clock and API cost budget guards so container never times out or overspends
        if (time.monotonic() - self._created_at) > self.max_wall_sec:
            return None
        snap = self.usage_snapshot()
        if snap["estimated_cost_usd"] >= self.max_cost_usd:
            return None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self.max_retries + 1):
            eff_max = max_tokens * 3 if self._is_reasoning_model else max_tokens
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if self._use_max_completion_tokens:
                payload["max_completion_tokens"] = eff_max
            else:
                payload["max_tokens"] = eff_max
            if self._supports_temperature:
                payload["temperature"] = 0.0
            if self._supports_json_mode:
                payload["response_format"] = {"type": "json_object"}

            try:
                data = self._post_http(payload)
                usage = data.get("usage") or {}
                p_toks = usage.get("prompt_tokens")
                c_toks = usage.get("completion_tokens")
                choices = data.get("choices") or []
                content = ""
                if choices:
                    msg_obj = choices[0].get("message") or {}
                    raw_content = msg_obj.get("content")
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif isinstance(raw_content, list):
                        content = "\n".join(
                            str(part.get("text", ""))
                            for part in raw_content
                            if isinstance(part, dict) and part.get("type") in (None, "text", "output_text")
                        )
                if p_toks is None:
                    p_toks = max(1, (len(system_prompt) + len(user_prompt)) // 4)
                if c_toks is None:
                    c_toks = max(1, len(content) // 4)
                self._record_usage(stage, int(p_toks), int(c_toks))

                parsed = extract_json_payload(content)
                with self._lock:
                    self._cache[cache_key] = parsed
                return parsed
            except RuntimeError as err:
                msg = str(err)
                # Provider capability downgrade on HTTP 400
                if "HTTP_400" in msg:
                    if ("max_tokens" in msg or "max_completion_tokens" in msg) and not self._use_max_completion_tokens:
                        self._use_max_completion_tokens = True
                        self._is_reasoning_model = True
                        continue
                    if "temperature" in msg and self._supports_temperature:
                        self._supports_temperature = False
                        continue
                    if "response_format" in msg and self._supports_json_mode:
                        self._supports_json_mode = False
                        continue
                if attempt + 1 < self.max_retries and any(
                    code in msg for code in ("HTTP_429", "HTTP_500", "HTTP_502", "HTTP_503", "HTTP_504")
                ):
                    backoff = (0.8 * (2**attempt)) + random.uniform(0.05, 0.35)
                    time.sleep(backoff)
                    continue
                return None
            except Exception:
                if attempt + 1 < self.max_retries:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                return None
        return None

    def complete_json_batch(
        self,
        stage: str,
        tasks: Sequence[Tuple[str, str, str]],
        max_tokens: int = 1600,
    ) -> Dict[str, Any]:
        """Run (task_key, system_prompt, user_prompt) items with bounded thread concurrency."""
        if not self.is_available or not tasks:
            return {}
        results: Dict[str, Any] = {}
        workers = min(self.max_concurrency, len(tasks))
        if workers <= 1:
            for key, sys_p, usr_p in tasks:
                res = self.complete_json(stage, sys_p, usr_p, max_tokens=max_tokens)
                if res is not None:
                    results[key] = res
            return results

        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_map = {
                pool.submit(
                    self.complete_json, stage, sys_p, usr_p, max_tokens
                ): key
                for key, sys_p, usr_p in tasks
            }
            for fut in as_completed(fut_map):
                key = fut_map[fut]
                try:
                    val = fut.result()
                    if val is not None:
                        results[key] = val
                except Exception:
                    continue
        return results
