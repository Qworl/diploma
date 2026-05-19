"""LLM HTTP clients: OpenRouter + Ollama backends."""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)


def call_openrouter(messages: list[dict], model: str, api_key: str,
                    *, enforce_json: bool = True, max_tokens: int = 2048) -> dict:
    """Call OpenRouter API. Returns dict with keys: raw, usage, latency_ms.

    - raw: response text (str)
    - usage: dict from provider's usage field ({prompt_tokens, completion_tokens, ...}),
             or {} if not present
    - latency_ms: wall-clock time of the HTTP call in milliseconds (float)

    enforce_json=True passes response_format={"type": "json_object"} — supported by
    Anthropic/OpenAI/Gemini models on OpenRouter, ignored by others (no harm).
    max_tokens defaults to 2048 to accommodate reasoning models (gpt-oss, o-series)
    that consume tokens in a hidden <think> chain before writing the visible response.

    Optional env-driven config (для reasoning models или slow providers):
    - OPENROUTER_PROVIDER: comma-separated provider names (e.g. "google-vertex,groq,fireworks")
      → maps to payload["provider"]["order"], with allow_fallbacks=true
    - OPENROUTER_REASONING_EFFORT: "low" | "medium" | "high"
      → maps to payload["reasoning"]["effort"] for reasoning models (gpt-oss, o-series)
    - OPENROUTER_TIMEOUT: seconds (default 30) — increase for reasoning models

    429 retry: exponential backoff 1s → 2s → 4s, then raises. Honors Retry-After header.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if enforce_json:
        payload["response_format"] = {"type": "json_object"}

    provider_env = os.environ.get("OPENROUTER_PROVIDER", "").strip()
    if provider_env:
        payload["provider"] = {
            "order": [p.strip() for p in provider_env.split(",") if p.strip()],
            "allow_fallbacks": True,
        }

    reasoning_effort = os.environ.get("OPENROUTER_REASONING_EFFORT", "").strip()
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    timeout = int(os.environ.get("OPENROUTER_TIMEOUT", "30"))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    backoffs = [1, 2, 4]
    json_fallback_tried = False
    for attempt, wait in enumerate(backoffs + [None]):
        t0 = time.perf_counter()
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Some models (e.g. gpt-5.5) return 400 when response_format=json_object
        # is passed but the model doesn't support it. Retry once without it.
        if resp.status_code == 400 and enforce_json and not json_fallback_tried:
            logger.warning(
                "400 with enforce_json=True for model=%s — retrying without response_format",
                model,
            )
            payload.pop("response_format", None)
            json_fallback_tried = True
            continue

        if resp.status_code == 429:
            if wait is None:
                # Exhausted all retries — raise
                resp.raise_for_status()
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    sleep_sec = float(retry_after)
                except ValueError:
                    sleep_sec = wait
            else:
                sleep_sec = wait
            logger.warning(
                "429 rate-limit on attempt %d — sleeping %.1fs before retry",
                attempt + 1, sleep_sec,
            )
            time.sleep(sleep_sec)
            continue

        resp.raise_for_status()
        body = resp.json()
        raw = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        return {"raw": raw, "usage": usage, "latency_ms": latency_ms}


def call_ollama(messages: list[dict], model: str, base_url: str = "http://localhost:11434",
                *, enforce_json: bool = True, max_tokens: int = 2048) -> str:
    """Call Ollama API. Returns raw response text.

    enforce_json=True sets format="json" — Ollama then constrains output to valid JSON.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    if enforce_json:
        payload["format"] = "json"
    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
