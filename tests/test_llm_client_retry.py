"""Tests for 429 retry logic and new return shape of call_openrouter."""
from unittest.mock import MagicMock, patch

import pytest

from src.llm.client import call_openrouter


def _make_response(status_code: int, body: dict | None = None,
                   headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    if status_code == 200:
        resp.raise_for_status = MagicMock()
        resp.json.return_value = body or {}
    else:
        import requests

        def _raise():
            raise requests.HTTPError(response=resp)

        resp.raise_for_status = _raise
        resp.json.return_value = body or {}
    return resp


def _ok_body(text: str = "hello") -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_call_openrouter_returns_dict_shape():
    """Successful call returns dict with raw, usage, latency_ms."""
    ok_resp = _make_response(200, _ok_body("result"))
    with patch("src.llm.client.requests.post", return_value=ok_resp):
        result = call_openrouter(
            [{"role": "user", "content": "hi"}],
            model="openai/gpt-4o-mini",
            api_key="sk-test",
        )
    assert isinstance(result, dict)
    assert result["raw"] == "result"
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0


def test_429_retry_with_retry_after_header():
    """One 429 with Retry-After: 0 then 200 — should succeed on second attempt."""
    rate_limit_resp = _make_response(429, headers={"Retry-After": "0"})
    ok_resp = _make_response(200, _ok_body("after_retry"))

    call_count = [0]

    def fake_post(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return rate_limit_resp
        return ok_resp

    with patch("src.llm.client.requests.post", side_effect=fake_post), \
         patch("src.llm.client.time.sleep") as mock_sleep:
        result = call_openrouter(
            [{"role": "user", "content": "hi"}],
            model="openai/gpt-4o-mini",
            api_key="sk-test",
        )

    assert result["raw"] == "after_retry"
    assert call_count[0] == 2
    mock_sleep.assert_called_once_with(0.0)


def test_429_retry_exponential_backoff_no_header():
    """429 with no Retry-After header uses exponential backoff: 1, 2, 4."""
    rate_limit_resp = _make_response(429)
    ok_resp = _make_response(200, _ok_body("ok"))

    responses = [rate_limit_resp, rate_limit_resp, ok_resp]
    idx = [0]

    def fake_post(*args, **kwargs):
        r = responses[idx[0]]
        idx[0] += 1
        return r

    with patch("src.llm.client.requests.post", side_effect=fake_post), \
         patch("src.llm.client.time.sleep") as mock_sleep:
        result = call_openrouter(
            [{"role": "user", "content": "hi"}],
            model="openai/gpt-4o-mini",
            api_key="sk-test",
        )

    assert result["raw"] == "ok"
    assert mock_sleep.call_count == 2
    assert mock_sleep.call_args_list[0][0][0] == 1  # first backoff
    assert mock_sleep.call_args_list[1][0][0] == 2  # second backoff


def test_429_exhausted_raises():
    """Four consecutive 429s (3 retries + 1 give-up) should raise HTTPError."""
    import requests as req_lib

    rate_limit_resp = _make_response(429)

    def fake_post(*args, **kwargs):
        return rate_limit_resp

    with patch("src.llm.client.requests.post", side_effect=fake_post), \
         patch("src.llm.client.time.sleep"):
        with pytest.raises(req_lib.HTTPError):
            call_openrouter(
                [{"role": "user", "content": "hi"}],
                model="openai/gpt-4o-mini",
                api_key="sk-test",
            )
