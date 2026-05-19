"""Tests for src.manual_label.off_fetcher."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.manual_label.off_fetcher import fetch_off_product, OFFFetchError


def _mock_response(status_code: int, json_body: dict | None = None):
    m = MagicMock()
    m.status_code = status_code
    if json_body is not None:
        m.json.return_value = json_body
    return m


def test_fetch_caches_to_disk(tmp_path):
    code = "8000139007057"
    payload = {"status": 1, "product": {"product_name": "Penne", "code": code}}
    with patch("src.manual_label.off_fetcher.requests.get",
               return_value=_mock_response(200, payload)) as get:
        result = fetch_off_product(code, cache_dir=tmp_path)
        assert result == payload["product"]
        # Cache file written
        assert (tmp_path / f"{code}.json").exists()
        get.assert_called_once()


def test_fetch_uses_cache_on_second_call(tmp_path):
    code = "8000139007057"
    payload = {"status": 1, "product": {"product_name": "Penne", "code": code}}
    with patch("src.manual_label.off_fetcher.requests.get",
               return_value=_mock_response(200, payload)) as get:
        fetch_off_product(code, cache_dir=tmp_path)
        fetch_off_product(code, cache_dir=tmp_path)
        # Only one HTTP call total
        assert get.call_count == 1


def test_fetch_404_raises(tmp_path):
    code = "0000000000000"
    with patch("src.manual_label.off_fetcher.requests.get",
               return_value=_mock_response(404)):
        with pytest.raises(OFFFetchError, match="404"):
            fetch_off_product(code, cache_dir=tmp_path)


def test_fetch_status_zero_raises(tmp_path):
    """OFF returns 200 with status:0 when product not in DB."""
    code = "0000000000000"
    payload = {"status": 0, "status_verbose": "product not found"}
    with patch("src.manual_label.off_fetcher.requests.get",
               return_value=_mock_response(200, payload)):
        with pytest.raises(OFFFetchError, match="not found"):
            fetch_off_product(code, cache_dir=tmp_path)
