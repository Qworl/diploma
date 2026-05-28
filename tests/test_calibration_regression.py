"""Регресс-тест ECE-калибровки моделей.

Цель: переводит «целесообразно добавить регресс-тест ECE» из Future Work
в инфраструктурный факт (см. тикет docs/po/tickets/2026-05-28-ece-regression-test.md
и §6 Conclusion thesis).

Дизайн (snapshot-style регрессия):

1. **Baseline**: фиксированный snapshot в `tests/data/ece_baseline.json`
   с ECE per (prefix, attr) на момент последнего accepted retrain.
   Снимок версионируется в git вместе с моделями — приёмка изменения
   ECE происходит явно через обновление этого файла.
2. **Current**: live ECE из `models/{prefix}_{attr}_calibration.json`
   (перезаписывается каждым `src.pipeline.ml.train`).
3. **Assert**: `current_ECE - baseline_ECE <= tolerance` (по умолчанию 0,01).

Дополнительные проверки:
- **Integrity**: рекомпьют ECE из bin-breakdown (count/acc/mean_conf) даёт
  ту же цифру, что и поле `ece_raw`/`ece_calibrated` — ловит порчу
  файла или арифметическую несогласованность.
- **Sanity cap**: ECE ≤ SANITY_CAP (0,13) для любого атрибута —
  ловит совсем поломанные калибровки (отсечка чуть выше текущего
  худшего значения pasta_shape ECE_cal ≈ 0,115).

Если артефакт `models/*_calibration.json` отсутствует локально (например,
после `git clean -fdx models/`) — тест пропускает соответствующую запись
с pytest.skip() вместо падения; это позволяет работать без обученных
моделей. Если baseline-файл отсутствует — тест fail (он коммитится в git).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
BASELINE_PATH = Path(__file__).resolve().parent / "data" / "ece_baseline.json"

# Соответствует упоминанию в §6 Conclusion: «ухудшение ECE более чем на 0,01».
DEFAULT_TOLERANCE = 0.01
# Sanity cap: ECE выше этого порога означает фундаментально сломанную
# калибровку (худшее текущее значение — pasta_shape ECE_cal ≈ 0,115).
SANITY_CAP = 0.13


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.fail(
            f"Baseline ECE snapshot не найден: {BASELINE_PATH}. "
            "Этот файл коммитится в git и обновляется только при осознанном "
            "принятии нового baseline после retrain."
        )
    with BASELINE_PATH.open() as f:
        return json.load(f)


def _load_calibration(prefix: str, attr: str) -> dict | None:
    path = MODELS_DIR / f"{prefix}_{attr}_calibration.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _ece_from_bins(bins: list[dict]) -> float | None:
    """Recompute ECE-max-prob style из stored bin-breakdown.

    Зеркалит `src.pipeline.ml.train.compute_ece`: бин-веса по count,
    суммируем |acc - mean_conf|. Возвращает None если все бины пусты.
    """
    if not bins:
        return None
    total = sum(b.get("count", 0) for b in bins)
    if total == 0:
        return None
    ece = 0.0
    for b in bins:
        cnt = b.get("count", 0)
        if cnt == 0:
            continue
        acc = b.get("acc")
        conf = b.get("mean_conf")
        if acc is None or conf is None:
            continue
        ece += (cnt / total) * abs(acc - conf)
    return ece


def _baseline_cells() -> list[tuple[str, str, dict]]:
    payload = _load_baseline()
    cells: list[tuple[str, str, dict]] = []
    for prefix, attrs in payload["baselines"].items():
        for attr, ece_dict in attrs.items():
            cells.append((prefix, attr, ece_dict))
    return cells


@pytest.fixture(scope="module")
def baseline_payload() -> dict:
    return _load_baseline()


def test_baseline_file_well_formed(baseline_payload):
    """Snapshot-файл должен иметь schema + tolerance + хотя бы один baseline."""
    assert "__tolerance__" in baseline_payload
    assert "baselines" in baseline_payload
    assert len(baseline_payload["baselines"]) > 0
    tol = baseline_payload["__tolerance__"]
    assert 0 < tol <= 0.05, f"Подозрительный tolerance: {tol}"


@pytest.mark.parametrize(
    "prefix,attr,baseline_ece",
    [(p, a, e) for p, a, e in _baseline_cells()],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_ece_regression(prefix: str, attr: str, baseline_ece: dict):
    """Главный тест: current ECE не должен превышать baseline + tolerance."""
    payload = _load_baseline()
    tolerance = payload.get("__tolerance__", DEFAULT_TOLERANCE)

    current = _load_calibration(prefix, attr)
    if current is None:
        pytest.skip(
            f"Артефакт {prefix}_{attr}_calibration.json отсутствует — "
            "тест пропущен (видимо, локальная обрезка моделей)."
        )

    failures: list[str] = []
    for key in ("ece_raw", "ece_calibrated"):
        base = baseline_ece.get(key)
        curr = current.get(key)
        if base is None and curr is None:
            continue
        if base is None and curr is not None:
            failures.append(
                f"  {key}: baseline=None, current={curr:.4f} "
                "(новый атрибут — обнови baseline сознательно)"
            )
            continue
        if curr is None and base is not None:
            failures.append(
                f"  {key}: baseline={base:.4f}, current=None "
                "(калибровка пропала — проверь train.py)"
            )
            continue
        delta = curr - base
        if delta > tolerance:
            failures.append(
                f"  {key}: baseline={base:.4f}, current={curr:.4f}, "
                f"Δ=+{delta:.4f} > tolerance={tolerance}"
            )

    if failures:
        msg = (
            f"\nECE-регрессия в {prefix} / {attr}:\n"
            + "\n".join(failures)
            + "\n\nЕсли это accepted retrain — обнови "
            "tests/data/ece_baseline.json"
        )
        pytest.fail(msg)


@pytest.mark.parametrize(
    "prefix,attr,baseline_ece",
    [(p, a, e) for p, a, e in _baseline_cells()],
)
def test_ece_recompute_from_bins_matches_stored(prefix: str, attr: str, baseline_ece: dict):
    """Integrity: ECE рекомпьютнутый из bins должен совпадать со stored ECE.

    Ловит порчу JSON-файла либо изменение формулы ECE без перегенерации.
    """
    current = _load_calibration(prefix, attr)
    if current is None:
        pytest.skip(f"{prefix}_{attr}_calibration.json отсутствует")

    drifts: list[str] = []
    for ece_key, bins_key in (("ece_raw", "bins_raw"),
                              ("ece_calibrated", "bins_calibrated")):
        stored = current.get(ece_key)
        bins = current.get(bins_key)
        if stored is None or bins is None:
            continue
        recomputed = _ece_from_bins(bins)
        if recomputed is None:
            continue
        drift = abs(stored - recomputed)
        if drift > 1e-6:
            drifts.append(
                f"  {ece_key}: stored={stored:.6f}, "
                f"recomputed_from_bins={recomputed:.6f}, drift={drift:.2e}"
            )
    if drifts:
        pytest.fail(
            f"\nECE-integrity нарушен в {prefix} / {attr}:\n" + "\n".join(drifts)
        )


@pytest.mark.parametrize(
    "prefix,attr,baseline_ece",
    [(p, a, e) for p, a, e in _baseline_cells()],
)
def test_ece_sanity_cap(prefix: str, attr: str, baseline_ece: dict):
    """Любая stored ECE ≤ SANITY_CAP — ловит совсем сломанные калибровки."""
    current = _load_calibration(prefix, attr)
    if current is None:
        pytest.skip(f"{prefix}_{attr}_calibration.json отсутствует")

    breaches: list[str] = []
    for key in ("ece_raw", "ece_calibrated"):
        val = current.get(key)
        if val is None:
            continue
        if val > SANITY_CAP:
            breaches.append(f"  {key}={val:.4f} > sanity_cap={SANITY_CAP}")
    if breaches:
        pytest.fail(
            f"\nECE превысил sanity-cap в {prefix} / {attr}:\n"
            + "\n".join(breaches)
        )
