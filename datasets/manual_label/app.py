"""Веб-интерфейс для просмотра silver vs manual разметки.

Запуск:
    python webapp/app.py [--port 8000]

Затем открыть http://localhost:8000/
"""
from __future__ import annotations

import argparse
import csv
import html
import shutil
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

# Ensure repo root is importable when this file is run as a script
# (sys.path[0] would otherwise be `datasets/manual_label/` and the
# `src.manual_label.schemas_loader` import below would fail).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.manual_label.schemas_loader import load_domain_attrs, load_pasta_attrs
    _PASTA_ATTRS = load_pasta_attrs()

    def _attrs_for(category: str) -> dict:
        try:
            return load_domain_attrs(category)
        except KeyError:
            return {}
except Exception:  # noqa: BLE001 — running app from non-repo cwd
    _PASTA_ATTRS = {}

    def _attrs_for(category: str) -> dict:
        return _PASTA_ATTRS if category == "pasta" else {}

try:
    from src.manual_label.status import derive_status
except Exception:  # noqa: BLE001 — running app from non-repo cwd
    def derive_status(silver: str, manual: str, prev: str) -> str:
        """Fallback used only when src/ isn't importable; behaviour is identical."""
        if prev == "unsure":
            return "unsure"
        s = (silver or "").strip()
        m = (manual or "").strip()
        if m == "":
            return "empty"
        if s == "":
            return "manual_only"
        if m == s:
            return "confirmed"
        return "override"


def schema_values(category: str, attr: str) -> list[str] | None:
    """Return canonical values for `(category, attr)` or None to fall back to CSV inference."""
    attrs = _attrs_for(category)
    if attr in attrs:
        spec = attrs[attr]
        vals = list(spec.get("values", []))
        if spec.get("nullable"):
            vals.append("")  # explicit "no value"
        return vals
    return None


def _find_label_dir() -> Path:
    """Ищем папку с *_labeled.csv: рядом со скриптом или в datasets/manual_label/."""
    here = Path(__file__).resolve().parent
    candidates = [
        here,
        here / "datasets" / "manual_label",
        here.parent / "datasets" / "manual_label",
        here.parent.parent / "datasets" / "manual_label",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*_labeled.csv")):
            return c
    return here


ROOT = _find_label_dir()

CATEGORIES = ["cosmetics", "cheeses", "cereals",
              "pasta", "chocolate", "beverages"]

# Каждая категория размечается на собственной платформе семейства Open Food Facts.
CATEGORY_HOST = {
    "cheeses": "world.openfoodfacts.org",
    "cereals": "world.openfoodfacts.org",
    "cosmetics": "world.openbeautyfacts.org",
    "pasta": "world.openfoodfacts.org",
    "chocolate": "world.openfoodfacts.org",
    "beverages": "world.openfoodfacts.org",
}

CATEGORY_FILE = {
    "pasta": "pasta_gold_250.csv",  # Trek D gold annotation in progress
    "cosmetics": "cosmetics_labeled.csv",
    "cheeses": "cheeses_labeled.csv",
    "cereals": "cereals_labeled.csv",
    "chocolate": "chocolate_labeled.csv",
    "beverages": "beverages_labeled.csv",
}

# Trek E gold CSVs override the legacy *_labeled.csv when they exist on disk.
# This keeps the label-by-hand workflow alive while pointing the UI at the
# audit-ready files.
_TREK_E_GOLD = {
    "chocolate": "chocolate_gold_239.csv",
    "cheeses": "cheeses_gold_239.csv",
}
for _cat, _fname in _TREK_E_GOLD.items():
    if (ROOT / _fname).exists():
        CATEGORY_FILE[_cat] = _fname


def load_category(category: str) -> tuple[list[str], list[dict]]:
    filename = CATEGORY_FILE.get(category, f"{category}_labeled.csv")
    path = ROOT / filename
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def attribute_pairs(fieldnames: list[str]) -> list[str]:
    """Список атрибутов, для которых есть и silver_, и manual_ колонка."""
    silvers = {c[len("silver_"):] for c in fieldnames if c.startswith("silver_")}
    manuals = {c[len("manual_"):] for c in fieldnames if c.startswith("manual_")}
    common = silvers & manuals
    # Сохраняем порядок из CSV
    ordered: list[str] = []
    for c in fieldnames:
        if c.startswith("silver_"):
            attr = c[len("silver_"):]
            if attr in common and attr not in ordered:
                ordered.append(attr)
    return ordered


def category_progress(category: str) -> dict:
    """Compute aggregate progress over a category's labeled CSV."""
    fieldnames, rows = load_category(category)
    if not rows:
        return {"total": 0, "done": 0, "median_sec": None, "eta_hours": None}
    attrs = attribute_pairs(fieldnames)
    # Include legacy "confident" for back-compat with un-migrated CSVs.
    DONE_STATUSES = {"confirmed", "override", "manual_only", "unsure", "confident"}
    total = len(rows)
    done = 0
    timestamps = []
    for r in rows:
        statuses = [
            (r.get(f"manual_{a}_status") or "").strip()
            for a in attrs
        ]
        if statuses and all(s in DONE_STATUSES for s in statuses):
            done += 1
        for a in attrs:
            ts = (r.get(f"manual_{a}_at") or "").strip()
            if ts:
                timestamps.append(ts)
    median_sec = None
    eta_hours = None
    if len(timestamps) >= 4:
        import datetime as _dt
        ts_sorted = sorted(timestamps)
        deltas = []
        prev = None
        for t in ts_sorted:
            try:
                cur = _dt.datetime.fromisoformat(t)
            except ValueError:
                continue
            if prev is not None:
                d = (cur - prev).total_seconds()
                if 1 <= d <= 600:  # ignore breaks and instant double-saves
                    deltas.append(d)
            prev = cur
        if deltas:
            deltas.sort()
            median_sec = deltas[len(deltas) // 2]
            remaining = total - done
            n_attrs = len(attrs)
            eta_hours = round(median_sec * remaining * n_attrs / 3600, 1)
    AUDITED_STATUSES = {"confirmed", "override", "manual_only", "confident"}
    override_rate_by_mode: dict[str, dict[str, dict]] = {}
    for a in attrs:
        by_mode: dict[str, dict] = {}
        for mode in ("blind", "prefill", "llm"):
            n_audited = 0
            n_override = 0
            for r in rows:
                if (r.get(f"manual_{a}_mode") or "").strip() != mode:
                    continue
                st = (r.get(f"manual_{a}_status") or "").strip()
                if st not in AUDITED_STATUSES:
                    continue
                n_audited += 1
                if st == "override":
                    n_override += 1
            by_mode[mode] = {
                "n_audited": n_audited,
                "n_override": n_override,
                "override_rate": (n_override / n_audited) if n_audited else None,
            }
        override_rate_by_mode[a] = by_mode

    return {
        "total": total,
        "done": done,
        "median_sec": median_sec,
        "eta_hours": eta_hours,
        "override_rate_by_mode": override_rate_by_mode,
    }


def render(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 980px; margin: 24px auto; padding: 0 16px; color: #222; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 18px; margin-top: 28px; }}
  a {{ color: #0a58ca; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  nav.crumbs {{ font-size: 14px; color: #666; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 14px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; }}
  tr.diff td.silver, tr.diff td.manual {{ background: #fff4d6; }}
  tr.match td.silver, tr.match td.manual {{ background: #e9f7ec; }}
  td.attr {{ font-weight: 600; width: 28%; font-family: ui-monospace, Menlo, monospace; }}
  td.silver, td.manual {{ width: 36%; font-family: ui-monospace, Menlo, monospace; white-space: pre-wrap; }}
  .meta {{ background: #fafafa; border: 1px solid #eee; padding: 12px 16px; border-radius: 6px; }}
  .meta dt {{ font-weight: 600; color: #555; margin-top: 6px; font-size: 13px; }}
  .meta dd {{ margin: 0 0 4px 0; }}
  ul.products {{ list-style: none; padding: 0; }}
  ul.products li {{ padding: 6px 0; border-bottom: 1px solid #f0f0f0; }}
  .pill {{ display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 10px;
           background: #eef; color: #336; margin-left: 6px; }}
  .badge-diff {{ background: #fff4d6; color: #7a5b00; }}
  .badge-match {{ background: #e9f7ec; color: #1a7a3a; }}
  .empty {{ color: #999; font-style: italic; }}
  td.manual input[type=text] {{ width: calc(100% - 32px); padding: 4px 6px; font: inherit;
                                 font-family: ui-monospace, Menlo, monospace; box-sizing: border-box; }}
  button.copy {{ width: 26px; margin-left: 4px; padding: 4px 0; cursor: pointer;
                  background: #eef; border: 1px solid #ccd; border-radius: 4px; font-weight: bold; }}
  .actions {{ margin-top: 16px; display: flex; gap: 10px; }}
  button.save {{ padding: 8px 18px; font-size: 14px; cursor: pointer;
                  background: #0a58ca; color: white; border: none; border-radius: 4px; }}
  button.save.secondary {{ background: #5b6770; }}
  button.save:hover {{ filter: brightness(1.1); }}
  .pagenav {{ margin-top: 24px; display: flex; gap: 12px; }}
  .navbtn {{ padding: 6px 12px; background: #f5f5f5; border-radius: 4px; }}
  .saved {{ background: #d1f5d6; color: #1a5a2a; padding: 8px 12px; border-radius: 4px;
            margin: 12px 0; font-weight: 600; }}
  .chips {{ margin-top: 4px; display: flex; flex-wrap: wrap; gap: 4px; }}
  button.chip {{ font: inherit; font-size: 11px; padding: 2px 8px; cursor: pointer;
                  background: #f0f4ff; border: 1px solid #ccd; border-radius: 10px;
                  color: #335; font-family: ui-monospace, Menlo, monospace; }}
  button.chip:hover {{ background: #dde7ff; }}
  tr[data-status="unsure"] td.manual {{ background: #fff0e0; }}
  tr[data-status="empty"] td.manual {{ background: #fafafa; }}
  tr[data-status="auto"] td.manual {{ background: #fffbe6; }}
  tr[data-status="confirmed"] td.manual {{ background: #e9f7ec; }}
  tr[data-status="override"] td.manual {{ background: #ffe9d6; }}
  tr[data-status="manual_only"] td.manual {{ background: #e6f0ff; }}
  .mode-badge {{ font-family: ui-monospace, Menlo, monospace; font-size: 10px;
                padding: 1px 4px; margin-left: 4px; border: 1px solid #bbb;
                border-radius: 3px; color: #555; }}
  button.confirm {{ margin-left: 4px; padding: 4px 6px; cursor: pointer;
                    background: #e9f7ec; border: 1px solid #95c8a7;
                    border-radius: 4px; }}
  button.confirm[hidden] {{ display: none; }}
  .status-pill {{ font-size: 11px; padding: 1px 6px; margin-left: 6px;
                  background: #eee; color: #555; border-radius: 8px; }}
  button.unsure {{ margin-left: 4px; padding: 4px 6px; cursor: pointer;
                   background: #ffe9d4; border: 1px solid #d9b08c; border-radius: 4px; }}
  .progress {{ background: #eef6ff; border: 1px solid #b8d4ff; padding: 8px 14px;
              border-radius: 4px; margin-bottom: 14px; font-size: 14px; }}
  .filters {{ margin: 12px 0; font-size: 13px; }}
  .filters a {{ display: inline-block; padding: 3px 10px; margin-right: 6px;
               border: 1px solid #ccd; border-radius: 12px; color: #335; }}
  .filters a.f-on {{ background: #335; color: white; border-color: #335; }}
  .filters a.f-off {{ background: #f5f7fa; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def render_index() -> str:
    items = []
    for cat in CATEGORIES:
        fieldnames, rows = load_category(cat)
        items.append(
            f'<li><a href="/{cat}">{html.escape(cat)}</a> '
            f'<span class="pill">{len(rows)} товаров</span></li>'
        )
    body = (
        "<h1>Silver vs Manual labeling</h1>"
        "<p>Просмотр размеченных продуктов из <code>datasets/manual_label/</code>. "
        "Слева — silver (автотеги OFF), справа — manual (Sonnet + ручная верификация).</p>"
        "<ul class='products'>" + "".join(items) + "</ul>"
    )
    return render("Silver vs Manual", body)


def render_category(category: str, only: str = "") -> str | None:
    fieldnames, rows = load_category(category)
    if not fieldnames:
        return None
    attrs = attribute_pairs(fieldnames)

    def passes(row: dict) -> bool:
        if only == "":
            return True
        if only == "empty":
            return any(
                (row.get(f"manual_{a}_status") or "empty") == "empty"
                for a in attrs
            )
        if only == "disagreed":
            return any(
                (row.get(f"silver_{a}", "") or "").strip()
                != (row.get(f"manual_{a}", "") or "").strip()
                for a in attrs
            )
        if only.endswith("_empty"):
            target = only[: -len("_empty")]
            return (row.get(f"manual_{target}_status") or "empty") == "empty"
        if only == "auto":
            return any(
                (row.get(f"manual_{a}_status") or "").strip() == "auto"
                for a in attrs
            )
        if only == "override":
            return any(
                (row.get(f"manual_{a}_status") or "").strip() == "override"
                for a in attrs
            )
        if only == "manual_only":
            return any(
                (row.get(f"manual_{a}_status") or "").strip() == "manual_only"
                for a in attrs
            )
        return True

    rows_filt = [r for r in rows if passes(r)]
    prog = category_progress(category)
    prog_bar = (
        f'<div class="progress">'
        f'<b>{prog["done"]}</b> / {prog["total"]} размечено'
    )
    if prog["median_sec"] is not None:
        prog_bar += (
            f' &nbsp;·&nbsp; темп ~{int(prog["median_sec"])} сек/атрибут'
            f' &nbsp;·&nbsp; ETA ~{prog["eta_hours"]} ч'
        )
    prog_bar += "</div>"

    filter_chips = ['<div class="filters">фильтр: ']
    for label, q in [
        ("все", ""), ("спорные", "disagreed"), ("пустые", "empty"),
        ("ждут подтверждения", "auto"),
        ("переопределено", "override"),
        ("manual-only", "manual_only"),
    ]:
        cls = "f-on" if only == q else "f-off"
        href = f"/{category}" + (f"?only={q}" if q else "")
        filter_chips.append(f'<a class="{cls}" href="{href}">{label}</a> ')
    for a in attrs:
        cls = "f-on" if only == f"{a}_empty" else "f-off"
        filter_chips.append(
            f'<a class="{cls}" href="/{category}?only={a}_empty">{html.escape(a)} пуст</a> '
        )
    filter_chips.append("</div>")

    items = []
    for row in rows_filt:
        code = row.get("code", "").strip()
        name = row.get("product_name", "").strip() or "(без названия)"
        brands = row.get("brands", "").strip()
        diffs = sum(
            1 for a in attrs
            if (row.get(f"silver_{a}", "") or "").strip()
               != (row.get(f"manual_{a}", "") or "").strip()
        )
        empties = sum(
            1 for a in attrs
            if (row.get(f"manual_{a}_status") or "empty") == "empty"
        )
        badge_cls = "badge-diff" if diffs else "badge-match"
        badge_txt = f"{diffs} расхождений" if diffs else "совпадает"
        empty_pill = (
            f' <span class="pill" style="background:#eee">{empties} пустых</span>'
            if empties else ""
        )
        items.append(
            f'<li><a href="/{category}/{html.escape(code)}">'
            f'{html.escape(name)}</a> '
            f'<span style="color:#777">— {html.escape(brands)}</span> '
            f'<span class="pill {badge_cls}">{badge_txt}</span>'
            f"{empty_pill}</li>"
        )

    body = (
        f'<nav class="crumbs"><a href="/">← все категории</a></nav>'
        f"<h1>{html.escape(category)} "
        f'<span class="pill">{len(rows_filt)} из {len(rows)}</span></h1>'
        f"{prog_bar}"
        f"{''.join(filter_chips)}"
        f"<ul class='products'>{''.join(items)}</ul>"
    )
    return render(f"{category} — список", body)


def allowed_values(rows: list[dict], attr: str) -> list[str]:
    """Уникальные непустые значения, встречающиеся в silver_<attr>/manual_<attr>."""
    vals: set[str] = set()
    for r in rows:
        for col in (f"silver_{attr}", f"manual_{attr}"):
            v = (r.get(col) or "").strip()
            if v:
                vals.add(v)
    return sorted(vals)


def save_row(
    category: str,
    code: str,
    manual_values: dict[str, str],
    manual_statuses: dict[str, str] | None = None,
    manual_modes: dict[str, str] | None = None,
) -> bool:
    """Overwrite the row with given `code` in <category> CSV.

    `manual_values` maps attr → string value.
    `manual_statuses` maps attr → status string. Recognised values:
    - `empty` — not reviewed
    - `auto` — pre-filled from silver, awaiting review (Task 3+)
    - `confirmed` / `override` / `manual_only` — derived by `derive_status`
    - `unsure` — explicit doubt (sticky)
    - `confident` — legacy, kept for backwards compat with pre-pivot CSVs
    If the dict omits an attr (or sets it to None/""), the status is derived
    from (silver, manual, prev) via `derive_status`.
    `manual_modes` maps attr → "blind" | "prefill" | "". Only set when
    non-empty (server never clears mode). Persists once set.
    Backs file up to `.bak` before writing.
    """
    filename = CATEGORY_FILE.get(category, f"{category}_labeled.csv")
    path = ROOT / filename
    if not path.exists():
        return False
    fieldnames, rows = load_category(category)
    statuses = manual_statuses or {}
    found = False
    for row in rows:
        if row.get("code", "").strip() == code:
            found = True
            import datetime as _dt
            now_iso = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
            for attr, val in manual_values.items():
                col = f"manual_{attr}"
                st_col = f"manual_{attr}_status"
                at_col = f"manual_{attr}_at"
                prev_val = row.get(col, "") if col in fieldnames else ""
                prev_status = row.get(st_col, "") if st_col in fieldnames else ""
                new_status = statuses.get(attr)
                if new_status is None or new_status == "":
                    silver_val = row.get(f"silver_{attr}", "") if f"silver_{attr}" in fieldnames else ""
                    new_status = derive_status(silver_val, val, prev_status)
                changed = (val != prev_val) or (new_status != prev_status)
                if col in fieldnames:
                    row[col] = val
                if st_col in fieldnames:
                    row[st_col] = new_status
                if at_col in fieldnames and changed:
                    row[at_col] = now_iso
                mode_col = f"manual_{attr}_mode"
                new_mode = (manual_modes or {}).get(attr, "")
                if mode_col in fieldnames and new_mode:
                    row[mode_col] = new_mode
            break
    if not found:
        return False
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def render_product(category: str, code: str, *, saved: bool = False) -> str | None:
    fieldnames, rows = load_category(category)
    if not fieldnames:
        return None
    idx = next((i for i, r in enumerate(rows) if r.get("code", "").strip() == code), -1)
    if idx < 0:
        return None
    row = rows[idx]

    attrs = attribute_pairs(fieldnames)
    name = row.get("product_name", "").strip() or "(без названия)"
    host = CATEGORY_HOST.get(category, "world.openfoodfacts.org")
    off_url = f"https://{host}/product/{html.escape(code)}"
    off_label = {
        "world.openfoodfacts.org": "Open Food Facts",
        "world.openbeautyfacts.org": "Open Beauty Facts",
        "world.openpetfoodfacts.org": "Open Pet Food Facts",
    }.get(host, "OFF")

    prev_code = rows[idx - 1].get("code", "").strip() if idx > 0 else ""
    next_code = rows[idx + 1].get("code", "").strip() if idx < len(rows) - 1 else ""

    meta = "<dl>"
    for col in ["code", "product_name", "brands", "quantity", "ingredients_text"]:
        if col in row and row[col]:
            meta += f"<dt>{html.escape(col)}</dt><dd>{html.escape(row[col])}</dd>"
    meta += "</dl>"

    rows_html = []
    datalists_html = []
    for a in attrs:
        s = (row.get(f"silver_{a}", "") or "").strip()
        m = (row.get(f"manual_{a}", "") or "").strip()
        cls = "diff" if s != m else "match"
        s_disp = html.escape(s) if s else '<span class="empty">—</span>'
        input_name = f"manual_{a}"
        list_id = f"opts_{a}"
        canonical = schema_values(category, a)
        opts = canonical if canonical is not None else allowed_values(rows, a)
        datalists_html.append(
            f'<datalist id="{html.escape(list_id)}">'
            + "".join(f'<option value="{html.escape(o)}">' for o in opts)
            + "</datalist>"
        )
        chips = " ".join(
            f'<button type="button" class="chip" '
            f'data-target="{html.escape(input_name)}" '
            f'data-val="{html.escape(o)}">{html.escape(o)}</button>'
            for o in opts
        )
        chips_html = f'<div class="chips">{chips}</div>' if chips else ""
        status_val = (row.get(f"manual_{a}_status", "") or "empty").strip() or "empty"
        mode_val = (row.get(f"manual_{a}_mode", "") or "").strip()
        confirm_hidden = "" if status_val == "auto" else "hidden"
        badge_letter = "B" if mode_val == "blind" else "P" if mode_val == "prefill" else ""
        rows_html.append(
            f'<tr class="{cls}" data-attr="{html.escape(a)}" '
            f'data-status="{html.escape(status_val)}" '
            f'data-mode="{html.escape(mode_val)}">'
            f'<td class="attr">{html.escape(a)}</td>'
            f'<td class="silver">{s_disp}</td>'
            f'<td class="manual">'
            f'<input type="text" name="{html.escape(input_name)}" '
            f'id="{html.escape(input_name)}" '
            f'list="{html.escape(list_id)}" '
            f'value="{html.escape(m)}" data-silver="{html.escape(s)}">'
            f'<input type="hidden" name="status_{html.escape(a)}" '
            f'id="status_{html.escape(a)}" value="{html.escape(status_val)}">'
            f'<input type="hidden" name="mode_{html.escape(a)}" '
            f'id="mode_{html.escape(a)}" value="{html.escape(mode_val)}">'
            f'<button type="button" class="copy" title="Скопировать silver">=</button>'
            f'<button type="button" class="confirm" {confirm_hidden} '
            f'title="Подтвердить (c)">✓</button>'
            f'<button type="button" class="unsure" title="Пометить как unsure (u)">?</button>'
            f'<span class="status-pill">{html.escape(status_val)}</span>'
            f'<span class="mode-badge">{badge_letter}</span>'
            f"{chips_html}"
            f"</td></tr>"
        )

    nav_btns = []
    if prev_code:
        nav_btns.append(f'<a class="navbtn" href="/{category}/{html.escape(prev_code)}">← prev</a>')
    nav_btns.append(f'<a class="navbtn" href="/{category}">список</a>')
    if next_code:
        nav_btns.append(f'<a class="navbtn" href="/{category}/{html.escape(next_code)}">next →</a>')
    nav_html = " ".join(nav_btns)

    next_after_save = next_code or code
    saved_banner = (
        '<div class="saved">✓ сохранено</div>' if saved else ""
    )

    body = (
        f'<nav class="crumbs">'
        f'<a href="/">все категории</a> / '
        f'<a href="/{category}">{html.escape(category)}</a> / '
        f'{html.escape(code)}'
        f' &nbsp;·&nbsp; #{idx + 1} из {len(rows)}'
        f"</nav>"
        f"<h1>{html.escape(name)}</h1>"
        f'<p><a href="{off_url}" target="_blank" rel="noopener">'
        f'Открыть в {html.escape(off_label)} ↗</a></p>'
        f'<div class="meta">{meta}</div>'
        f"{saved_banner}"
        f"<h2>Атрибуты</h2>"
        f'<form method="post" action="/{category}/{html.escape(code)}">'
        f'<input type="hidden" name="_next" value="/{category}/{html.escape(next_after_save)}">'
        f"{''.join(datalists_html)}"
        f"<table><thead><tr>"
        f"<th>атрибут</th><th>silver</th><th>manual</th>"
        f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
        f'<div class="actions">'
        f'<button type="submit" class="save">Сохранить</button>'
        f'<button type="submit" name="_stay" value="1" class="save secondary">Сохранить и остаться</button>'
        f"</div>"
        f"</form>"
        f'<div class="pagenav">{nav_html}</div>'
        '<script>'
        'function deriveStatus(silver, manual, prev) {'
        '  if (prev === "unsure") return "unsure";'
        '  const s = (silver || "").trim();'
        '  const m = (manual || "").trim();'
        '  if (m === "") return "empty";'
        '  if (s === "") return "manual_only";'
        '  if (m === s) return "confirmed";'
        '  return "override";'
        '}'
        'function modeBadgeLetter(mode) {'
        '  return mode === "blind" ? "B" : mode === "prefill" ? "P" : "";'
        '}'
        'function updateRow(tr) {'
        '  const attr = tr.dataset.attr;'
        '  const inp = tr.querySelector(\'input[type=text]\');'
        '  const st = document.getElementById("status_" + attr);'
        '  const md = document.getElementById("mode_" + attr);'
        '  const silver = inp.dataset.silver || "";'
        '  st.value = deriveStatus(silver, inp.value, st.value);'
        '  if (!md.value) md.value = "blind";'
        '  tr.dataset.status = st.value;'
        '  tr.dataset.mode = md.value;'
        '  tr.querySelector(".status-pill").textContent = st.value;'
        '  tr.querySelector(".mode-badge").textContent = modeBadgeLetter(md.value);'
        '  const cb = tr.querySelector("button.confirm");'
        '  if (cb) cb.hidden = st.value !== "auto";'
        '}'
        'document.querySelectorAll("button.copy").forEach(b => {'
        '  b.addEventListener("click", () => {'
        '    const inp = b.parentElement.querySelector("input");'
        '    inp.value = inp.dataset.silver;'
        '    inp.dispatchEvent(new Event("input", {bubbles: true}));'
        '  });'
        '});'
        'document.querySelectorAll("button.chip").forEach(b => {'
        '  b.addEventListener("click", () => {'
        '    const t = document.getElementById(b.dataset.target);'
        '    if (t) {'
        '      t.value = b.dataset.val;'
        '      t.dispatchEvent(new Event("input", {bubbles: true}));'
        '    }'
        '  });'
        '});'
        'document.querySelectorAll("button.unsure").forEach(b => {'
        '  b.addEventListener("click", () => {'
        '    const tr = b.closest("tr");'
        '    const attr = tr.dataset.attr;'
        '    const st = document.getElementById("status_" + attr);'
        '    const md = document.getElementById("mode_" + attr);'
        '    st.value = (st.value === "unsure") ? "empty" : "unsure";'
        '    if (st.value === "unsure" && !md.value) md.value = "blind";'
        '    tr.dataset.status = st.value;'
        '    tr.dataset.mode = md.value;'
        '    tr.querySelector(".status-pill").textContent = st.value;'
        '    tr.querySelector(".mode-badge").textContent = modeBadgeLetter(md.value);'
        '    const cb = tr.querySelector("button.confirm");'
        '    if (cb) cb.hidden = st.value !== "auto";'
        '  });'
        '});'
        'document.querySelectorAll("button.confirm").forEach(b => {'
        '  b.addEventListener("click", () => {'
        '    const tr = b.closest("tr");'
        '    const inp = tr.querySelector(\'input[type=text]\');'
        '    inp.dispatchEvent(new Event("input", {bubbles: true}));'
        '  });'
        '});'
        'document.querySelectorAll(\'td.manual input[type=text]\').forEach(inp => {'
        '  inp.addEventListener("input", () => updateRow(inp.closest("tr")));'
        '});'
        'document.addEventListener("keydown", e => {'
        '  if (e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return;'
        '  const inp = document.activeElement;'
        '  if (!inp || inp.tagName !== "INPUT" || inp.type !== "text") return;'
        '  const tr = inp.closest("tr");'
        '  if (!tr) return;'
        '  if (/^[1-9]$/.test(e.key)) {'
        '    const chips = tr.querySelectorAll("button.chip");'
        '    const idx = parseInt(e.key, 10) - 1;'
        '    if (chips[idx]) { e.preventDefault(); chips[idx].click(); }'
        '    return;'
        '  }'
        '  if (e.key === "s") { e.preventDefault(); tr.querySelector("button.copy").click(); return; }'
        '  if (e.key === "u") { e.preventDefault(); tr.querySelector("button.unsure").click(); return; }'
        '  if (e.key === "c") {'
        '    e.preventDefault();'
        '    const cb = tr.querySelector("button.confirm");'
        '    if (cb && !cb.hidden) cb.click();'
        '    else inp.dispatchEvent(new Event("input", {bubbles: true}));'
        '  }'
        '});'
        'const form = document.querySelector("form");'
        'async function ajaxSave() {'
        '  const manual = {}, status = {}, mode = {};'
        '  document.querySelectorAll(\'td.manual input[type=text]\').forEach(inp => {'
        '    manual[inp.closest("tr").dataset.attr] = inp.value;'
        '  });'
        '  document.querySelectorAll(\'input[name^="status_"]\').forEach(h => {'
        '    status[h.name.slice("status_".length)] = h.value;'
        '  });'
        '  document.querySelectorAll(\'input[name^="mode_"]\').forEach(h => {'
        '    mode[h.name.slice("mode_".length)] = h.value;'
        '  });'
        '  const url = form.action.replace(/\\/(\\w+)\\/([^/?]+)$/, "/api/save/$1/$2");'
        '  const resp = await fetch(url, {'
        '    method: "POST",'
        '    headers: {"Content-Type": "application/json"},'
        '    body: JSON.stringify({manual, status, mode}),'
        '  });'
        '  if (!resp.ok) { alert("save failed: " + resp.status); return false; }'
        '  const banner = document.createElement("div");'
        '  banner.className = "saved";'
        '  banner.textContent = "✓ сохранено";'
        '  form.parentElement.insertBefore(banner, form);'
        '  setTimeout(() => banner.remove(), 1200);'
        '  return true;'
        '}'
        'form.addEventListener("submit", async e => {'
        '  e.preventDefault();'
        '  const ok = await ajaxSave();'
        '  if (!ok) return;'
        '  const next = form.querySelector(\'input[name="_next"]\').value;'
        '  if (e.submitter && e.submitter.name === "_stay") return;'
        '  window.location.href = next;'
        '});'
        '</script>'
    )
    return render(f"{name} — {category}", body)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        raw_path = self.path
        query = ""
        if "?" in raw_path:
            raw_path, query = raw_path.split("?", 1)
        path = unquote(raw_path.rstrip("/")) or "/"
        parts = [p for p in path.split("/") if p]
        params = parse_qs(query)
        saved = params.get("saved", ["0"])[0] == "1"

        if not parts:
            self._send(200, render_index())
            return

        if len(parts) == 1 and parts[0] in CATEGORIES:
            only = (params.get("only", [""])[0] or "").strip()
            html_body = render_category(parts[0], only=only)
            if html_body is None:
                self._send(404, "<h1>404</h1>")
            else:
                self._send(200, html_body)
            return

        if len(parts) == 2 and parts[0] in CATEGORIES:
            html_body = render_product(parts[0], parts[1], saved=saved)
            if html_body is None:
                self._send(404, "<h1>404 — товар не найден</h1>")
            else:
                self._send(200, html_body)
            return

        self._send(404, "<h1>404</h1>")

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(self.path.rstrip("/")) or "/"
        parts = [p for p in path.split("/") if p]
        if parts[:1] == ["api"] and parts[1:2] == ["save"] and len(parts) == 4 and parts[2] in CATEGORIES:
            self._handle_json_save(parts[2], parts[3])
            return
        if len(parts) != 2 or parts[0] not in CATEGORIES:
            self._send(404, "<h1>404</h1>")
            return

        category, code = parts[0], parts[1]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        form = parse_qs(raw, keep_blank_values=True)

        manual_values = {
            k[len("manual_"):]: (v[0] if v else "").strip()
            for k, v in form.items()
            if k.startswith("manual_")
        }
        manual_statuses = {
            k[len("status_"):]: (v[0] if v else "").strip()
            for k, v in form.items()
            if k.startswith("status_")
        }
        manual_modes = {
            k[len("mode_"):]: (v[0] if v else "").strip()
            for k, v in form.items()
            if k.startswith("mode_")
        }
        ok = save_row(category, code, manual_values, manual_statuses, manual_modes)
        if not ok:
            self._send(404, "<h1>404 — товар не найден</h1>")
            return

        stay = "_stay" in form
        if stay:
            redirect = f"/{category}/{code}?saved=1"
        else:
            next_url = (form.get("_next") or [f"/{category}/{code}?saved=1"])[0]
            sep = "&" if "?" in next_url else "?"
            redirect = f"{next_url}{sep}saved=1"
        self.send_response(303)
        self.send_header("Location", redirect)
        self.end_headers()

    def _handle_json_save(self, category: str, code: str) -> None:
        import json
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "invalid json"})
            return
        values = payload.get("manual") or {}
        statuses = payload.get("status") or {}
        modes = payload.get("mode") or {}
        ok = save_row(
            category, code,
            {k: str(v) for k, v in values.items()},
            {k: str(v) for k, v in statuses.items()},
            {k: str(v) for k, v in modes.items()},
        )
        if not ok:
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        self._send_json(200, {"ok": True})

    def _send_json(self, code: int, body: dict) -> None:
        import json
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send(self, code: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:  # тише в stdout
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl+C для остановки)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
