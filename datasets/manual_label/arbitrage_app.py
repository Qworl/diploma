"""Веб-интерфейс для ручного арбитража silver vs 3-LLM consensus.

Источник: datasets/manual_label/arbitrage_{category}_{attr}.csv
(создаётся через `python -m src.eval.build_arbitrage_csv --category C --attr A`).

Каждая строка — пара (product, attribute). Показывает рядом:
silver (OFF-tag), gpt-4o-mini, gpt-oss-120b, llama-3.2-3b.
Ты выбираешь финальный ответ (your_arbitrage) и опционально note.

Сохраняется обратно в тот же CSV (с .bak).

Запуск:
    python datasets/manual_label/arbitrage_app.py [--port 8001] [--cat cereals] [--attr cereal_type]

Затем открыть http://localhost:8001/
"""
from __future__ import annotations

import argparse
import csv
import html
import shutil
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote


CSV_PATH: Path | None = None
CATEGORY = ""
ATTR = ""

CATEGORY_HOST = {
    "cheeses": "world.openfoodfacts.org",
    "cereals": "world.openfoodfacts.org",
    "cosmetics": "world.openbeautyfacts.org",
    "pasta": "world.openfoodfacts.org",
    "chocolate": "world.openfoodfacts.org",
    "beverages": "world.openfoodfacts.org",
}

STATUS_ORDER = ["no_majority", "silver_diff", "silver_missing", "no_llm_data", "agree"]
STATUS_LABEL = {
    "no_majority":    "LLM расходятся",
    "silver_diff":    "silver ≠ LLM consensus",
    "silver_missing": "silver пуст",
    "no_llm_data":    "нет LLM голосов",
    "agree":          "все согласны",
}
STATUS_PRIORITY_BADGE = {
    "no_majority":    "high",
    "silver_diff":    "high",
    "silver_missing": "med",
    "no_llm_data":    "low",
    "agree":          "low",
}


def load_rows() -> tuple[list[str], list[dict]]:
    if CSV_PATH is None or not CSV_PATH.exists():
        return [], []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def enum_values(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    enum_str = (rows[0].get("enum_choices") or "").strip()
    if not enum_str:
        return []
    return [v.strip() for v in enum_str.split(",") if v.strip()]


def save_row(idx: int, your_arbitrage: str, note: str) -> bool:
    fieldnames, rows = load_rows()
    if idx < 0 or idx >= len(rows):
        return False
    rows[idx]["your_arbitrage"] = your_arbitrage
    rows[idx]["note"] = note
    backup = CSV_PATH.with_suffix(CSV_PATH.suffix + ".bak")
    shutil.copy2(CSV_PATH, backup)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def render(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 1080px; margin: 24px auto; padding: 0 16px; color: #222; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 18px; margin-top: 28px; }}
  a {{ color: #0a58ca; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  nav.crumbs {{ font-size: 14px; color: #666; margin-bottom: 16px; }}
  table.opinions {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 14px; }}
  table.opinions th, table.opinions td {{ border: 1px solid #ddd; padding: 8px 12px;
                                            text-align: left; vertical-align: top;
                                            font-family: ui-monospace, Menlo, monospace; }}
  table.opinions th {{ background: #f5f5f5; }}
  td.source-silver {{ background: #eef9ee; }}
  td.source-llm    {{ background: #eef4f9; }}
  td.source-auto   {{ background: #fff0e0; font-weight: 600; }}
  td.source-pick   {{ background: #fff6d6; font-weight: 600; }}
  td.empty {{ color: #999; font-style: italic; }}
  .meta {{ background: #fafafa; border: 1px solid #eee; padding: 12px 16px; border-radius: 6px;
            margin: 12px 0; }}
  .meta dt {{ font-weight: 600; color: #555; margin-top: 6px; font-size: 13px;
              font-family: -apple-system, sans-serif; }}
  .meta dd {{ margin: 0 0 4px 0; }}
  .progress {{ background: #f0f0f0; border-radius: 4px; height: 8px; margin: 6px 0; }}
  .progress > div {{ background: #0a58ca; height: 100%; border-radius: 4px; }}
  ul.products {{ list-style: none; padding: 0; }}
  ul.products li {{ padding: 8px 0; border-bottom: 1px solid #f0f0f0; display: flex;
                     gap: 8px; align-items: baseline; }}
  ul.products li.done {{ background: #f0f7f0; }}
  .pill {{ display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 10px;
           background: #eef; color: #336; }}
  .pill.high {{ background: #ffe0d6; color: #8a3a1a; }}
  .pill.med  {{ background: #fff4d6; color: #7a5b00; }}
  .pill.low  {{ background: #eee; color: #555; }}
  .pill.done {{ background: #d1f5d6; color: #1a5a2a; }}
  .filters {{ margin: 12px 0; }}
  .filters a {{ padding: 4px 10px; border-radius: 4px; background: #f0f0f0; margin-right: 4px;
                font-size: 13px; }}
  .filters a.active {{ background: #0a58ca; color: white; }}
  .actions {{ margin-top: 16px; display: flex; gap: 10px; }}
  button.save, .navbtn {{ padding: 8px 18px; font-size: 14px; cursor: pointer; border: none;
                            border-radius: 4px; }}
  button.save {{ background: #0a58ca; color: white; }}
  button.save.secondary {{ background: #5b6770; }}
  button.skip {{ background: #aaa; color: white; }}
  .navbtn {{ background: #f5f5f5; color: #222; padding: 6px 12px; border: 1px solid #ddd; }}
  button.save:hover, button.skip:hover {{ filter: brightness(1.1); }}
  .input-line {{ display: flex; align-items: center; gap: 8px; margin-top: 8px; }}
  .input-line input[type=text] {{ padding: 6px 10px; font: inherit;
                                     font-family: ui-monospace, Menlo, monospace; width: 240px;
                                     border: 2px solid #0a58ca; border-radius: 4px; }}
  .input-line input.note {{ width: 360px; border-color: #aaa; }}
  .chips {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }}
  button.chip {{ font: inherit; font-size: 12px; padding: 4px 10px; cursor: pointer;
                  background: #f0f4ff; border: 1px solid #ccd; border-radius: 12px;
                  color: #335; font-family: ui-monospace, Menlo, monospace; }}
  button.chip:hover {{ background: #dde7ff; }}
  button.chip.silver-val {{ background: #d6ead6; border-color: #9c9; }}
  button.chip.llm-val {{ background: #d6e0ea; border-color: #99c; }}
  .saved {{ background: #d1f5d6; color: #1a5a2a; padding: 8px 12px; border-radius: 4px;
            margin: 12px 0; font-weight: 600; }}
  .pagenav {{ margin-top: 24px; display: flex; gap: 12px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def render_index(filter_status: str = "todo") -> str:
    fieldnames, rows = load_rows()
    if not rows:
        return render("404", "<h1>Нет данных</h1>")

    # Прогресс. Различаем заполнено вручную vs auto-filled.
    n_total = len(rows)
    n_done = sum(1 for r in rows if (r.get("your_arbitrage") or "").strip())
    n_auto = sum(
        1 for r in rows
        if (r.get("auto_arbitrage") or "").strip()
        and (r.get("your_arbitrage") or "").strip() == (r.get("auto_arbitrage") or "").strip()
    )
    n_manual = n_done - n_auto
    n_todo_priority = sum(
        1 for r in rows
        if r.get("status") in ("no_majority", "silver_diff", "silver_missing")
        and not (r.get("your_arbitrage") or "").strip()
    )

    # Фильтр
    def keep(r):
        st = r.get("status", "")
        your = (r.get("your_arbitrage") or "").strip()
        auto = (r.get("auto_arbitrage") or "").strip()
        done = bool(your)
        is_auto = bool(auto) and your == auto
        if filter_status == "todo":
            # Приоритетные status, ещё не заполненные. Если auto уже подставил
            # ответ — кейс уходит из TODO (но появляется в auto_check).
            return not done and st in ("no_majority", "silver_diff", "silver_missing")
        if filter_status == "auto_check":
            # Где auto подставил — нужно spot-check, особенно где silver/LLM
            # расходятся с auto.
            if not is_auto:
                return False
            silver = (r.get("silver") or "").strip().lower()
            return silver != "" and silver != auto.lower()
        if filter_status == "no_majority":
            return st == "no_majority"
        if filter_status == "silver_diff":
            return st == "silver_diff"
        if filter_status == "silver_missing":
            return st == "silver_missing"
        if filter_status == "done":
            return done
        if filter_status == "all":
            return True
        return False

    indices = [i for i, r in enumerate(rows) if keep(r)]

    # Сортируем: high-priority sets первые
    def sort_key(i):
        r = rows[i]
        st = r.get("status", "")
        done = bool((r.get("your_arbitrage") or "").strip())
        return (STATUS_ORDER.index(st) if st in STATUS_ORDER else 99, done, i)
    indices.sort(key=sort_key)

    pct = (n_done / n_total * 100) if n_total else 0
    items = []
    for i in indices[:300]:  # safeguard для UI
        r = rows[i]
        st = r.get("status", "")
        prio = STATUS_PRIORITY_BADGE.get(st, "low")
        done = bool((r.get("your_arbitrage") or "").strip())
        done_pill = '<span class="pill done">✓ done</span>' if done else ""
        name = (r.get("product_name") or "").strip() or "(без названия)"
        brands = (r.get("brands") or "").strip()
        st_label = STATUS_LABEL.get(st, st)
        items.append(
            f'<li class="{"done" if done else ""}">'
            f'<a href="/{i}">{html.escape(name)}</a> '
            f'<span style="color:#777">— {html.escape(brands)}</span> '
            f'<span class="pill {prio}">{html.escape(st_label)}</span> '
            f'{done_pill}'
            f'</li>'
        )

    def fbtn(key, label):
        active = "active" if filter_status == key else ""
        return f'<a class="{active}" href="/?filter={key}">{label}</a>'

    filters = (
        f'<div class="filters">'
        f'Фильтр: {fbtn("todo", "TODO приоритет")} '
        f'{fbtn("auto_check", "verify auto-filled")} '
        f'{fbtn("no_majority", "LLM расходятся")} '
        f'{fbtn("silver_diff", "silver≠LLM")} '
        f'{fbtn("silver_missing", "silver пуст")} '
        f'{fbtn("done", "выполнено")} '
        f'{fbtn("all", "все")}'
        f'</div>'
    )

    body = (
        f"<h1>Арбитраж: {html.escape(CATEGORY)}/{html.escape(ATTR)}</h1>"
        f"<p>Прогресс: <b>{n_done}</b> / {n_total} проверено "
        f"(<b>{n_manual}</b> вручную, <b>{n_auto}</b> auto-filled rule-based; "
        f"приоритетных осталось: <b>{n_todo_priority}</b>)</p>"
        f'<div class="progress"><div style="width:{pct:.1f}%"></div></div>'
        f"{filters}"
        f"<ul class='products'>{''.join(items)}</ul>"
    )
    return render(f"Арбитраж {CATEGORY}/{ATTR}", body)


def render_product(idx: int, *, saved: bool = False) -> str | None:
    fieldnames, rows = load_rows()
    if idx < 0 or idx >= len(rows):
        return None
    row = rows[idx]
    enum = enum_values(rows)

    name = (row.get("product_name") or "").strip() or "(без названия)"
    code = (row.get("code") or "").strip()
    brands = (row.get("brands") or "").strip()
    generic = (row.get("generic_name") or "").strip()
    ingredients = (row.get("ingredients_text") or "").strip()
    silver = (row.get("silver") or "").strip()
    gpt4o = (row.get("gpt4omini") or "").strip()
    gptoss = (row.get("gptoss") or "").strip()
    llama = (row.get("llama3b") or "").strip()
    status = row.get("status", "")
    current = (row.get("your_arbitrage") or "").strip()
    note = (row.get("note") or "").strip()
    auto = (row.get("auto_arbitrage") or "").strip()
    auto_reason = (row.get("auto_reason") or "").strip()
    is_auto_filled = bool(auto) and current == auto

    host = CATEGORY_HOST.get(CATEGORY, "world.openfoodfacts.org")
    off_url = f"https://{host}/product/{html.escape(code)}"

    # Navigation: prev/next в текущем приоритетном subset
    def is_priority(r):
        st = r.get("status", "")
        done = bool((r.get("your_arbitrage") or "").strip())
        return not done and st in ("no_majority", "silver_diff", "silver_missing")

    priority_indices = [i for i, r in enumerate(rows) if is_priority(r) or i == idx]
    try:
        pos = priority_indices.index(idx)
    except ValueError:
        pos = -1
    prev_idx = priority_indices[pos - 1] if pos > 0 else None
    next_idx = priority_indices[pos + 1] if pos >= 0 and pos < len(priority_indices) - 1 else None

    saved_banner = '<div class="saved">✓ сохранено</div>' if saved else ""
    auto_banner = (
        f'<div class="saved" style="background:#fff0e0;color:#7a4a00">'
        f'⚠ auto-filled моим rule-based classifier (reason: <code>{html.escape(auto_reason)}</code>). '
        f'Проверь и подтверди или поменяй.'
        f'</div>'
        if is_auto_filled else ""
    )

    # Голоса как таблица (silver + 3 LLM + claude-rule auto)
    auto_cell = (
        f'<td class="source-auto" title="rule: {html.escape(auto_reason)}">'
        f'{html.escape(auto)}<br><span style="font-size:11px;color:#666">'
        f'{html.escape(auto_reason)}</span></td>'
        if auto else '<td class="empty">—</td>'
    )
    opinions_html = (
        '<table class="opinions">'
        '<thead><tr><th>silver (OFF)</th><th>gpt-4o-mini</th><th>gpt-oss-120b</th>'
        '<th>llama-3.2-3b</th><th>claude-rule</th></tr></thead>'
        f'<tbody><tr>'
        f'<td class="source-silver">{html.escape(silver) if silver else "<em>пусто</em>"}</td>'
        f'<td class="source-llm">{html.escape(gpt4o) if gpt4o else "<em>пусто</em>"}</td>'
        f'<td class="source-llm">{html.escape(gptoss) if gptoss else "<em>пусто</em>"}</td>'
        f'<td class="source-llm">{html.escape(llama) if llama else "<em>пусто</em>"}</td>'
        f'{auto_cell}'
        f'</tr></tbody></table>'
    )

    # Chip-кнопки: значения из голосов + enum
    chip_values: list[tuple[str, str]] = []  # (value, css_class)
    seen = set()
    for v in [silver]:
        if v and v not in seen:
            chip_values.append((v, "silver-val")); seen.add(v)
    for v in [gpt4o, gptoss, llama]:
        if v and v not in seen:
            chip_values.append((v, "llm-val")); seen.add(v)
    for v in enum:
        if v not in seen:
            chip_values.append((v, "")); seen.add(v)

    chips_html = "".join(
        f'<button type="button" class="chip {cls}" data-val="{html.escape(v)}">{html.escape(v)}</button>'
        for v, cls in chip_values
    )

    next_target = f"/{next_idx}" if next_idx is not None else "/"

    body = (
        f'<nav class="crumbs">'
        f'<a href="/">← список</a> &nbsp;·&nbsp; #{idx + 1} из {len(rows)} '
        f'&nbsp;·&nbsp; status: <b>{html.escape(STATUS_LABEL.get(status, status))}</b>'
        f"</nav>"
        f"<h1>{html.escape(name)}</h1>"
        f'<p>'
        f'<a href="{off_url}" target="_blank" rel="noopener">Открыть в OFF ↗</a>'
        f' &nbsp;·&nbsp; <code>{html.escape(code)}</code>'
        f' &nbsp;·&nbsp; brand: <b>{html.escape(brands) if brands else "—"}</b>'
        f"</p>"
        f'<div class="meta">'
        + (f'<dt style="color:#0a58ca">generic_name (часто содержит ответ!)</dt>'
           f'<dd style="font-size:15px;font-weight:600">{html.escape(generic)}</dd>'
           if generic else '')
        + f'<dt>ingredients_text</dt><dd>{html.escape(ingredients) if ingredients else "<em>—</em>"}</dd>'
        + f'</div>'
        f"{saved_banner}"
        f"{auto_banner}"
        f"<h2>Мнения</h2>"
        f"{opinions_html}"
        f"<h2>Твой ответ</h2>"
        f'<form method="post" action="/{idx}">'
        f'<input type="hidden" name="_next" value="{next_target}">'
        f'<div class="input-line">'
        f'  <label>arbitrage: '
        f'  <input type="text" name="your_arbitrage" id="arb" value="{html.escape(current)}" '
        f'         autocomplete="off" autofocus></label>'
        f'  <label>note: '
        f'  <input type="text" name="note" class="note" value="{html.escape(note)}" '
        f'         placeholder="опционально, например &quot;sweetened → granola&quot;"></label>'
        f'</div>'
        f'<div class="chips">{chips_html}</div>'
        f'<div class="actions">'
        f'  <button type="submit" class="save">Сохранить и далее →</button>'
        f'  <button type="submit" name="_stay" value="1" class="save secondary">Сохранить и остаться</button>'
        f'  <button type="submit" name="_skip" value="1" class="skip">Пропустить →</button>'
        f"</div>"
        f"</form>"
        '<script>'
        'document.querySelectorAll("button.chip").forEach(b=>{'
        'b.addEventListener("click",()=>{'
        'document.getElementById("arb").value=b.dataset.val;'
        '});});'
        # Hotkeys: цифры 1-9 для chip
        'document.addEventListener("keydown",e=>{'
        'if(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA") return;'
        'const k=parseInt(e.key);if(!Number.isFinite(k)||k<1||k>9) return;'
        'const chips=document.querySelectorAll("button.chip");'
        'if(chips[k-1]) chips[k-1].click();'
        '});'
        '</script>'
    )
    return render(f"{name} — арбитраж", body)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        raw_path = self.path
        query = ""
        if "?" in raw_path:
            raw_path, query = raw_path.split("?", 1)
        path = unquote(raw_path.rstrip("/")) or "/"
        params = parse_qs(query)
        saved = params.get("saved", ["0"])[0] == "1"
        filter_status = params.get("filter", ["todo"])[0]

        if path == "/":
            self._send(200, render_index(filter_status))
            return

        parts = [p for p in path.split("/") if p]
        if len(parts) == 1:
            try:
                idx = int(parts[0])
            except ValueError:
                self._send(404, "<h1>404</h1>")
                return
            body = render_product(idx, saved=saved)
            if body is None:
                self._send(404, "<h1>404</h1>")
            else:
                self._send(200, body)
            return

        self._send(404, "<h1>404</h1>")

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(self.path.rstrip("/")) or "/"
        parts = [p for p in path.split("/") if p]
        if len(parts) != 1:
            self._send(404, "<h1>404</h1>")
            return
        try:
            idx = int(parts[0])
        except ValueError:
            self._send(404, "<h1>404</h1>")
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        form = parse_qs(raw, keep_blank_values=True)

        is_skip = "_skip" in form
        arb = (form.get("your_arbitrage", [""])[0]).strip() if not is_skip else ""
        note = (form.get("note", [""])[0]).strip()

        if not is_skip:
            ok = save_row(idx, arb, note)
            if not ok:
                self._send(404, "<h1>404</h1>")
                return

        if "_stay" in form:
            redirect = f"/{idx}?saved=1"
        else:
            next_url = form.get("_next", ["/"])[0]
            sep = "&" if "?" in next_url else "?"
            redirect = f"{next_url}{sep}saved=1" if not is_skip else next_url

        self.send_response(303)
        self.send_header("Location", redirect)
        self.end_headers()

    def _send(self, code: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    global CSV_PATH, CATEGORY, ATTR
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cat", default="cereals")
    parser.add_argument("--attr", default="cereal_type")
    args = parser.parse_args()

    CATEGORY = args.cat
    ATTR = args.attr
    here = Path(__file__).resolve().parent
    CSV_PATH = here / f"arbitrage_{CATEGORY}_{ATTR}.csv"
    if not CSV_PATH.exists():
        print(f"Нет {CSV_PATH}. Запусти:")
        print(f"  python -m src.eval.build_arbitrage_csv --category {CATEGORY} --attr {ATTR}")
        return

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl+C для остановки)")
    print(f"CSV: {CSV_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
