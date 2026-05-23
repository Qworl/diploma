#!/usr/bin/env python3
"""Smoke tests for ``check_regression.py``.

Run from the repository root via::

    python -m unittest discover -s docs/thesis/docx_build \
        -p 'check_regression_test.py' -v

Or from this directory::

    python -m unittest check_regression_test.py

Tests rely only on the standard library plus ``python-docx`` / ``nbformat``
(which are required by the module under test). No pytest, no fixtures.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

# Make the sibling module importable regardless of where tests are launched.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import check_regression as cr  # noqa: E402  (must follow sys.path tweak)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_docx(path: Path, paragraphs: list[str]) -> None:
    """Create a minimal DOCX containing the given paragraphs."""
    import docx

    doc = docx.Document()
    for para in paragraphs:
        doc.add_paragraph(para)
    doc.save(str(path))


def _make_notebook(path: Path, markdown_cells: list[str],
                   code_cells: list[str] | None = None) -> None:
    """Create a minimal ipynb with the given markdown / code cells."""
    import nbformat

    nb = nbformat.v4.new_notebook()
    cells = []
    for src in markdown_cells:
        cells.append(nbformat.v4.new_markdown_cell(src))
    for src in (code_cells or []):
        cells.append(nbformat.v4.new_code_cell(src))
    nb.cells = cells
    with path.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)


def _empty_golden_numbers() -> str:
    return (
        "# GOLDEN_NUMBERS\n\n"
        "## Правила\n\n"
        "Prose.\n\n"
        "## Колонки\n\n"
        "| Число | Файл | Раздел | Происхождение |\n"
        "|---|---|---|---|\n\n"
    )


def _empty_golden_strings() -> str:
    return (
        "# GOLDEN_STRINGS\n\n"
        "## Правила\n\n"
        "Prose only, no numbered data sections.\n\n"
        "## Чек-листы\n\n"
        "- [ ] `not-a-real-anchor.txt`  (this section is ignored)\n"
    )


# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------

class TestNormalize(unittest.TestCase):
    def test_normalize_handles_nbsp(self):
        # NBSP, tab, newline and multiple spaces collapse to a single space.
        raw = "93,81\xa0%\tof\ncascade   solutions"
        self.assertEqual(cr.normalize(raw), "93,81 % of cascade solutions")

    def test_normalize_strips_edges(self):
        self.assertEqual(cr.normalize("  \xa0 hello  \t"), "hello")

    def test_normalize_empty(self):
        self.assertEqual(cr.normalize(""), "")
        self.assertEqual(cr.normalize("   \xa0\t\n"), "")


class TestParseGoldenNumbers(unittest.TestCase):
    def test_parse_golden_numbers_skips_example_section(self):
        body = (
            "# title\n\n"
            "## Правила\n\n"
            "Prose.\n\n"
            "## Колонки\n\n"
            "| Число | Файл | Раздел | Происхождение |\n"
            "|---|---|---|---|\n"
            "| `93,81 %` | `04.md` | §4.3 | `out.parquet` |\n"
            "| `n=1539` | `03.md` | §3.3.5 | `gold.parquet` |\n\n"
            "## Пример формата (удалить при наполнении)\n\n"
            "| Число | Файл | Раздел | Происхождение |\n"
            "|---|---|---|---|\n"
            "| `999,99 %` | `fake.md` | example | n/a |\n"
            "| `n=42` | `fake.md` | example | n/a |\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "GOLDEN_NUMBERS.md"
            _write(path, body)
            result = cr.parse_golden_numbers(path)
        self.assertIn("93,81 %", result)
        self.assertIn("n=1539", result)
        self.assertNotIn("999,99 %", result)
        self.assertNotIn("n=42", result)
        self.assertEqual(len(result), 2)

    def test_parse_golden_numbers_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                cr.parse_golden_numbers(Path(tmp) / "missing.md"), []
            )


class TestParseGoldenStrings(unittest.TestCase):
    def test_parse_golden_strings_extracts_backticked(self):
        body = (
            "# title\n\n"
            "## Правила\n\n"
            "Some prose with a `decoy` token that must be ignored.\n\n"
            "## 1. Имена моделей\n\n"
            "- [ ] `paraphrase-multilingual-MiniLM-L12-v2`\n"
            "- [ ] `gpt-oss-120b`\n"
            "- (дополнить при наполнении)\n"
            "- [x] `Claude Sonnet 4.5`\n\n"
            "## 2. Parquet\n\n"
            "- [ ] `headline_v3e_after_fix.parquet`\n"
            "- [ ] (наполнить из `placeholder.parquet` — это комментарий)\n\n"
            "## Чек-листы\n\n"
            "- [ ] `must-not-be-picked.txt`\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "GOLDEN_STRINGS.md"
            _write(path, body)
            result = cr.parse_golden_strings(path)
        self.assertEqual(
            result,
            [
                "paraphrase-multilingual-MiniLM-L12-v2",
                "gpt-oss-120b",
                "Claude Sonnet 4.5",
                "headline_v3e_after_fix.parquet",
            ],
        )
        self.assertNotIn("decoy", result)
        self.assertNotIn("must-not-be-picked.txt", result)
        self.assertNotIn("placeholder.parquet", result)


# ---------------------------------------------------------------------------
# Integration tests via main()
# ---------------------------------------------------------------------------

class TestMainIntegration(unittest.TestCase):
    """Drive ``cr.main`` end-to-end with patched argv."""

    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["check_regression.py", *argv]):
            with redirect_stdout(buf):
                code = cr.main()
        return code, buf.getvalue()

    # --- Setup helpers ---------------------------------------------------

    def _build_workspace(self, tmp: Path,
                        numbers_body: str,
                        strings_body: str,
                        docx_paragraphs: list[str] | None,
                        notebook_md: list[str] | None,
                        notebook_code: list[str] | None = None) -> dict:
        numbers_path = tmp / "GOLDEN_NUMBERS.md"
        strings_path = tmp / "GOLDEN_STRINGS.md"
        _write(numbers_path, numbers_body)
        _write(strings_path, strings_body)
        docx_path = tmp / "VKR.docx"
        nb_path = tmp / "thesis.ipynb"
        if docx_paragraphs is not None:
            _make_docx(docx_path, docx_paragraphs)
        if notebook_md is not None:
            _make_notebook(nb_path, notebook_md, notebook_code)
        return {
            "numbers": numbers_path,
            "strings": strings_path,
            "docx": docx_path,
            "notebook": nb_path,
        }

    def _argv(self, paths: dict) -> list[str]:
        return [
            "--docx", str(paths["docx"]),
            "--notebook", str(paths["notebook"]),
            "--numbers", str(paths["numbers"]),
            "--strings", str(paths["strings"]),
            "--quiet",
        ]

    # --- Cases -----------------------------------------------------------

    def test_true_positive_docx(self):
        numbers = (
            "## Колонки\n\n"
            "| Число | Файл | Раздел | Происхождение |\n"
            "|---|---|---|---|\n"
            "| `93,81 %` | `04.md` | §4.3 | n/a |\n"
        )
        strings = (
            "## 1. Models\n\n"
            "- [ ] `XGBoost`\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._build_workspace(
                Path(tmp),
                numbers_body=numbers,
                strings_body=strings,
                docx_paragraphs=[
                    "Итоговая точность каскада 93,81 % при сохранении охвата.",
                    "Классификатор XGBoost обучен на эмбеддингах.",
                ],
                notebook_md=["Empty notebook"],
            )
            code, out = self._run_main(self._argv(paths))
        self.assertEqual(code, 0, msg=out)
        self.assertIn("all 2 golden anchors found", out)

    def test_true_negative_docx(self):
        numbers = (
            "## Колонки\n\n"
            "| Число | Файл | Раздел | Происхождение |\n"
            "|---|---|---|---|\n"
            "| `93,81 %` | `04.md` | §4.3 | n/a |\n"
        )
        strings = "## 1. Models\n\n- [ ] `XGBoost`\n"
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._build_workspace(
                Path(tmp),
                numbers_body=numbers,
                strings_body=strings,
                docx_paragraphs=["Unrelated content without the number."],
                notebook_md=["Also unrelated."],
            )
            code, out = self._run_main(self._argv(paths))
        self.assertEqual(code, 1, msg=out)
        self.assertIn("[FAIL] golden number '93,81 %'", out)
        self.assertIn("[FAIL] golden string 'XGBoost'", out)
        self.assertIn("MISSING", out)

    def test_true_positive_with_nbsp(self):
        # Golden file: regular space; DOCX: NBSP. Normalisation must bridge it.
        numbers = (
            "## Колонки\n\n"
            "| Число | Файл | Раздел | Происхождение |\n"
            "|---|---|---|---|\n"
            "| `93,81 %` | `04.md` | §4.3 | n/a |\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._build_workspace(
                Path(tmp),
                numbers_body=numbers,
                strings_body=_empty_golden_strings(),
                docx_paragraphs=["Headline: 93,81\xa0% точность."],
                notebook_md=["irrelevant"],
            )
            code, out = self._run_main(self._argv(paths))
        self.assertEqual(code, 0, msg=out)
        self.assertIn("all 1 golden anchors found", out)

    def test_notebook_only_anchor_counts(self):
        strings = (
            "## 1. Parquet\n\n"
            "- [ ] `cv_stability_10seed.parquet`\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._build_workspace(
                Path(tmp),
                numbers_body=_empty_golden_numbers(),
                strings_body=strings,
                docx_paragraphs=["DOCX without the parquet name."],
                notebook_md=["Markdown only."],
                notebook_code=[
                    "df = pd.read_parquet('cv_stability_10seed.parquet')",
                ],
            )
            code, out = self._run_main(self._argv(paths))
        self.assertEqual(code, 0, msg=out)
        self.assertIn("all 1 golden anchors found", out)

    def test_empty_golden_files_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._build_workspace(
                Path(tmp),
                numbers_body=_empty_golden_numbers(),
                strings_body=_empty_golden_strings(),
                docx_paragraphs=["any content"],
                notebook_md=["any content"],
            )
            code, out = self._run_main(self._argv(paths))
        self.assertEqual(code, 0, msg=out)
        self.assertIn("no golden items configured", out)

    def test_missing_both_sources_exits_2(self):
        # Provide a non-empty golden list so we actually try to look something up.
        numbers = (
            "## Колонки\n\n"
            "| Число | Файл | Раздел | Происхождение |\n"
            "|---|---|---|---|\n"
            "| `42` | `x.md` | s | n/a |\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = {
                "numbers": tmp_path / "GOLDEN_NUMBERS.md",
                "strings": tmp_path / "GOLDEN_STRINGS.md",
                "docx": tmp_path / "absent.docx",
                "notebook": tmp_path / "absent.ipynb",
            }
            _write(paths["numbers"], numbers)
            _write(paths["strings"], _empty_golden_strings())
            code, out = self._run_main(self._argv(paths))
        self.assertEqual(code, 2, msg=out)
        self.assertIn("[WARN] docx not found", out)
        self.assertIn("[WARN] notebook not found", out)
        self.assertIn("neither DOCX nor notebook", out)


if __name__ == "__main__":
    unittest.main()
