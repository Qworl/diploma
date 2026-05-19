"""Build notebooks/01_phase3_session_log.ipynb — single source of truth for this session."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src))


md("""# Phase 3 — Corpus expansion + promptfix re-audit + leak-free CV

**Session date**: 2026-05-17 / 2026-05-18
**Branch**: `phase2-recomputes-on-v2-gold`
**Goal**: устранить системные источники шума в hybrid cascade и получить честные cross-validated цифры.

## Ход работы
1. v2 vs v3 honest 80/20 holdout (предыдущая сессия) — v3 +2.57pp acc, -11pp cov
2. **v3b (tier-weighted)** silver=1, t1=6, t2=4, t3=2 — best на single holdout (89.13%) и на OFF-truth (78.04%)
3. v3c (silver=0) — slightly worse, silver НЕ просто шум
4. **OFF-derived truth** ×27 expansion eval (40k rows для derived attrs)
5. **Silver-leak fix**: trainer передавал silver labels holdout-кодов → inflate ~0.57pp
6. **5-fold leak-free CV** на Tier1+2: 88.45% ± 0.57% (95% CI ±0.50pp)
7. **Promptfix re-audit**:
   - Прежний промпт не содержал nutriments → Opus/gemini заполняли nutri_score/protein_class в 4-7% случаев
   - Pilot подтвердил: с фиксом — 90-97% fill
   - Full re-audit: Opus 5,930 codes ($60+12=72), gemini 34,433 codes ($15)
8. **v3d retrain** на чистом корпусе (Tier 1 promptfix + Tier 3 promptfix, no gpt-5.5)

## Cost summary
| Item | Cost |
|---|---|
| Phase 1 historical | ~$200 (квота исчерпана) |
| Opus retry v2 promptfix | ~$11 |
| Opus expand 4.5 (5283 codes) | ~$60 |
| Opus expand retry | ~$12 |
| Gemini retry promptfix | ~$15 |
| GPT-5.5 retry (killed early) | ~$4 |
| **Total session** | **~$102** |
""")

code("""import warnings; warnings.filterwarnings('ignore')
import json, glob
import pandas as pd
import numpy as np
from pathlib import Path
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)""")

md("## 1. Lineage labels")

code("""# Tier hierarchy after all annotation rounds
print('SILVER (OFF tags-derived, no LLM):')
for cat in ['pasta','chocolate','cheeses']:
    s = pd.read_parquet(f'../datasets/processed/{cat}_stratified_silver_standard.parquet')
    print(f'  {cat}: {len(s):,} rows')

print('\\nTIER 1 — Opus blind OFF-grounded:')
print('  Phase 1 (old prompt): 717 codes (deprecated)')
print('  Phase 1 PROMPTFIX (new prompt): 717 + retry = ~743 codes')
print('  Expand 4.5 PROMPTFIX: 5,313 codes (4359+954 retry)')
print('  Total Tier 1 (promptfix): 6,056 codes (3,058× expansion vs original 239/cat)')

print('\\nTIER 2 — gpt-5.5 expansion (deprecated for v3d):')
print('  1,949 codes (old prompt, derived attrs mostly null)')
print('  gpt-5.5 promptfix retry KILLED — too slow/expensive (~$45 + 2h)')

print('\\nTIER 3 — gemini-2.5-flash B3 (38k codes):')
print('  Old prompt: 38,747 codes (used in v3/v3b/v3c)')
print('  Promptfix: 34,433/38,947 codes (88.4% coverage)')

print('\\nDIRECT LLM (production validation, partner_input — без OFF):')
print('  sonnet-4.5, gpt-4o, gemini-2.5-flash, gpt-oss-120b pilots (~200-300 codes/cat)')
print('  Used as semi-independent ground truth, NOT in training')""")

md("## 2. v2 vs v3 vs v3b vs v3c — single 80/20 holdout (Tier1+2, OLD silver-leak)")

code("""cmp = pd.read_parquet('../datasets/processed/holdout_eval_4way.parquet')
overall = pd.DataFrame([{
    'acc_v2': cmp['acc_v2'].mean(), 'cov_v2': cmp['cov_v2'].mean(),
    'acc_v3': cmp['acc_v3'].mean(), 'cov_v3': cmp['cov_v3'].mean(),
    'acc_v3b': cmp['acc_v3b'].mean(), 'cov_v3b': cmp['cov_v3b'].mean(),
    'acc_v3c': cmp['acc_v3c'].mean(), 'cov_v3c': cmp['cov_v3c'].mean(),
}])
print('OVERALL MEAN (NOTE: contains silver-leak ~0.5pp inflate):')
print(overall.to_string(index=False, float_format='%.4f'))
print('\\nWinner: v3b on accuracy (+2.94pp over v2)')""")

md("""## 3. OFF-derived truth holdout (×27 expansion)

Для derived attrs (nutri_score_grade, protein_class, fat_class) истина = детерминированная формула из OFF nutriments. 56,756 truth-строк vs 1,600 в Tier1+2 holdout.""")

code("""truth = pd.read_parquet('../datasets/processed/off_derived_truth.parquet')
print(f'Total OFF-derived truth rows: {len(truth):,}')
print('Per category × attr:')
print(truth.groupby(['category','attr']).size().unstack(fill_value=0))""")

code("""off_eval = pd.read_parquet('../datasets/processed/eval_off_truth_4way.parquet')
print('Per-attr (acc on 4500+ OFF-derived truth obs):')
cols = ['category','attr','n_total','n_v2','acc_v2','cov_v2','acc_v3b','cov_v3b','acc_v3c','cov_v3c']
print(off_eval[cols].to_string(index=False, float_format='%.3f'))""")

code("""# Overall mean on OFF-truth (4500+ obs, robust CI)
print('OVERALL MEAN (OFF-truth holdout, post-leak-fix):')
print(f'  v2:  acc={off_eval[\"acc_v2\"].mean():.4f}  cov={off_eval[\"cov_v2\"].mean():.4f}')
print(f'  v3:  acc={off_eval[\"acc_v3\"].mean():.4f}  cov={off_eval[\"cov_v3\"].mean():.4f}')
print(f'  v3b: acc={off_eval[\"acc_v3b\"].mean():.4f}  cov={off_eval[\"cov_v3b\"].mean():.4f}')
print(f'  v3c: acc={off_eval[\"acc_v3c\"].mean():.4f}  cov={off_eval[\"cov_v3c\"].mean():.4f}')
print('\\nKey finding: cheeses/fat_class v2->v3+ = +24.7pp (gemini unlocked)')
print('Regression: chocolate/protein_class -13pp (gemini old-prompt nulled — v3d should fix)')""")

md("""## 4. Silver-leak finding + fix

**Bug**: trainer передавал `silver_keep` без фильтра по holdout-кодам. Для каждого holdout-кода silver label (`~85-90% acc`) ВКЛЮЧАЛСЯ в обучение, потом тот же код тестировался → утечка.

**Fix**: добавлен `--holdout-codes` arg в `train_hybrid_cascade.py`. Silver и gold отфильтровываются.

**Magnitude**: leak boosted accuracy by ~0.57pp (fold 0 leaky=89.22% → leak-free=88.65%).

n_silver упал на 17-32% за счёт исключения holdout-кодов:
- pasta/grain_type: 334 → 226 (-108)
- pasta/is_filled: 539 → 378 (-161)
- pasta/nutri_score_grade: 786 → 654 (-132)
""")

md("## 5. 5-fold CV leak-free (v3b config)")

code("""cv = pd.read_parquet('../datasets/processed/cv5fold_v3b.parquet')
print('Per-fold:')
print(cv.to_string(index=False, float_format='%.4f'))
print(f'\\nAccuracy: {cv[\"acc\"].mean():.4f} ± {cv[\"acc\"].std(ddof=1):.4f}')
print(f'Coverage: {cv[\"cov\"].mean():.4f} ± {cv[\"cov\"].std(ddof=1):.4f}')
# Bootstrap CI
np.random.seed(42)
bs = sorted(np.random.choice(cv['acc'].values, size=(1000, len(cv)), replace=True).mean(axis=1))
print(f'Bootstrap 95% CI: [{bs[25]:.4f}, {bs[975]:.4f}]')""")

md("""## 6. Promptfix re-audit — pilot validation

Старый промпт не содержал `nutriments` block → derived attrs (nutri_score, protein, fat) почти всегда null.
Pilot на 30 codes/cat показал драматический эффект:

| attr | Phase 1 fill | Promptfix fill | Δ |
|---|---|---|---|
| pasta/nutri_score_grade | 4% | 97% | +93pp |
| pasta/protein_class | 4% | 97% | +93pp |
| chocolate/nutri_score_grade | 7% | 90% | +83pp |
| chocolate/protein_class | 7% | 90% | +83pp |
| cheeses/fat_class | 97% | 100% | +3pp |

→ Full re-audit запущен: Opus 5,930 codes (~$72), gemini 34,433 codes (~$15).
""")

md("## 7. v3d build (consensus_v3d.parquet)")

code("""v3d = pd.read_parquet('../datasets/processed/consensus_v3d.parquet')
print(f'v3d: {len(v3d):,} rows, {v3d[\"code\"].nunique():,} codes')
print('\\nPer category × tier (codes):')
print(v3d.groupby(['category','tier'])['code'].nunique().unstack(fill_value=0))""")

md("## 8. v3d eval (TODO — fill after retrain + eval)")

code("""# After v3d retrain finishes, this loads new comparison
try:
    v3d_eval = pd.read_parquet('../datasets/processed/holdout_eval_v3d_vs_baselines.parquet')
    print(v3d_eval.to_string(index=False, float_format='%.4f'))
except FileNotFoundError:
    print('Pending: v3d retrain in progress. Re-run this cell after completion.')""")

md("""## 9. Open / Pending

- [ ] v3d retrain + eval (in progress)
- [ ] 5-fold CV repeat on v3d
- [ ] Add v3d to OFF-truth eval
- [ ] Final cost reconcile via OpenRouter dashboard
- [ ] Update §6 of thesis with new numbers + lineage diagram
- [ ] Decide whether to keep gpt-5.5 Tier 2 in narrative (currently abandoned for v3d)

## Files / artifacts produced this session

```
src/experiments/
  build_off_derived_truth.py       — OFF nutriments → derived truth (56k rows)
  build_session_notebook.py        — this notebook builder
  cv_5fold_hybrid.py               — 5-fold CV with leak-free trainer
  eval_4way_v2_v3_v3b_v3c.py       — single-holdout 4-way comparator
  eval_holdout_v2_vs_v3.py         — v2-vs-v3 first comparator
  eval_off_truth_holdout.py        — OFF-truth 4-way evaluator
  extend_embeddings_b3.py          — B3 codes → silver + embeddings
  extend_embeddings_off_truth.py   — OFF-truth codes → silver + embeddings
  make_holdout_split.py            — stratified 80/20 split (+ tier column)
  merge_hybrid_v3.py               — v2 + Tier3 → consensus_hybrid_v3
  merge_v3d_corpus.py              — consensus_v3d builder (Opus promptfix + gemini promptfix)
  train_hybrid_cascade.py          — + tier-weighted, + leak-free (--holdout-codes)

models_backup/
  v2_clean/, v3_clean/, v3b_weighted/, v3c_no_silver/   — 44 .pkl each
  full_snapshot_2026-05-17/                              — 523 файла (229MB)

datasets/processed/
  consensus_holdout.parquet, consensus_v2_train.parquet, consensus_v3_train.parquet
  consensus_hybrid_v3.parquet, consensus_v3d.parquet
  holdout_eval_v2_vs_v3.parquet, holdout_eval_4way.parquet
  off_derived_truth.parquet, eval_off_truth_4way.parquet
  cv5fold_v3b.parquet
```
""")

nb["cells"] = cells
out_path = Path("notebooks/01_phase3_session_log.ipynb")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path} with {len(cells)} cells")
