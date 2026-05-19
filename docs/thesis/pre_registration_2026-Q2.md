# Pre-registration: Phase 1 + Phase 3 hypothesis tests (2026-Q2)

**Цель документа:** зафиксировать гипотезы и decision rules ДО запуска
экспериментов. Метка времени commit-hash подтверждает, что гипотезы
не подгонялись под результаты post-hoc.

**Авторская подпись:** Mikhail Frolov, 2026-05-15.

## Phase 1 hypotheses (risk-first)

### E1 — Layer 4 circularity check
**H_E1:** При замене Layer 4 LLM с `gpt-oss-120b` на `claude-sonnet-4-5`
(член consensus_gold) на том же brand-disjoint consensus_gold test
(n=1539) с тем же per-attribute static policy, Δ headline accuracy
< +2.0 п.п.

**Decision rules:**
- Δ < +1.5 п.п., 95 % bootstrap CI пересекает 0 → циркулярности НЕТ,
  headline 82.1 % валиден.
- +1.5 ≤ Δ < +3.0 п.п. → пограничный случай, abstract получает
  open-weight оговорку.
- Δ ≥ +3.0 п.п. → циркулярность подтверждена, двойной headline в
  abstract.

**Control:** также прогоняется Gemini 2.5 Flash (член консенсуса),
GPT-4o (член консенсуса), Llama-3.2-3b (не член, слабый). Если Sonnet
+ Gemini + GPT-4o cluster выше gpt-oss, а Llama ниже — циркулярность
доказана независимо.

### E2 — Hold-out pasta retrain
**H_E2:** Δ accuracy на override∪manual_only холдаут-ячейках (20 %
brand-disjoint split от pasta_gold_250) после применения silver fix
+ ML retrain ≥ +35 п.п. (теряет ≤14 п.п. от заявленных +49.4).

**Decision rules:**
- Δ ≥ +35 п.п. → +49.4 п.п. репортится как honest (fix не overfit).
- +25 ≤ Δ < +35 → частичный overfit, формулировка изменяется на
  «+35 на independent holdout, +49.4 на full audit set».
- Δ < +25 → fix overfits, принципиальный пересмотр §6.14.7.3.

### E3 — ECE staleness regen
**H_E3:** После silver-extractor fix (`§6.14.7.2`, commits c1a0815 +
4bbd9e2) на ≤ 1 атрибуте из текущего набора пар (cat, attr) ECE
превысит 0.10.

**Decision rules:**
- 0 атрибутов с ECE > 0.10 → §6.1 валиден as-is.
- 1–3 атрибута → применить isotonic regression
  (`CalibratedClassifierCV(method='isotonic', cv=5)`), обновить §6.1.
- ≥ 4 атрибутов → fix сдвинул calibration, добавить thread в §6.14.7.3.

### E5 — Cold-start replication on chocolate + cheeses
**H_E5:** Cold-start headline (95.4 % fill / 91 % accuracy на pasta)
реплицируется на chocolate и cheeses: fill ≥ 90 % и accuracy ≥ 85 %
на обоих доменах vs Opus-audited gold.

**Decision rules:**
- Оба ≥ 0.85 → headline reformulated as «cross-domain cold-start:
  90–95 % fill / 85–91 % accuracy на 3 доменах».
- Хотя бы один в [0.70, 0.85) → headline остаётся «pasta-specific,
  chocolate/cheeses показывают X/Y».
- Оба < 0.70 → cold-start headline переезжает из §1 в §6 как
  domain-dependent observation.

## Phase 3 hypotheses (предзарегистрированы здесь, используются позже)

### E7 — Opus audit expansion per domain
**H_E7 (per domain):** На каждом из {beverages, cereals, cosmetics}
audit→fix→retrain цикл даёт Δ accuracy на override∪manual_only cells
≥ +15 п.п.

**Decision rules per domain:**
- Δ ≥ +15 п.п. → full validation, domain enters §6.18 table as
  "validated".
- 0 < Δ < +15 п.п. → partial.
- Δ ≤ 0 ИЛИ semantic drift без deterministic fix → scope-limit
  (валидный результат: «методология применима к N/6 доменам»).

### E8 — TXtract-style baseline on pasta
**H_E8:** Multi-task encoder с category-conditioning поверх SBERT не
превосходит per-attribute XGBoost на brand-disjoint test более чем
на +1.5 п.п. macro-accuracy.

**Decision rules:**
- Δ < +1.5 п.п. → текущий выбор XGBoost подтверждён эмпирически.
- +1.5 ≤ Δ < +3 п.п. → пограничное, §7.2 direction.
- Δ ≥ +3 п.п. → architectural finding в §1.5 + §7.2 (НЕ
  pipeline-rewrite).

---

## Bonferroni correction

Phase 1 содержит 4 independent тестируемых гипотезы (E1, E2, E3, E5).
Bonferroni-скорректированный α = 0.05 / 4 = 0.0125. Для гипотез,
требующих p-value (E2 paired bootstrap, E1 bootstrap), используется
этот порог. Гипотезы E3, E5 фиксированы через absolute thresholds
без p-value.
