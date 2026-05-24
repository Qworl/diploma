# Critic Reviews — Defense Prep Spec (2026-05-24)

Two-pass critic review on the spec. Reviews kept verbatim for honesty / future reference.

---

## Pass 1 — Review of v1

### Top 3 blockers

**1. Slide 4 «5 задач» is academically circular and will get torched.** The verbs/outcomes table had classic Russian-thesis tautologies: «Формализовать → Каскадная архитектура» (формализация = архитектура), «Разработать слои → Слои L1-L4» (literally restated), «Реализовать комплекс → Демо-стек» (synonyms). МАИ committee will ask: "Так в чём задача, а в чём результат?" The "проверить" task also conflates *методологию* (brand-disjoint protocol) with *результат* (93.81%) — those are two different things. Task 1 «выделен пробел: per-attr policy с честным negative result» is also nonsensical — *negative result* is not a *пробел в литературе*, it's *your finding*. Rewrite as: task = action on a problem object, result = quantifiable artefact. Currently they are nominalized restatements.

**2. Backup B1 (H1 router FAIL) framed as a feature is a footgun without explicit defensive scaffolding.** Spec §4.2 row B1 had a trigger question «мало данных?» — but no answer architecture. МАИ committee (esp. 806, инженерная) does NOT reward "честный negative result" the way an arxiv reviewer does; they read it as "вы не справились с задачей". The router H1 FAIL (memory `router_h1_decision.md`: p=0.488–1.000 on n=1539) is in fact a *strong methodological result*, but the spec didn't say HOW you frame it on Slide 11 bullet («H1 router отклонена»). One-line bullet on Slide 11 + B1 backup is not enough. You need a positive reframe ON MAIN slide: "Доказано: per-attr static policy достаточна, обучаемый маршрутизатор не оправдывает усложнения — экономия inference" — i.e. it's a *design decision*, not a *failed experiment*.

**3. Headline 93.81% @ +5% LLM cost had no CI on Slide 10 and the spec didn't even mention this gap.** §4.1 row 10 said «таблица с зелёными %». The committee's first quantitative question is *always* "доверительный интервал?" or "разброс по доменам?". Memory `accuracy_squeeze_deploy_2026-05-20.md` shows the number is *holdout-defended* — but defended ≠ CI'd. With n=1539 and headline +1.03 пп over prior, the binomial CI half-width is ~±1.2 пп — meaning 93.81% and 92.78% overlap. You need either Wilson CI on the slide or a McNemar p-value, otherwise the headline is statistically unanchored. Spec was silent on this.

### Top 3 redundancies

**1. Three TikZ diagrams D1/D2/D3 for a 1-month deadline is over-engineered.** D2 and D3 are 90% the same content (5-layer cascade) — D3 just adds library names. §7.1 admitted this («5 блоков, как у D2»). Merge into one TikZ + one PNG with stack overlay. D1 (partner inputs vs outputs) is also solvable with a simple two-column table, not TikZ. The spec's own §11 risk row 1 confirms this concern.

**2. `01_demo_live.ipynb` with 5 cells × 3 SKUs = effectively a print of cached results.** §6.2 said "Outputs закоммитить — если live re-run упадёт, jury всё равно увидит сохранённые результаты." If you commit outputs, it is *not a live demo* — it's a notebook viewer. МАИ defenses with 7-10 min talk slot rarely have time for a notebook walkthrough anyway. Drop the live demo; replace with 1 slide of pre-baked screenshots from `00_thesis_main.ipynb` (which already exists). Saves a day of work and a risk vector.

**3. Speaker notes via beamer `\note{}` + separate notes-PDF target in Makefile is two-source-of-truth pretending to be one.** §8 + §3.3 set up `\setbeameroption{show only notes}` AND `show notes on second screen`. МАИ rooms have one projector, one HDMI cable, sometimes no presenter view. You will print the notes-PDF. So `\note{}` is just an awkward Markdown — write a plain `SPEAKER_NOTES.md`, save half a day of beamer plumbing.

### Hidden assumptions

- **Spec assumed МАИ defense format = 7-10 min talk → 14 slides @ ~40 sec each (§4.3).** No source. Кафедра 806 magistrant defenses are commonly 10–12 min + Q&A; 14 content slides is doable but tight if the chair runs strict. Verify with научный руководитель *before* finalizing structure.
- **Spec assumed `\includepdf` for ВКР task PDF can wait (§2 out of scope + line 40 «оставлен закомментированным до получения PDF от кафедры»).** Without the *задание* signed by заведующий кафедрой Крылов С.С. (memory `thesis_state.md`), the bound thesis is *incomplete* per ОД-093-СМК-ПОЛ-001-Ф form templates. This is not a defense-day item — it must be obtained 2-3 weeks before, signed, scanned, and bound. Spec treated it as optional, which is wrong.
- **Spec assumed the slide miniframes theme + seahorse color combine cleanly (§3.3 + §11 risk row 2).** Half-day fight, no fallback documented beyond «готов fallback».
- **Spec assumed «3 main + 3 extensions» reframing is purely textual (§5.3 — 4 files).** Memory `thesis_state.md` line 60 said «22 целевых атрибута в трёх категориях OFF (pasta 8, chocolate 7, beverages 7)» — i.e. earlier text used beverages as main, not cheeses. If you swap cheeses in as "main", all attribute counts and per-domain Pareto numbers shift. Spec didn't mention this audit.
- **Spec assumed the committee won't ask «почему 3 а не 7?».** §3.2 just said «зафиксировано в §3.1 текста ВКР». That's circular — *because we wrote it that way* is not a defense.

### What's missing

- **Отзыв научного руководителя and рецензия рецензента** — both required МАИ documents, both have hard deadlines (typically 7-10 days before defense for the reviewer copy). Not mentioned anywhere.
- **Антиплагиат-проверка**: §12 said "out of scope (отдельная процедура)" — but the spec also didn't say *when* it gets done. Per CLAUDE.md «оригинальность ≥80%, заимствования ≤15%» is a hard requirement. Needs explicit milestone, not "out of scope".
- **Регистрация на защиту** in расписание ГИА — typically 2 weeks before, with submitted PDF copy.
- **Раздаточный материал** (handout) — §12 dismissed as "одна команда pdfjam в день защиты". МАИ кафедра 806 historically asks for *7 копий раздатки* (по числу членов комиссии) — printed and stapled, not just generated. Day-of with no printer access = disaster.
- **Bound paper copy** of the diploma — physically required at МАИ defense, ~2-3 day turnaround at a binding shop. Not in checklist §9.
- **Numbers consistency between slides and thesis text.** Memory `accuracy_squeeze_deploy_2026-05-20.md` line 82-83 EXPLICITLY warned: "Thesis text (главы 3.3.2, 3.3.3, 3.3.4) — всё ещё содержит цифры 92,8 % / 90,5 % / 8,6 % / 3,3 %". Spec didn't include a text-sync task. If slides say 93.81% and the bound diploma the committee is *reading* says 92.78%, you will be eaten alive.
- **`reproduce.sh`** is mentioned on Slide 14 (table row 4 «Реализовать → reproduce.sh, репозиторий на GitHub»). Does it exist? Spec didn't audit.
- **«Справка о внедрении»** — CLAUDE.md says «Желательно: справка о внедрении/использовании». Spec dismissed ("внедрение синтетическое"). If you have *any* employer letterhead — it adds points on Глава 4.

### Verdict

**Needs revision — not rewrite, but ~30% of the spec is wrong or hollow.** Core architecture (14 slides, TikZ-light, separate demo notebook) is defensible, but blockers 1-3 (circular tasks, undefended negative result, missing CI on headline), the bureaucratic gaps (отзыв/рецензия/раздатка/binding/задание ВКР), and the un-synced 92.78 vs 93.81 numbers in the actual bound thesis would each tank the defense individually. Fix those before any TikZ work begins.

---

## Pass 2 — Review of v2

### Author's self-found errors

**Error 1 (n=1539 vs n=4350):** **CONFIRMED.** Chapter 3 line 132 fixes n=4350 on `consensus_gold_v2_expanded.parquet` (holdout, 22 pairs, brand-disjoint). Line 392 fixes n=1539 on `router_pareto_gold.parquet` for the H1 router pre-registered test. These are independent artifacts; the v2 spec §4.3 wording "подмножество основного holdout n=4350" is factually wrong and will be torn apart at defense. Fix: drop the subset claim entirely, just say "на router-test n=1539".

**Error 2 (720× / +5 % LLM cost outdated):** **CONFIRMED.** Memory dated 2026-05-20 explicitly logs LLM call rate 3.3 % → 8.2 % (+4.9 пп = +149 % relative), cost ratio 720× → ≈290×, slides 10/13/14/15/18 already updated. Spec §3.1 sync table keeps "3,47 %" and "720×" everywhere — completely wrong. The "+5 % LLM cost" phrasing in §3.1 is a misreading of "+4.9 пп" (absolute) as "5 % relative".

**Error 3 (gpt-oss numbers):** **CONFIRMED.** Memory: 92.3 % → 93.3 %, +23.0 пп → +24.0 пп vs direct Sonnet. Spec §3.1 number-sync table omits gpt-oss entirely; §4.2 slide 9 table still shows "92,3 %" / "+23,0 пп" implicitly via the unchanged baseline.

### v1 fixes assessment

- **§4.1.1 «5 задач» — PARTIAL FIX.** Tasks 1, 2, 4 are non-circular. **Task 3 still circular**: "Спроектировать и обучить слои каскада" → "4 обученных слоя" — result is literally the deliverable of the action. The .pkl naming doesn't change the tautology. **Task 5 still half-circular**: action says "провести оценку", result says "числа + отвергнутая H1" — number IS the output of the eval. Better framing: result = "пересмотр архитектуры (отказ от обучаемого роутера) на основе предзарегистрированного теста".

- **§4.3 H1 reframe — WEAK.** "Доказано: per-attr static policy достаточна" is the classic affirming-the-consequent dressed up. With MDE ≈4.4 пп (line 417 chapter 3 already concedes this), the spec is committing to a claim the thesis itself walks back. A hostile reviewer ("Лебедев-style") will ask: "Вы доказали достаточность или у вас не хватило мощности?" Spec has no rebuttal prepared. Recommend softer wording: "**Не обнаружено** превосходства обучаемого маршрутизатора при MDE ≈4 пп; для production выбрано статическое правило по принципу простоты."

- **§4.2 Wilson CI — STILL BROKEN.** User is correct: Wilson CIs at 92.78 and 93.81 overlap; overlap test cannot establish significance. Spec mentions "McNemar p-value vs baseline" — but baseline there is **all-sonnet 83,8 %**, not **92,78 % pre-deploy**. The relevant paired test is post-deploy vs pre-deploy, on the SAME 4350 ячеек. Spec doesn't specify this paired McNemar. As written, the +1.03 пп gain is statistically undefended.

- **§13 bureaucracy tracker — MOSTLY SOLID** but missing: (a) дата выдачи приказа об утверждении тем ВКР, (b) подпись на титульнике научного руководителя (отдельная встреча), (c) явный пункт «нормоконтроль кафедры» (обычно T-7 дней, между антиплагиатом и переплётом), (d) electronic подпись для NTB МАИ загрузки.

### New v2 issues

1. **Slide 9 (§4.2) inconsistency:** "Каскад + gemini-flash | 93,81 % | **3,47 %** | **720×**". Memory's post-deploy: 8,2 % LLM rate, ≈290× cheaper. Either roll forward (use 8.2/290×) or roll back (use 92.78/3.3/720×) — current spec is a Frankenstein.
2. **§4.1.2 row 5** keeps "93,81 % при **720×** сокращении" — same Frankenstein.
3. **§14 Q3** to advisor is muddled and doesn't list the actual trade-off ("+1,03 пп accuracy vs 2,5× больше LLM-вызовов"). Committee will absolutely ask "почему увеличили LLM cost?" — spec has no defensive bullet.
4. **§3.1 missing entries:** `0-abstract.tex` doesn't only have 92,8 % at line 14/16 — need to verify other abstract numbers (3,3 %, 720×, +9,0 пп) all sync. Spec lists 8 lines, real list is likely 15+.
5. **§5.2 QR appendix:** OK, but `\appendixsection` is not a standard command — depends on which class file. Need to verify it isn't `\section`.
6. **§6 «notebook не трогаем»** conflicts with §3.1 number sync — if the notebook contains 92,8 %, it must be re-run or annotated. Spec doesn't address.
7. **Slide 11 demo screenshots:** brittle — requires `ml_service` running. No fallback plan if Docker-compose fails. Should pre-bake screenshots into repo NOW.
8. **Timing:** 14 main slides ÷ ~10 min report = ~43 sec/slide. §4.2 slide 9 SPEAKER_NOTES says "50 сек". Sum: ~14×45 = 10:30. Plus titlepage. **Tight, no slack.** МАИ comms ВКР defense typically allows 7–10 min. Spec doesn't commit to a target time.
9. **§4.1.1 Task 1**: 6 critеria list includes "voice deployment" — that's electronics cold-start, not in the 3 main domains. Mismatch with §3.2 scope.

### Critical recommendation: 93,81 OR 92,8?

**Roll back to 92,8 % @ 720×.** Reasoning for МАИ engineering ВКР:

| | 92,8 % @ 720× (pre-deploy) | 93,81 % @ 290× (post-deploy) |
|---|---|---|
| Cleanness of story | Single Pareto-dominating point | Trade-off requires explanation |
| Numbers consistency | Thesis text already says this | Requires sync at 15+ places + Wilson recompute + paired McNemar |
| LLM dependency | 3,3 % calls — sells "независимость от LLM" | 8,2 % — committee asks "зачем LLM, если каскад сам справляется?" |
| Risk at defense | Low | Medium — +1,03 пп gain inside MDE noise floor |
| Bureaucratic risk | Zero rework | Антиплагиат, нормоконтроль уже могут быть на старом числе |

For an МАИ инженерная ВКР, the "**720× cheaper** при 92,8 %" headline is the cleaner Pareto win and matches what's already in `report/`. The +1,03 пп is statistically marginal (within MDE), and the 2,5× LLM call increase is hard to defend. The accuracy-squeeze deploy is a great engineering achievement for production, but a **net storytelling negative** for a 10-minute defense.

**Recommended:** roll text+slides back to 92,8 %, mention accuracy-squeeze as a **bullet on slide 12 ("Перспективы внедрения: режим accuracy-priority +1,03 пп при 2,5× LLM-вызовов")** — present as a knob, not as headline.

### Verdict

**NEEDS ANOTHER PASS.** Three things must land before plan execution:
1. Resolve the headline number choice (recommend 92,8 %).
2. Fix Task 3 and Task 5 circularity properly.
3. Replace Wilson-overlap reasoning with explicit paired McNemar on 4350 cells in §4.2 (or drop the +1.03 claim altogether if rolling back).

---

## Disposition (v3)

All Pass 2 structural findings are addressed in v3:
- Task 3 and Task 5 properly rewritten (non-tautological)
- H1 reframe softer («не обнаружено превосходства при MDE ≈4 пп»)
- Paired McNemar vs pre-deploy specified, Wilson CI overlap removed as significance test
- Frankenstein numbers eliminated → all numbers as placeholders
- Bureaucracy tracker expanded (нормоконтроль, e-signature NTB МАИ, приказ о темах)
- Slide 11 pre-baked screenshots NOW (Phase 1), not on defense day
- Task 1 "voice deployment" → "локальное развёртывание"
- Timing fallback (compression to 10-12) documented in §11

Headline number choice (92,8 vs 93,81) — left to user; v3 spec works with either via placeholder mechanism.
