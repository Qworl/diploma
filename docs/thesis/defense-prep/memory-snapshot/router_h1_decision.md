---
name: Router H1 decision 2026-05-13
description: H1 PASS/FAIL verdict on verified strong-seed gold + Bonferroni; determines Phase F narrative path
type: project
---

**H1 verdict: FAIL** on 2026-05-13.

Pre-registered evaluation (Bonferroni @α/3 = 0.0167) on n=1539 brand-disjoint test slice with strong-seed verified gold (Sonnet 4.5 + GPT-4o + Gemini 2.5 Flash consensus):

| Budget | Δ (router − static) | p_McNemar | 95% CI |
|---|---|---|---|
| 25% | +0.3 пп | 0.737 | [−1.4, +1.8] |
| 40% | +0.5 пп | 0.488 | [−1.0, +1.9] |
| 50% | +0.1 пп | 1.000 | [−0.9, +1.0] |

**Significant budgets: []** → 0 of 3 budgets pass Bonferroni-corrected α=0.0167.

LOCO cross-domain (mean Δ): −3.6 пп @25%, −3.9 @40%, −4.1 @50%. Cosmetics −11.95 пп (catastrophic). Only pasta shows positive transfer (+2.18 пп @50%).

**Why:** Router learns legitimate signal (is_rare_class, cascade_layer_none) and adds +0.024 AUC over cascade_conf alone (0.847→0.871). 18-26% of routing decisions differ between router and static threshold, and on those router is +8-10 пп smarter (lower wasted-LLM-on-correct-cascade rate). But net effect on full test (~+0.3-0.5 пп) is below noise floor with n=1539 after Bonferroni correction.

**Brand-disjoint split revealed brand-attr leakage** in original §6.14: brand_attr_acc_table lookup hit-rate on test = 0.0% (343 entries, all unseen brands). Previous +1.4-1.8 пп gain was partially this leakage.

**Plan B4 narrative path active.** Main contribution reframes from "learned router" to "production-ready cost-aware enrichment system":
- Per-attr static table at 58% LLM cost achieves 83.9% accuracy — best Pareto point.
- Learned router is one studied strategy with honest negative result.
- Brand-disjoint methodology + per-source taxonomy + pre-registration are methodological contributions.

**Phase F text revisions** (notebook §6.14.7 + abstract + §0 + §7) must use Path B language.
