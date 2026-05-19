"""E8: cascade + Layer 4 LLM hybrid accuracy on v2-gold brand-disjoint test.

For each LLM Layer-4 candidate model (sonnet45 / gpt4o / gemini25flash / gptoss /
llama3b) compute per (cat, attr) and aggregate:

* `acc_cascade_e2e`        — current headline definition (abstain = wrong).
* `acc_hybrid_proxy`       — Tier-2 estimate: abstained cells are credited at
                             the LLM's per-attr accuracy (from direct_llm_eval).
* `acc_hybrid_with_router` — multiplies oracle by Layer-0 router_v3 accuracy.

Outputs:
    datasets/processed/cascade_plus_llm4_hybrid.parquet
    datasets/processed/cascade_plus_llm4_summary.parquet

Re-uses only pre-computed parquet artifacts; no LLM API calls.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CATS = ("pasta", "chocolate", "cheeses")
LLMS = ("sonnet45", "gpt4o", "gemini25flash", "gptoss", "llama3b")

PROC = Path("datasets/processed")


def load_cascade(cat: str) -> pd.DataFrame:
    return pd.read_parquet(
        PROC / f"cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet"
    )


def load_gold() -> pd.DataFrame:
    g = pd.read_parquet(PROC / "consensus_gold_v2_expanded.parquet")
    g = g[g["gold_is_null"] == False]  # noqa: E712
    return g[["category", "code", "attr", "gold_value"]].copy()


def load_direct_llm(cat: str, model: str) -> pd.DataFrame | None:
    fp = PROC / f"direct_llm_eval_{cat}_stratified_{model}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    return df


def load_router_v3_acc() -> dict[str, float]:
    h = pd.read_parquet(PROC / "headline_v3e_final.parquet")
    return h.groupby("category")["router_v3_acc"].first().to_dict()


def llm_acc_per_attr(llm_df: pd.DataFrame) -> pd.Series:
    """Per-attr LLM accuracy restricted to cells with non-null gt.

    Falls back to overall mean when an attr has no rows.
    """
    sub = llm_df[llm_df["gt_non_null"] == 1].copy()
    if len(sub) == 0:
        return pd.Series(dtype=float)
    return sub.groupby("attr")["correct_when_both_present"].mean()


def llm_acc_overlap(
    cascade_abstain: pd.DataFrame, llm_df: pd.DataFrame, gold: pd.DataFrame
) -> tuple[float | None, int]:
    """Optional Tier-1 direct overlap accuracy estimate.

    Joins cascade-abstain cells with direct LLM eval on (code, attr) and
    measures LLM accuracy on the intersection that also has gold.
    Only returns a value when ≥30 cells overlap.
    """
    if len(cascade_abstain) == 0 or len(llm_df) == 0:
        return None, 0
    j = cascade_abstain.merge(
        llm_df[["code", "attr", "pred", "predicted_non_null"]],
        on=["code", "attr"],
        how="inner",
    )
    j = j.merge(gold, on=["code", "attr"], how="inner")
    if len(j) < 30:
        return None, len(j)
    j["correct"] = j["pred"].astype(str) == j["gold_value"].astype(str)
    return float(j["correct"].mean()), len(j)


def compute_for_model(
    model: str, gold: pd.DataFrame, router_v3_acc: dict[str, float]
) -> pd.DataFrame:
    rows: list[dict] = []
    for cat in CATS:
        cp = load_cascade(cat)
        gold_cat = gold[gold["category"] == cat][["code", "attr", "gold_value"]]
        merged = cp.merge(gold_cat, on=["code", "attr"], how="inner")
        # restrict to cells with gold (matches headline definition)
        merged["correct_e2e"] = (merged["layer"] != "abstain") & (
            merged["predicted"].astype(str) == merged["gold_value"].astype(str)
        )
        merged["correct_covered"] = merged["predicted"].astype(str) == merged[
            "gold_value"
        ].astype(str)

        llm_df = load_direct_llm(cat, model)
        if llm_df is None:
            continue
        attr_llm_acc = llm_acc_per_attr(llm_df)
        overall_llm_acc = float(
            llm_df.loc[llm_df["gt_non_null"] == 1, "correct_when_both_present"].mean()
        )

        for attr, sub in merged.groupby("attr"):
            n_test = int(len(sub))
            n_abstain = int((sub["layer"] == "abstain").sum())
            n_covered = n_test - n_abstain
            acc_cascade_e2e = float(sub["correct_e2e"].mean())
            if n_covered > 0:
                acc_on_covered = float(
                    sub.loc[sub["layer"] != "abstain", "correct_covered"].mean()
                )
            else:
                acc_on_covered = float("nan")
            coverage = n_covered / n_test if n_test else 0.0

            llm_acc_attr = float(attr_llm_acc.get(attr, overall_llm_acc))

            # Tier 2 proxy
            acc_hybrid_proxy = (
                n_covered * acc_on_covered if n_covered > 0 else 0.0
            ) + n_abstain * llm_acc_attr
            acc_hybrid_proxy = acc_hybrid_proxy / n_test if n_test else float("nan")

            # Tier 1 overlap (optional)
            abstain_sub = sub[sub["layer"] == "abstain"][["code", "attr"]]
            acc_overlap, n_overlap = llm_acc_overlap(
                abstain_sub, llm_df[llm_df["attr"] == attr], gold_cat
            )

            router_acc = router_v3_acc.get(cat, 1.0)
            acc_hybrid_with_router = acc_hybrid_proxy * router_acc

            rows.append(
                dict(
                    category=cat,
                    attr=attr,
                    llm_model=model,
                    n_test=n_test,
                    n_covered=n_covered,
                    n_abstain=n_abstain,
                    coverage=coverage,
                    acc_on_covered=acc_on_covered,
                    llm_acc_on_attr=llm_acc_attr,
                    acc_cascade_e2e=acc_cascade_e2e,
                    acc_hybrid_proxy=acc_hybrid_proxy,
                    acc_hybrid_overlap=acc_overlap,
                    n_overlap=n_overlap,
                    router_v3_acc=router_acc,
                    acc_hybrid_with_router=acc_hybrid_with_router,
                )
            )
    return pd.DataFrame(rows)


def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    """Per-LLM grand aggregates.

    Grand accuracy = micro-average over (cat, attr) cells, weighted by n_test
    so it remains consistent with how headline_v3e_final is summarised.
    """
    out_rows: list[dict] = []
    for model, sub in detail.groupby("llm_model"):
        n_total = int(sub["n_test"].sum())
        grand_cascade = float(
            (sub["acc_cascade_e2e"] * sub["n_test"]).sum() / n_total
        )
        grand_hybrid = float(
            (sub["acc_hybrid_proxy"] * sub["n_test"]).sum() / n_total
        )
        grand_hybrid_router = float(
            (sub["acc_hybrid_with_router"] * sub["n_test"]).sum() / n_total
        )
        grand_coverage = float(
            (sub["coverage"] * sub["n_test"]).sum() / n_total
        )
        out_rows.append(
            dict(
                llm_model=model,
                grand_acc_cascade_only_e2e=grand_cascade,
                grand_acc_hybrid_oracle=grand_hybrid,
                grand_acc_hybrid_with_router=grand_hybrid_router,
                grand_coverage=grand_coverage,
                delta_layer4_pp=(grand_hybrid - grand_cascade) * 100.0,
                n_test_total=n_total,
            )
        )
    return pd.DataFrame(out_rows).sort_values(
        "grand_acc_hybrid_oracle", ascending=False
    )


def per_cat_table(detail: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = detail[detail["llm_model"] == model]
    out = (
        sub.groupby("category")
        .apply(
            lambda x: pd.Series(
                {
                    "n_test": int(x["n_test"].sum()),
                    "coverage": (x["coverage"] * x["n_test"]).sum() / x["n_test"].sum(),
                    "acc_cascade_e2e": (
                        x["acc_cascade_e2e"] * x["n_test"]
                    ).sum() / x["n_test"].sum(),
                    "acc_hybrid_oracle": (
                        x["acc_hybrid_proxy"] * x["n_test"]
                    ).sum() / x["n_test"].sum(),
                    "acc_hybrid_w_router": (
                        x["acc_hybrid_with_router"] * x["n_test"]
                    ).sum() / x["n_test"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return out


def lift_by_attr(detail: pd.DataFrame, model: str, top_k: int = 10) -> pd.DataFrame:
    sub = detail[detail["llm_model"] == model].copy()
    sub["lift_pp"] = (sub["acc_hybrid_proxy"] - sub["acc_cascade_e2e"]) * 100.0
    return sub.sort_values("lift_pp", ascending=False).head(top_k)[
        [
            "category",
            "attr",
            "n_test",
            "coverage",
            "acc_cascade_e2e",
            "llm_acc_on_attr",
            "acc_hybrid_proxy",
            "lift_pp",
        ]
    ]


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    gold = load_gold()
    router_v3_acc = load_router_v3_acc()

    all_details = []
    for model in LLMS:
        d = compute_for_model(model, gold, router_v3_acc)
        all_details.append(d)
    detail = pd.concat(all_details, ignore_index=True)
    summary = summarise(detail)

    out_detail = PROC / "cascade_plus_llm4_hybrid.parquet"
    out_summary = PROC / "cascade_plus_llm4_summary.parquet"
    detail.to_parquet(out_detail, index=False)
    summary.to_parquet(out_summary, index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    print("\n========== A. Aggregate (grand-means, micro-weighted by n_test) ==========")
    show = summary.copy()
    for col in (
        "grand_acc_cascade_only_e2e",
        "grand_acc_hybrid_oracle",
        "grand_acc_hybrid_with_router",
        "grand_coverage",
    ):
        show[col] = (show[col] * 100).round(2)
    show["delta_layer4_pp"] = show["delta_layer4_pp"].round(2)
    print(show.to_string(index=False))

    print("\n========== B. Per-category breakdown (best LLM by hybrid_oracle) ==========")
    best = summary.iloc[0]["llm_model"]
    pc = per_cat_table(detail, best)
    for col in ("coverage", "acc_cascade_e2e", "acc_hybrid_oracle", "acc_hybrid_w_router"):
        pc[col] = (pc[col] * 100).round(2)
    print(f"(model = {best})")
    print(pc.to_string(index=False))

    print("\n========== C. Top-10 attrs gaining most from LLM fallback (best LLM) ==========")
    lift = lift_by_attr(detail, best, top_k=10)
    for col in ("coverage", "acc_cascade_e2e", "llm_acc_on_attr", "acc_hybrid_proxy"):
        lift[col] = (lift[col] * 100).round(2)
    lift["lift_pp"] = lift["lift_pp"].round(2)
    print(lift.to_string(index=False))

    print("\n========== Sanity ==========")
    print(f"cascade_only_e2e grand mean (across all 5 models, identical): "
          f"{summary['grand_acc_cascade_only_e2e'].iloc[0]*100:.2f}%")
    print(f"Best LLM: {best}; lift over cascade-only: "
          f"{summary.iloc[0]['delta_layer4_pp']:.2f} pp")
    print(f"Outputs written:\n  - {out_detail}\n  - {out_summary}")


if __name__ == "__main__":
    main()
