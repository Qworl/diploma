"""LLM validator for TEXT-derived attributes (partner-input only).

Tag-derived attrs are 100% in eval due to circularity (silver=tags, gold=tags).
Text-derived attrs have honest cascade errors (cocoa_percentage ~75%, texture
~77%, etc.) where LLM world knowledge ("Lindt 70% = dark", "parmesan = hard")
can really help.

Workflow:
  1. Take cascade predictions on text-derived attrs
  2. For low-confidence (< 0.85) predictions, ask gpt-oss-120b
  3. LLM sees ONLY partner input (name, brand, ingredients, quantity)
  4. Combine: if LLM disagrees and gold matches LLM → improvement

Output:
  datasets/processed/llm_validator_text_attrs.parquet
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
logger = logging.getLogger(__name__)

WORKTREE_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = WORKTREE_ROOT / "datasets" / "processed"
OUT_PATH = PROCESSED_DIR / "llm_validator_text_attrs.parquet"

VALIDATOR_MODEL = "openai/gpt-oss-120b"
CONFIDENCE_THRESHOLD = 1.01  # validate ALL — cascade conf is tightly calibrated (>0.85 for everything)
MAX_LLM_CALLS_PER_RUN = 3500  # budget cap (~$4)
PARTNER_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]

# Text-derived attrs with real cascade error (others were 100% due to feature shortcuts)
TARGET_ATTRS = {
    "pasta": ["pasta_shape", "grain_type"],
    "chocolate": ["cocoa_percentage", "chocolate_type"],
}

COST_IN_PER_M = 0.04
COST_OUT_PER_M = 0.18


def build_partner_text(row: pd.Series) -> dict:
    return {k: str(row.get(k, "") or "") for k in PARTNER_FIELDS}


def build_prompt(partner: dict, cat: str, attr: str, allowed_values: list[str],
                 cascade_pred: str) -> str:
    name = partner.get("product_name", "")
    brand = partner.get("brands", "")
    ingredients = partner.get("ingredients_text", "")
    quantity = partner.get("quantity", "")

    questions = {
        ("pasta", "pasta_shape"): (
            "What is the pasta shape? Common values: spaghetti, penne, fusilli, "
            "tagliatelle, lasagne, macaroni, farfalle, ravioli, tortellini, "
            "linguine, rigatoni, conchiglie, gnocchi, noodles, other."
        ),
        ("pasta", "grain_type"): (
            "What is the grain type? Common values: wheat, durum_wheat, semolina, "
            "rice, corn, buckwheat, spelt, other."
        ),
        ("chocolate", "cocoa_percentage"): (
            "What is the cocoa percentage as a class? Values: low (<30%), "
            "medium (30-60%), high (>60%), unknown."
        ),
        ("chocolate", "chocolate_extra"): (
            "What extras does the chocolate contain? Values: nuts, fruits, "
            "caramel, mint, none, other."
        ),
        ("chocolate", "chocolate_type"): (
            "What type of chocolate? Values: dark, milk, white, other."
        ),
        ("chocolate", "contains_nuts"): (
            "Does this chocolate contain nuts (almonds, hazelnuts, peanuts, "
            "pistachios, walnuts, cashews)? Answer: yes / no."
        ),
        ("cheeses", "texture"): (
            "What is the cheese texture? Values: hard, soft, semi-hard, "
            "semi-soft, fresh, other."
        ),
        ("cheeses", "milk_source"): (
            "What is the milk source? Values: cow, goat, sheep, buffalo, "
            "mixed, other."
        ),
    }
    q = questions.get((cat, attr), f"Predict {attr} for this {cat} product.")

    allowed_str = ", ".join(allowed_values) if allowed_values else "see above"

    return f"""You are validating an automated classifier's prediction.

Product data (only what the partner sent — NOT OFF database):
- Product name: {name}
- Brand: {brand}
- Ingredients: {ingredients}
- Quantity: {quantity}

Classifier predicted: {attr} = {cascade_pred}
Allowed values: {allowed_str}

{q}

Answer in JSON ONLY:
{{"answer": "<one of allowed values>", "confidence": "high"|"medium"|"low", "reason": "<1 sentence>"}}"""


def parse_response(text: str) -> tuple[str, str, str]:
    """Returns (answer, confidence, reason)."""
    if not text:
        return "", "", ""
    # Try JSON
    try:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            ans = str(obj.get("answer", "")).strip().lower()
            conf = str(obj.get("confidence", "")).strip().lower()
            reason = str(obj.get("reason", ""))[:200]
            return ans, conf, reason
    except Exception:  # noqa: BLE001
        pass
    return "", "", text[:100]


def call_openrouter(prompt: str, api_key: str, max_retries: int = 3) -> tuple[str, dict]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": VALIDATOR_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=60,
            )
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning") or ""
            return text, data.get("usage", {})
        except Exception as exc:  # noqa: BLE001
            logger.warning("attempt %d: %s", attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    return "", {}


def norm_str(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip().lower()
    return s


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        env_path = WORKTREE_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return

    # Load gold + preds
    gold = pd.read_parquet(PROCESSED_DIR / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold["gold_is_null"]].copy()
    gold["code"] = gold["code"].astype(str)

    # Load partner data
    code_to_row: dict[str, dict] = {}
    for cat in TARGET_ATTRS:
        p = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["code"] = df["code"].astype(str)
            for _, row in df.iterrows():
                code_to_row[row["code"]] = build_partner_text(row)
    logger.info("Partner data loaded for %d codes", len(code_to_row))

    # Per-attr: get allowed values from gold distribution
    rows = []
    for cat, attrs in TARGET_ATTRS.items():
        preds = pd.read_parquet(PROCESSED_DIR / f"cascade_preds_{cat}_v2_gold.parquet")
        preds["code"] = preds["code"].astype(str)
        for attr in attrs:
            g = gold[(gold["category"] == cat) & (gold["attr"] == attr)]
            p = preds[preds["attr"] == attr]
            m = g.merge(p[["code", "predicted", "confidence"]], on="code", how="inner")
            m["cat"] = cat
            m["attribute"] = attr
            m["gold_norm"] = m["gold_value"].apply(norm_str)
            m["cascade_norm"] = m["predicted"].apply(norm_str)
            m["allowed_values"] = [
                sorted(set(g["gold_value"].dropna().astype(str).str.lower().tolist()))
            ] * len(m)
            rows.append(m)
    full = pd.concat(rows, ignore_index=True)
    full = full[full["code"].isin(code_to_row)].copy()

    # Cascade-only baseline accuracy
    full["cascade_correct"] = (full["gold_norm"] == full["cascade_norm"])
    n_total = len(full)
    n_correct = full["cascade_correct"].sum()
    logger.info("Cascade baseline: %d / %d = %.2f%%",
                n_correct, n_total, 100 * n_correct / n_total)

    # Per-attr cascade baseline
    print(f"\n{'='*60}")
    print(f"Cascade baseline per (cat, attr):")
    for (cat, attr), grp in full.groupby(["cat", "attribute"]):
        acc = grp["cascade_correct"].mean()
        print(f"  {cat}/{attr:<25} n={len(grp):>4} acc={acc*100:>5.1f}%")

    # Filter candidates for LLM validation
    candidates = full[full["confidence"] < CONFIDENCE_THRESHOLD].copy()
    logger.info("\nCandidates (conf<%.2f): %d", CONFIDENCE_THRESHOLD, len(candidates))
    if len(candidates) > MAX_LLM_CALLS_PER_RUN:
        candidates = candidates.sample(MAX_LLM_CALLS_PER_RUN, random_state=42)
        logger.info("Capped to %d", MAX_LLM_CALLS_PER_RUN)

    # Run validator with thread pool (gpt-oss-120b is slow due to reasoning)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    def worker(row_dict: dict) -> dict | None:
        partner = code_to_row[row_dict["code"]]
        prompt = build_prompt(
            partner, row_dict["cat"], row_dict["attribute"],
            row_dict["allowed_values"], row_dict["cascade_norm"],
        )
        text, usage = call_openrouter(prompt, api_key)
        if not text:
            return None
        cost = (usage.get("prompt_tokens", 0) * COST_IN_PER_M +
                usage.get("completion_tokens", 0) * COST_OUT_PER_M) / 1e6
        ans, conf, reason = parse_response(text)
        return {
            "code": row_dict["code"], "category": row_dict["cat"],
            "attribute": row_dict["attribute"],
            "gold_norm": row_dict["gold_norm"],
            "cascade_norm": row_dict["cascade_norm"],
            "cascade_conf": row_dict["confidence"],
            "llm_ans": ans, "llm_conf": conf, "llm_reason": reason,
            "cost_usd": cost,
        }

    results: list[dict] = []
    total_cost = 0.0
    n_done = 0
    n_lock = threading.Lock()
    candidate_dicts = [
        {**row.to_dict()} for _, row in candidates.iterrows()
    ]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(worker, rd): i for i, rd in enumerate(candidate_dicts)}
        for fut in as_completed(futures):
            res = fut.result()
            with n_lock:
                n_done += 1
                if res:
                    results.append(res)
                    total_cost += res["cost_usd"]
                if n_done % 100 == 0:
                    logger.info("%d/%d  $%.3f spent", n_done, len(candidates), total_cost)
                    # Incremental save
                    pd.DataFrame(results).to_parquet(OUT_PATH, index=False)

    out = pd.DataFrame(results)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %d validator results, $%.3f total", len(out), total_cost)

    if len(out) == 0:
        return

    # Combine: if LLM answer in allowed values and differs from cascade → use LLM
    out["combined"] = out["cascade_norm"]
    # If LLM is high-conf and gives non-empty answer → use LLM
    use_llm = (
        out["llm_ans"].notna() & (out["llm_ans"] != "")
        & (out["llm_ans"] != out["cascade_norm"])
        & out["llm_conf"].isin(["high", "medium"])
    )
    out.loc[use_llm, "combined"] = out.loc[use_llm, "llm_ans"]

    # Score each: 1 if matches gold, 0 otherwise
    out["cascade_hit"] = (out["cascade_norm"] == out["gold_norm"])
    out["llm_hit"] = (out["llm_ans"] == out["gold_norm"])
    out["combined_hit"] = (out["combined"] == out["gold_norm"])

    print(f"\n{'='*72}")
    print(f"LLM validator on text-derived attrs (cascade conf < {CONFIDENCE_THRESHOLD})")
    print(f"{'='*72}")
    print(f"Validated: {n_done} predictions, ${total_cost:.3f} spent")
    print()
    print(f"{'cat':<10} {'attr':<22} {'n':>5} {'cascade':>8} {'+validator':>11} {'lift':>7}")
    print("-" * 70)
    for (cat, attr), grp in out.groupby(["category", "attribute"]):
        casc = grp["cascade_hit"].mean()
        comb = grp["combined_hit"].mean()
        lift = (comb - casc) * 100
        print(f"{cat:<10} {attr:<22} {len(grp):>5} "
              f"{casc*100:>7.2f}% {comb*100:>10.2f}% {lift:>+6.2f}pp")
    print("-" * 70)
    casc_all = out["cascade_hit"].mean()
    comb_all = out["combined_hit"].mean()
    print(f"{'GRAND':<10} {'(validated subset)':<22} {len(out):>5} "
          f"{casc_all*100:>7.2f}% {comb_all*100:>10.2f}% {(comb_all-casc_all)*100:>+6.2f}pp")
    print()
    # Disagreement analysis
    n_llm_unknown = (out["llm_ans"] == "").sum()
    n_llm_agree = ((out["llm_ans"] == out["cascade_norm"]) & (out["llm_ans"] != "")).sum()
    n_llm_flip = ((out["llm_ans"] != out["cascade_norm"]) & (out["llm_ans"] != "")).sum()
    n_flip_correct = ((out["combined"] != out["cascade_norm"]) & out["combined_hit"]).sum()
    n_flip_wrong = ((out["combined"] != out["cascade_norm"]) & ~out["combined_hit"]).sum()
    print(f"Disagreement analysis:")
    print(f"  LLM returned unknown/unparseable: {n_llm_unknown}")
    print(f"  LLM agreed with cascade: {n_llm_agree}")
    print(f"  LLM disagreed with cascade: {n_llm_flip}")
    print(f"    → flip improved accuracy: {n_flip_correct}")
    print(f"    → flip hurt accuracy: {n_flip_wrong}")


if __name__ == "__main__":
    main()
