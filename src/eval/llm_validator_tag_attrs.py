"""LLM validator for tag-derived attributes (partner-input only).

Tag-derived attrs (is_organic, is_vegan, is_gluten_free, is_pdo, contains_nuts,
is_ultra_processed, country_of_origin) have circularity in eval: silver gold
comes from OFF tags, train cascade learns to predict OFF tags, test gold also
from OFF tags. In production, partner does NOT send tags — cascade has to infer
attrs from text alone, where it's weaker.

This validator:
  1. Takes cascade predictions on tag-derived attrs
  2. For low-confidence (< threshold) predictions, asks gpt-oss-120b
  3. LLM sees ONLY partner input (name, brand, ingredients, quantity) — NO OFF tags
  4. Combines: if LLM disagrees confidently → use LLM; else keep cascade

Measures the lift: cascade-alone accuracy vs cascade+validator accuracy on v2 gold.

Output:
  datasets/processed/llm_validator_tag_attrs.parquet
"""
from __future__ import annotations

import json
import logging
import os
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
MODELS_DIR = WORKTREE_ROOT / "models"
OUT_PATH = PROCESSED_DIR / "llm_validator_tag_attrs.parquet"

CATEGORIES = ["pasta", "chocolate", "cheeses"]
VALIDATOR_MODEL = "openai/gpt-oss-120b"
CONFIDENCE_THRESHOLD = 1.01  # validate ALL (cascade conf on tag-derived is ~0.99 due to majority-class calibration; real value is in checking high-conf preds where cascade just memorized silver tag distribution)
MAX_LLM_CALLS_PER_RUN = 1500  # budget cap — pick ~50% random subset
PARTNER_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]

# OpenRouter pricing for gpt-oss-120b: $0.04 in / $0.18 out per 1M tokens
COST_IN_PER_M = 0.04
COST_OUT_PER_M = 0.18


def get_tag_derived_attrs() -> dict[str, list[str]]:
    tax = pd.read_parquet(PROCESSED_DIR / "attribute_signal_taxonomy.parquet")
    tag = tax[tax["signal_type"] == "tag_derived"]
    out: dict[str, list[str]] = {}
    for cat in CATEGORIES:
        out[cat] = tag[tag["category"] == cat]["attr"].tolist()
    return out


def build_partner_text(row: pd.Series) -> dict:
    return {k: str(row.get(k, "") or "") for k in PARTNER_FIELDS}


def build_prompt(partner: dict, attr: str, predicted: str) -> str:
    name = partner.get("product_name", "")
    brand = partner.get("brands", "")
    ingredients = partner.get("ingredients_text", "")
    quantity = partner.get("quantity", "")

    question_map = {
        "is_organic": "Is this product certified organic? Look for signals like 'BIO', 'ECOCERT', 'organic certified', 'AB' (Agriculture Biologique), 'USDA organic' in name or ingredients. Answer: yes / no / unsure.",
        "is_vegan": "Is this product vegan? It must contain no animal-derived ingredients (no milk, eggs, honey, gelatin). Answer: yes / no / unsure.",
        "is_gluten_free": "Is this product gluten-free? Look for explicit 'gluten-free' or 'sans gluten' markers, OR ingredients containing only non-gluten grains (rice, corn, quinoa). Answer: yes / no / unsure.",
        "is_pdo": "Is this cheese PDO (Protected Designation of Origin)? Look for 'AOP', 'AOC', 'DOP', 'PDO' in name or brand. Answer: yes / no / unsure.",
        "contains_nuts": "Does this product contain nuts (almonds, hazelnuts, walnuts, peanuts, pistachios, cashews)? Check ingredients carefully. Answer: yes / no / unsure.",
        "is_ultra_processed": "Is this product ultra-processed (NOVA group 4)? Look for industrial ingredients like emulsifiers (E471, E472), flavor enhancers, hydrogenated oils, high-fructose corn syrup. Answer: yes / no / unsure.",
        "country_of_origin": "What is the country of origin / manufacturing country of this product? Look for country mentions in name, brand, or PDO markers (Italian, French, etc.). Answer with country name only, or 'unsure'.",
    }
    question = question_map.get(attr, f"Predict the value of '{attr}' for this product. Answer concisely.")

    return f"""You are validating an automated classifier's prediction.

Product data (only what the partner sent — NOT the OFF database):
- Product name: {name}
- Brand: {brand}
- Ingredients: {ingredients}
- Quantity: {quantity}

Classifier predicted: {attr} = {predicted}

{question}

Answer in JSON: {{"answer": "yes" | "no" | "unsure", "reason": "<one short sentence>"}}"""


def parse_response(text: str, attr: str) -> tuple[str, str]:
    """Returns (answer, reason). answer in {yes, no, unsure} or country name."""
    import re
    # Try JSON block
    try:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            ans = str(obj.get("answer", "unsure")).strip().lower()
            reason = str(obj.get("reason", ""))[:200]
            return ans, reason
    except Exception:  # noqa: BLE001
        pass
    # Fallback: look for yes/no/unsure
    low = text.lower()
    for w in ("yes", "no", "unsure"):
        if w in low[:30]:
            return w, text[:100]
    return "unsure", text[:100]


def call_openrouter(prompt: str, api_key: str, max_retries: int = 3) -> tuple[str, dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": VALIDATOR_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,  # gpt-oss-120b is reasoning-model: needs headroom for thinking + output
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=60,
            )
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning("429, sleeping %ds", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            # gpt-oss-120b is reasoning model: content may be null, actual answer in reasoning
            text = msg.get("content") or msg.get("reasoning") or ""
            usage = data.get("usage", {})
            return text, usage
        except Exception as exc:  # noqa: BLE001
            logger.warning("Attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    return "", {}


def gold_to_str(val) -> str:
    if isinstance(val, bool):
        return "yes" if val else "no"
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip().lower()
    if s in ("true", "yes", "1"):
        return "yes"
    if s in ("false", "no", "0"):
        return "no"
    return s


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Try .env
        env_path = WORKTREE_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return

    tag_attrs = get_tag_derived_attrs()
    logger.info("Tag-derived attrs: %s", tag_attrs)

    # Load gold
    gold = pd.read_parquet(PROCESSED_DIR / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold["gold_is_null"]].copy()
    gold["code"] = gold["code"].astype(str)

    # Load cascade predictions
    pred_dfs = []
    for cat in CATEGORIES:
        p = PROCESSED_DIR / f"cascade_preds_{cat}_v2_gold.parquet"
        if not p.exists():
            logger.warning("Missing cascade preds for %s", cat)
            continue
        df = pd.read_parquet(p)
        df["code"] = df["code"].astype(str)
        df["category"] = cat
        pred_dfs.append(df)
    preds = pd.concat(pred_dfs, ignore_index=True)

    # Load partner-input data
    code_to_row: dict[str, dict] = {}
    for cat in CATEGORIES:
        p = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["code"] = df["code"].astype(str)
        for _, row in df.iterrows():
            code_to_row[row["code"]] = build_partner_text(row)

    # Join gold + preds — filter to tag-derived only
    rows_to_validate = []
    for cat, attrs in tag_attrs.items():
        for attr in attrs:
            gold_sub = gold[(gold["category"] == cat) & (gold["attr"] == attr)].copy()
            pred_sub = preds[(preds["category"] == cat) & (preds["attr"] == attr)].copy()
            merged = gold_sub.merge(
                pred_sub[["code", "predicted", "confidence"]],
                on="code", how="inner",
            )
            merged["cat"] = cat
            merged["attribute"] = attr
            rows_to_validate.append(merged)

    if not rows_to_validate:
        logger.error("No rows to validate")
        return
    full = pd.concat(rows_to_validate, ignore_index=True)
    full["gold_str"] = full["gold_value"].apply(gold_to_str)
    full["cascade_str"] = full["predicted"].apply(gold_to_str)
    logger.info("Total rows: %d. Cascade correct: %d (%.2f%%)",
                len(full),
                (full["gold_str"] == full["cascade_str"]).sum(),
                100 * (full["gold_str"] == full["cascade_str"]).mean())

    # Filter to low-confidence
    candidates = full[full["confidence"] < CONFIDENCE_THRESHOLD].copy()
    logger.info("Candidates for validation (conf < %.2f): %d",
                CONFIDENCE_THRESHOLD, len(candidates))
    if len(candidates) > MAX_LLM_CALLS_PER_RUN:
        logger.warning("Capping to %d calls (budget)", MAX_LLM_CALLS_PER_RUN)
        candidates = candidates.sample(MAX_LLM_CALLS_PER_RUN, random_state=42)

    # Run validator
    results: list[dict] = []
    total_cost = 0.0
    n_done = 0
    for i, row in candidates.iterrows():
        code = row["code"]
        if code not in code_to_row:
            continue
        partner = code_to_row[code]
        prompt = build_prompt(partner, row["attribute"], row["cascade_str"])

        text, usage = call_openrouter(prompt, api_key)
        if not text:
            continue
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost = (in_tok * COST_IN_PER_M + out_tok * COST_OUT_PER_M) / 1e6
        total_cost += cost

        ans, reason = parse_response(text, row["attribute"])
        results.append({
            "code": code, "category": row["cat"], "attr": row["attribute"],
            "gold_str": row["gold_str"], "cascade_str": row["cascade_str"],
            "confidence": row["confidence"],
            "llm_ans": ans, "llm_reason": reason,
            "cost_usd": cost,
        })
        n_done += 1
        if n_done % 100 == 0:
            logger.info("%d/%d done, $%.3f spent", n_done, len(candidates), total_cost)

    out = pd.DataFrame(results)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %d validator results, $%.3f total", len(out), total_cost)

    # --- Combine cascade + validator: if LLM strongly disagrees, use LLM ---
    full = full.set_index(["code", "attribute"]).copy()
    out = out.set_index(["code", "attr"])
    full["llm_ans"] = out["llm_ans"]
    full = full.reset_index()
    full["combined"] = full["cascade_str"]

    # Override rule:
    # If LLM said yes/no and disagrees with cascade → use LLM
    # If LLM said unsure → keep cascade
    mask_yn = full["llm_ans"].isin(["yes", "no"])
    mask_disagree = full["llm_ans"] != full["cascade_str"]
    mask = mask_yn & mask_disagree
    full.loc[mask, "combined"] = full.loc[mask, "llm_ans"]

    # For country_of_origin: country names not yes/no
    co_mask = full["attribute"] == "country_of_origin"
    co_use_llm = co_mask & ~full["llm_ans"].isin(["unsure", ""]) & full["llm_ans"].notna()
    full.loc[co_use_llm, "combined"] = full.loc[co_use_llm, "llm_ans"]

    # --- Compare accuracies ---
    print("\n" + "="*70)
    print(f"LLM Validator on tag-derived attrs (cascade conf < {CONFIDENCE_THRESHOLD})")
    print("="*70)
    print(f"\nValidator model: {VALIDATOR_MODEL}, total cost: ${total_cost:.3f}")
    print(f"Calls made: {n_done}")
    print()
    print(f"{'cat':<10} {'attr':<22} {'n':>5} {'cascade':>8} {'+validator':>11} {'lift':>7}")
    print("-" * 70)
    for cat in CATEGORIES:
        for attr in tag_attrs[cat]:
            sub = full[(full["cat"] == cat) & (full["attribute"] == attr)]
            if len(sub) == 0:
                continue
            casc = (sub["gold_str"] == sub["cascade_str"]).mean()
            comb = (sub["gold_str"] == sub["combined"]).mean()
            lift = (comb - casc) * 100
            print(f"{cat:<10} {attr:<22} {len(sub):>5} {casc*100:>7.2f}% {comb*100:>10.2f}% {lift:>+6.2f}pp")

    full_casc = (full["gold_str"] == full["cascade_str"]).mean()
    full_comb = (full["gold_str"] == full["combined"]).mean()
    print("-" * 70)
    print(f"{'GRAND':<10} {'(all tag attrs)':<22} {len(full):>5} {full_casc*100:>7.2f}% {full_comb*100:>10.2f}% {(full_comb-full_casc)*100:>+6.2f}pp")


if __name__ == "__main__":
    main()
