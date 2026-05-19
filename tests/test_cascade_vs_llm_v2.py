import pandas as pd
from src.eval.cascade_vs_llm_v2 import paired_compare


def test_refusal_counts_as_miss_in_accuracy():
    cascade = pd.DataFrame([
        {"code": "1", "attr": "x", "predicted": "A"},
        {"code": "2", "attr": "x", "predicted": "A"},
    ])
    llm = pd.DataFrame([
        {"code": "1", "attr": "x", "predicted": "A"},
        {"code": "2", "attr": "x", "predicted": None},
    ])
    gold = pd.DataFrame([
        {"code": "1", "attr": "x", "gold_value": "A", "gold_is_null": False},
        {"code": "2", "attr": "x", "gold_value": "A", "gold_is_null": False},
    ])
    out = paired_compare(cascade, llm, gold, model_name="M")
    row = out.iloc[0]
    assert row["cascade_acc"] == 1.0
    assert row["llm_acc_refusal_as_miss"] == 0.5
