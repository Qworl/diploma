import pandas as pd
from src.eval.llm_ceiling_check import build_table


def test_table_has_three_columns_per_model():
    cascade_acc = {"pasta": 0.80}
    llm_acc = pd.DataFrame([
        {"model": "M1", "context_mode": "partner_input", "category": "pasta", "accuracy": 0.65},
        {"model": "M1", "context_mode": "off_grounded",  "category": "pasta", "accuracy": 0.78},
    ])
    out = build_table(llm_acc, cascade_acc)
    row = out[out["model"] == "M1"].iloc[0]
    assert row["acc_partner_input"] == 0.65
    assert row["acc_off_grounded"] == 0.78
    assert row["acc_cascade"] == 0.80
    assert abs(row["ceiling_delta_pp"] - 13.0) < 1e-9
    assert abs(row["cascade_advantage_vs_partner_pp"] - 15.0) < 1e-9
