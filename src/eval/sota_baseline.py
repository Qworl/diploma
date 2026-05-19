"""
SOTA-baseline reproduction (OpenTag-style) for product attribute extraction
on pasta domain. Used in §6.11 as a learned baseline complementing
direct Claude Haiku 4.5.

Architecture: BiLSTM(embed=128, hidden=256) → mean-pool → linear,
sequence-level classification of the target attribute.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

_TOK_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)


def tokenize_for_opentag(text: str) -> list[str]:
    return [t.lower() for t in _TOK_RE.findall(text or "")]


class _PasteSeqDataset(Dataset):
    def __init__(self, df, vocab, label2id, target_col, max_len=24):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.label2id = label2id
        self.target_col = target_col
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        toks = tokenize_for_opentag(f"{row['product_name']} {row.get('brands','')}")[:self.max_len]
        ids = [self.vocab.get(t, 0) for t in toks]
        pad = self.max_len - len(ids)
        ids = ids + [0] * pad
        mask = [1] * (self.max_len - pad) + [0] * pad
        label = self.label2id.get(str(row[self.target_col]), 0)
        return torch.tensor(ids), torch.tensor(mask), torch.tensor(label)


class _BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, embed=128, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.lstm = nn.LSTM(embed, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x, mask):
        e = self.emb(x)
        out, _ = self.lstm(e)
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.fc(pooled)


def train_eval_opentag(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    max_epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    df = df.dropna(subset=[target_col]).copy()
    if len(df) < 30:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n": len(df)}

    all_toks = set()
    for _, row in df.iterrows():
        all_toks.update(tokenize_for_opentag(f"{row['product_name']} {row.get('brands','')}"))
    vocab = {tok: i + 1 for i, tok in enumerate(sorted(all_toks))}
    labels = sorted(df[target_col].astype(str).unique())
    label2id = {l: i for i, l in enumerate(labels)}

    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(df, test_size=test_size, random_state=seed,
                                stratify=df[target_col])

    train_ds = _PasteSeqDataset(tr, vocab, label2id, target_col)
    test_ds = _PasteSeqDataset(te, vocab, label2id, target_col)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    torch.manual_seed(seed)
    model = _BiLSTMClassifier(len(vocab) + 1, len(labels))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        model.train()
        for x, m, y in train_dl:
            opt.zero_grad()
            out = model(x, m)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, m, y in test_dl:
            out = model(x, m)
            preds.extend(out.argmax(dim=1).numpy().tolist())
            gts.extend(y.numpy().tolist())

    return {
        "accuracy": float(accuracy_score(gts, preds)),
        "macro_f1": float(f1_score(gts, preds, average="macro", zero_division=0)),
        "n": len(te),
        "n_classes": len(labels),
    }


def main():
    import os
    from src.common import PROCESSED_DIR, setup_logging

    setup_logging()
    df = pd.read_parquet(f"{PROCESSED_DIR}/pasta_stratified_silver_standard.parquet")
    rows = []
    for attr in ["grain_type", "pasta_shape", "is_organic", "is_filled"]:
        if attr not in df.columns:
            continue
        res = train_eval_opentag(df, attr)
        rows.append({"attr": attr, **res})
        logger.info("%s: acc=%.3f, macro_f1=%.3f, n=%d",
                     attr, res["accuracy"], res["macro_f1"], res.get("n", 0))
    pd.DataFrame(rows).to_parquet(f"{PROCESSED_DIR}/sota_baseline_pasta.parquet", index=False)
    logger.info("Saved sota_baseline_pasta.parquet")


if __name__ == "__main__":
    main()
