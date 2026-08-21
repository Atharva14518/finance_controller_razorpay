"""Load source CSVs into typed DataFrames."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date


def load_orders(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "orders_ledger.csv")
    df["created_date"] = _parse_date(df["created_date"])
    df["amount"] = df["amount"].astype(int)
    return df


def load_settlement(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "settlement_report.csv")
    df["settled_date"] = _parse_date(df["settled_date"])
    for col in ("amount", "fee", "tax", "net_amount"):
        df[col] = df[col].astype(int)
    return df


def load_bank(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "bank_statement.csv")
    df["date"] = _parse_date(df["date"])
    df["amount"] = df["amount"].astype(int)
    return df


def load_labeled_pairs(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "labeled_pairs_for_training.csv")


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders_df = load_orders()
    settlement_df = load_settlement()
    bank_df = load_bank()
    assert len(orders_df) == 62, f"Expected 62 orders, got {len(orders_df)}"
    assert len(settlement_df) == 72, f"Expected 72 settlement rows, got {len(settlement_df)}"
    assert len(bank_df) == 14, f"Expected 14 bank rows, got {len(bank_df)}"
    return orders_df, settlement_df, bank_df
