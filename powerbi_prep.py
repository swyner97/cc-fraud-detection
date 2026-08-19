"""
Prep cleaned/engineered fraud data for Power BI.
Run from repo root: python prepare_powerbi.py
Outputs go to data/processed/
"""

import pandas as pd
import numpy as np
from src.features import engineer_features

RAW_PATH = "data/fraudTrain.csv"
OUT_DIR = "data/processed"

# Columns to drop before export — PII, raw geo, or leakage-risk fields
DROP_COLS = [
    "cc_num",
    "first",
    "last",
    "street",
    "city",
    "zip",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "unix_time",
    "trans_num",
    "merchant",
    "historical_mean_amt",
    "historical_std_amt",  # noisy for viz
]

AGE_BINS = [18, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

AMT_BINS = [0, 50, 100, 500, 1000, 5000, 30000]
AMT_LABELS = ["$0-$50", "$50-$100", "$100-$500", "$500-$1K", "$1K-$5K", "$5K+"]


def build_transactions_table(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df)

    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    df["amt_bracket"] = pd.cut(df["amt"], bins=AMT_BINS, labels=AMT_LABELS)
    df["location_type"] = pd.cut(
        df["city_pop"],
        bins=[0, 25000, 100000, float("inf")],
        labels=["Rural", "Suburban", "Urban"],
    )

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def rate_table(col):
        g = df.groupby(col, observed=False)["is_fraud"].agg(
            total_transactions="count", fraud_count="sum"
        )
        g["fraud_rate"] = g["fraud_count"] / g["total_transactions"]
        return g.reset_index()

    return {
        "by_category": rate_table("category"),
        "by_state": rate_table("state"),
        "by_hour": rate_table("hour"),
        "by_age_group": rate_table("age_group"),
        "by_amt_bracket": rate_table("amt_bracket"),
        "by_gender": rate_table("gender"),
        "by_location_type": rate_table("location_type"),
    }


def main():
    import os

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    txns = build_transactions_table(df)

    txns.to_csv(f"{OUT_DIR}/powerbi_transactions.csv", index=False)
    print(
        f"Wrote {OUT_DIR}/powerbi_transactions.csv ({len(txns):,} rows, {txns.shape[1]} cols)"
    )

    for name, table in build_summary_tables(txns).items():
        path = f"{OUT_DIR}/summary_{name}.csv"
        table.to_csv(path, index=False)
        print(f"Wrote {path} ({len(table)} rows)")


if __name__ == "__main__":
    main()
