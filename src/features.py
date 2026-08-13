import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import math


def customer_id(df):
    """Assign unique customer_id based on customer attributes."""
    df = df.copy()
    df["customer_id"] = df.groupby(["first", "last", "zip", "dob", "cc_num"]).ngroup()
    return df

def customer_age(df):
    df = df.copy()
    df["age"] = (pd.Timestamp.now().year) - pd.to_datetime(df["dob"]).dt.year
    return df

def high_value_transaction(df):
    df = df.copy()
    df["percentile_95"] = np.percentile(df["amt"], 95)
    return df

def night_transaction(df):
    df = df.copy()
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['night_trans'] = (df['trans_date_trans_time'].dt.hour >= 22) | (df['trans_date_trans_time'].dt.hour < 6)
    return df


def amount_log_transformed(df):
    """Normalizes skewed amount distribution"""
    df = df.copy()
    df["amt_log"] = np.log1p(df["amt"])
    return df

def amount_normalized(df):
    """Standardizes transaction amount using StandardScaler"""

    df = df.copy()
    scaler = StandardScaler()

    df['amount_normalized'] = scaler.fit_transform(df[['amt']])
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in miles."""
    df = df.copy()
    R = 3959  # Earth's radius in miles
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def calc_distance(df):
    """Calculate distance between cardholder and merchant."""
    df = df.copy()
    df["distance"] = np.vectorize(haversine_distance)(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    return df


def transaction_amt_percentile(df):
    """
    Calculate the percentile of the current transaction amount
    relative to the cardholder's PRIOR transactions.
    """
    df = df.copy()

    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    def historical_percentile(amounts):
        values = amounts.to_numpy()
        result = np.full(len(values), 0.5, dtype=float)

        for i in range(1, len(values)):
            history = values[:i]
            result[i] = np.mean(history <= values[i])

        return pd.Series(result, index=amounts.index)

    df["amt_percentile"] = df.groupby("cc_num", group_keys=False)["amt"].apply(
        historical_percentile
    )

    return df


def transaction_velocity_features(df):
    """
    Create historical transaction velocity features for each cardholder.

    Features:
        - time_since_last_transaction: minutes since previous transaction
        - transactions_last_1h: prior transactions within the last hour
        - transactions_last_6h: prior transactions within the last 6 hours
        - transactions_last_24h: prior transactions within the last 24 hours

    Only transactions occurring before the current transaction are used.
    """
    df = df.copy()

    # make sure timestamp is datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # sort transactions chronologically for each card
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    # time since previous transaction

    df["previous_transaction_time"] = df.groupby("cc_num")[
        "trans_date_trans_time"
    ].shift(1)

    df["time_since_last_transaction"] = (
        df["trans_date_trans_time"] - df["previous_transaction_time"]
    ).dt.total_seconds() / 60

    # First transaction has no previous transaction
    df["is_first_transaction"] = (
    df["time_since_last_transaction"].isna().astype(int)
)

    df["time_since_last_transaction"] = (
        df["time_since_last_transaction"].fillna(0)
)
    df["log_time_since_last_transaction"] = np.log1p(df["time_since_last_transaction"])

    # count previous transactions in rolling windows

    indexed = df.set_index("trans_date_trans_time")

    df["transactions_last_1h"] = (
        indexed.groupby("cc_num")["amt"]
        .rolling("1h", closed="left")
        .count()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    df["transactions_last_6h"] = (
        indexed.groupby("cc_num")["amt"]
        .rolling("6h", closed="left")
        .count()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    df["transactions_last_24h"] = (
        indexed.groupby("cc_num")["amt"]
        .rolling("24h", closed="left")
        .count()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    # rolling counts can be nan when there is no history
    velocity_columns = [
        "transactions_last_1h",
        "transactions_last_6h",
        "transactions_last_24h",
    ]

    df[velocity_columns] = df[velocity_columns].fillna(0)

    df = df.drop(columns=["previous_transaction_time"])

    return df


def unusual_amt(df):
    """Calculate z-score for transaction amount relative to cardholder history."""
    df = df.copy()
    
    df = df.sort_values(["cc_num", "trans_date_trans_time"])

    grouped = df.groupby("cc_num")["amt"]

    df["historical_mean_amt"] = grouped.transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df["historical_std_amt"] = grouped.transform(
        lambda x: x.shift(1).expanding().std()
    )

    df["z_score"] = (
        (df["amt"] - df["historical_mean_amt"])
        / df["historical_std_amt"]
    )

    df["z_score"] = df["z_score"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def transaction_hour(df):
    """Extract hour from transaction timestamp."""
    df = df.copy()
    
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    return df


def transaction_day(df):
    """Extract day of week from transaction timestamp."""
    df = df.copy()
    
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["day_string"] = df["trans_date_trans_time"].dt.day_name()
    return df


def is_weekend(df):
    """Flag weekend transactions."""
    df = df.copy()
    
    df["is_weekend"] = df["day_string"].isin(["Saturday", "Sunday"]).astype(int)
    return df


def transaction_month(df):
    """Extract month name from transaction timestamp."""
    df = df.copy()
    
    df["transaction_month"] = df["trans_date_trans_time"].dt.month_name()
    return df


def highest_fraud_states(df):
    """
    Calculate the fraud rate for each state and create a feature that flags
    transactions from high-fraud states.

    Args:
        df: DataFrame with columns ['state', 'is_fraud']

    Returns:
        df: DataFrame with new columns 'state_fraud_rate' and 'high_fraud_state'
        state_fraud_stats: Stats by state
        high_fraud_states_list: List of high-fraud states
    """
    df = df.copy()
    

    # calculate fraud rate by state
    state_fraud_stats = (
        df.groupby("state")
        .agg(
            total_transactions=("is_fraud", "count"),
            fraud_transactions=("is_fraud", "sum"),
        )
        .reset_index()
    )

    # calculate the fraud rate
    state_fraud_stats["state_fraud_rate"] = (
        state_fraud_stats["fraud_transactions"]
        / state_fraud_stats["total_transactions"]
    )

    # Step 3: Sort by fraud rate (highest to lowest)
    state_fraud_stats = state_fraud_stats.sort_values(
        "state_fraud_rate", ascending=False
    )

    # identify high-fraud states (top 25%)
    high_fraud_threshold = state_fraud_stats["state_fraud_rate"].quantile(0.75)
    high_fraud_states_list = state_fraud_stats[
        state_fraud_stats["state_fraud_rate"] >= high_fraud_threshold
    ]["state"].tolist()

    # merge fraud rate back to original dataframe
    df = df.merge(
        state_fraud_stats[["state", "state_fraud_rate"]], on="state", how="left"
    )

    # Step 6: Create binary flag for high-fraud states
    df["high_fraud_state"] = df["state"].isin(high_fraud_states_list).astype(int)

    return df, state_fraud_stats, high_fraud_states_list


def engineer_features(df):
    """
    Apply all feature engineering functions to the dataframe.

    Args:
        df: Raw transactions dataframe

    Returns:
        df: DataFrame with all engineered features
    """
    df = df.copy()

    # Convert datetime and date columns first
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df = customer_id(df)

    # Apply all feature engineering functions
    df = calc_distance(df)
    df = transaction_amt_percentile(df)
    df = transaction_velocity_features(df)
    df = unusual_amt(df)
    df = transaction_hour(df)
    df = transaction_day(df)
    df = is_weekend(df)
    df = transaction_month(df)

    return df
