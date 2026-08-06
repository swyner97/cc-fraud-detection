import pandas as pd
import numpy as np
from datetime import datetime
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance in miles between two lat/long coordinates
    """
    R = 3959  # Earth's radius in miles

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def transaction_amt_percentile(df):
    """
    Optimized percentile calculation using vectorization.
    Calculates the percentile of the current transaction amount relative to
    all PREVIOUS transactions by the same cardholder.

    Args:
        df: DataFrame with columns ['cc_num', 'amt', 'trans_date_trans_time']
            Must be pre-sorted chronologically.
    Returns:
        Series of percentile values (0-1)
    """
    # count how many past transactions are <= current transaction (inclusive rank)
    # pct=False returns raw ranks (1, 2, 3...) instead of percentages
    inclusive_rank = (
        df.groupby("cc_num")["amt"]
        .expanding()
        .rank(pct=False)
        .reset_index(level=0, drop=True)
    )

    # track the total historical count for each transaction step
    total_past_count = df.groupby("cc_num").cumcount()

    # handle the mathematical shift to exclude the current transaction
    # Formula: (inclusive_rank - 1) / total_past_count
    # Using np.where avoids division by zero on a user's very first transaction
    percentiles = np.where(
        total_past_count > 0,
        (inclusive_rank - 1) / total_past_count,
        0.5,  # First transaction defaults to 0.5 (middle)
    )

    return pd.Series(percentiles, index=df.index)


def calculate_historical_transaction_rates(df):
    """
    Calculates the transaction rate (transactions per hour) for each customer 
    prior to their most recent transaction.

    Parameters:
    df (pd.DataFrame): Input transactions DataFrame containing customer profile info,
                       'customer_id', 'trans_num', and 'trans_date_trans_time'.

    Returns:
    pd.DataFrame: A customer-level DataFrame with the calculated rate column.
    """

    # identify the timestamp of the most recent transaction for each customer
    df['max_trans_time'] = df.groupby('customer_id')['trans_date_trans_time'].transform('max')

    # filter to keep ONLY transactions BEFORE that final timestamp
    historical_df = df[df['trans_date_trans_time'] < df['max_trans_time']].copy()

    # aggregate historical data metrics
    groupby_cols = ["customer_id", "first", "last", "gender", "zip", "dob"]
    customer_rates = historical_df.groupby(groupby_cols).agg(
        historical_trans_count=('trans_num', 'count'),
        first_trans_time=('trans_date_trans_time', 'min'),
        pre_max_trans_time=('trans_date_trans_time', 'max')
    ).reset_index()

    # calculate active time window in hours
    time_delta = customer_rates['pre_max_trans_time'] - customer_rates['first_trans_time']
    customer_rates['hours_active'] = time_delta.dt.total_seconds() / 3600

    # compute transactions per hour safely (handle division by zero)
    customer_rates['trans_per_hour_before_latest'] = np.where(
        customer_rates['hours_active'] > 0,
        customer_rates['historical_trans_count'] / customer_rates['hours_active'],
        0.0  # Default to 0 if all historical transactions happened at the exact same second
    )

    # drop intermediate calculation columns for a clean feature set
    feature_cols = groupby_cols + ['trans_per_hour_before_latest']
    customer_features_df = customer_rates[feature_cols].copy()

    return customer_features_df

def unusual_amt(df):
    df["mean"] = df.groupby("cardholder_id")["amount"].transform("mean")
    df["std"] = df.groupby("cardholder_id")["amount"].transform("std")

    # Calculate Z-score (handle zero standard deviation if all amounts are identical)
    df["z_score"] = (df["amount"] - df["mean"]) / df["std"]
    df["z_score"] = df["z_score"].fillna(0)

    return df
