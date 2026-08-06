import pandas as pd
import numpy as np

def date(df): 
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S"
    )
    return df
