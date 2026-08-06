import pandas as pd
import numpy as np

def customer_id(df):
    df["customer_id"] = df.groupby(["first", "last", "zip", "dob", "cc_num"]).ngroup()
    return df
