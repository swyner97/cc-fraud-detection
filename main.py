import pandas as pd

from src.clean_data import date
from src.transformations import (
    customer_id,
)


def main():
    df = pd.read_csv("data/fraudTrain.csv")
    df = date(df)

    df = customer_id(df)

    df.to_csv("data/processed/cleaned_transactions.csv", index=False)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
