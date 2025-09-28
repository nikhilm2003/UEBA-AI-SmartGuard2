# src/utils.py

import pandas as pd

def describe_dataset(file_path: str):
    """
    Print basic statistics of dataset.
    """
    df = pd.read_csv(file_path)
    print("Dataset preview:")
    print(df.head())
    print("\nStatistics:")
    print(df.describe())
