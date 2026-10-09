import pandas as pd
import numpy as np
import os

input_file = "data.csv"
test_size = 0.2
random_state = 42

COLUMNS = (
    ["id", "diagnosis"] +
    [f"feature_{i}" for i in range(1, 31)]
)

def split_data(input_file: str, output_dir: str, test_size: float = 0.2,
               random_state: int = 42):

    df = pd.read_csv(input_file, header=None, names=COLUMNS)
    df = df.drop(columns=["id"])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Seed: {random_state}")
    print(f"Обучающая выборка: {train_path}")
    print(f"Тестовая выборка: {test_path}")
    print(f"Соотношение: {1-test_size:.0%} / {test_size:.0%}")
    print(f"M(1): {(df['diagnosis']==1).sum()} | B(0): {(df['diagnosis']==-1).sum()}")


if __name__ == "__main__":
    split_data(input_file, ".", test_size, random_state)  