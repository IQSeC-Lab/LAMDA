import os
import pandas as pd

# Base path
BASE_PATH = "/home/mkamol/family_info/LAMDA/Baseline" # download this from https://huggingface.co/datasets/IQSeC-Lab/LAMDA/tree/main/Baseline
SAVE_DIR = "/home/mkamol/LAMDA/continual-learning"

# Load all data
def load_all_data():
    data = {}
    for year in sorted(os.listdir(BASE_PATH)):
        year_path = os.path.join(BASE_PATH, year)
        if os.path.isdir(year_path):
            try:
                train_path = os.path.join(year_path, f"{year}_train.parquet")
                test_path = os.path.join(year_path, f"{year}_test.parquet")
                
                # Load data
                train_df = pd.read_parquet(train_path)
                test_df = pd.read_parquet(test_path)

                # Filter: label==1 and family != 'unknown'
                train_df = train_df[
                    (train_df['label'] == 1) &
                    (~train_df['family'].str.lower().eq('unknown'))
                ]
                test_df = test_df[
                    (test_df['label'] == 1) &
                    (~test_df['family'].str.lower().eq('unknown'))
                ]

                data[year] = {"train": train_df, "test": test_df}
            except Exception as e:
                print(f"Error loading {year}: {e}")
    return data

# Task 5: Process families
def process_families(data):
    print("[Task 5] Processing families with >10 samples in test and existing in train...\n")
    os.makedirs(SAVE_DIR, exist_ok=True)

    overall_families = set()

    for year, parts in data.items():
        train_df = parts["train"]
        test_df = parts["test"]

        train_counts = train_df['family'].value_counts()
        test_counts = test_df['family'].value_counts()

        common_families = test_counts[test_counts > 10].index.intersection(train_counts.index)

        print(f"Year: {year}")
        print(f"Train families: {len(train_counts)}")
        print(f"Test families: {len(test_counts)}")
        print(f"Common (filtered) families: {len(common_families)}\n")

        if not common_families.empty:
            filtered_train = train_df[train_df['family'].isin(common_families)].copy()
            filtered_test = test_df[test_df['family'].isin(common_families)].copy()

            filtered_train.to_parquet(os.path.join(SAVE_DIR, f"{year}_train_filtered.parquet"))
            filtered_test.to_parquet(os.path.join(SAVE_DIR, f"{year}_test_filtered.parquet"))

            overall_families.update(common_families)
        else:
            print("No eligible families this year.\n")

    print(f"\n[Overall] Total unique common families: {len(overall_families)}")

# Run directly
if __name__ == "__main__":
    data = load_all_data()
    process_families(data)
