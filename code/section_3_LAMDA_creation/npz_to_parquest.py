import os
import numpy as np
import pandas as pd
import logging
from scipy.sparse import load_npz
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Paths ===
#input_path = '/home/shared-datasets/Feature_extraction/npz_yearwise_Final_rerun_0.01/'
#input_path = '/home/shared-datasets/Feature_extraction/npz_yearwise_Final_rerun_0.001/'
input_path = '/home/shared-datasets/Feature_extraction/npz_yearwise_Final_rerun_0.0001/'

#output_path = 'parquet_yearwise_Final_0.01'
#output_path = 'parquet_yearwise_Final_0.001'
output_path = 'parquet_yearwise_Final_0.0001'
log_path = os.path.join(output_path, 'conversion.log')
os.makedirs(output_path, exist_ok=True)

# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler()
    ]
)

# === Year list ===
years = ['2013', '2014', '2016', '2017', '2018', '2019',
         '2020', '2021', '2022', '2023', '2024', '2025']

# === Load vocabulary and create feature mapping ===
vocab_selected = np.load(os.path.join(input_path, "vocabulary_selected.npy"))
feature_names = [f"feat_{i}" for i in range(len(vocab_selected))]

# Save mapping
mapping_df = pd.DataFrame({
    "feature_name": vocab_selected,
    "mapped_name": feature_names
})
mapping_file = os.path.join(output_path, "feature_mapping.csv")
mapping_df.to_csv(mapping_file, index=False)
logging.info(f" Saved feature mapping to {mapping_file}")

# === Function to convert a single year ===
def process_year(year):
    try:
        logging.info(f" Starting {year}")

        # Load metadata
        meta_train = np.load(os.path.join(input_path, f"{year}_meta_train.npz"), allow_pickle=True)
        meta_test = np.load(os.path.join(input_path, f"{year}_meta_test.npz"), allow_pickle=True)

        df_train = pd.DataFrame({
            "label": meta_train["y"],
            "family": meta_train["family"],
            "vt_count": meta_train["vt_count"],
            "year_month": meta_train["year_month"],
            "hash": meta_train["hash"],
        })

        df_test = pd.DataFrame({
            "label": meta_test["y"],
            "family": meta_test["family"],
            "vt_count": meta_test["vt_count"],
            "year_month": meta_test["year_month"],
            "hash": meta_test["hash"],
        })

        # Load features and convert to dense
        X_train = load_npz(os.path.join(input_path, f"{year}_X_train.npz")).toarray()
        X_test = load_npz(os.path.join(input_path, f"{year}_X_test.npz")).toarray()

        # Assign feature column names
        df_train_features = pd.DataFrame(X_train, columns=feature_names, dtype=np.int8)
        df_test_features = pd.DataFrame(X_test, columns=feature_names, dtype=np.int8)

        # Combine metadata and features
        df_train_full = pd.concat([df_train.reset_index(drop=True), df_train_features], axis=1)
        df_test_full = pd.concat([df_test.reset_index(drop=True), df_test_features], axis=1)

        # Move 'hash' to the first column
        def move_hash_first(df):
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index('hash')))
            return df[cols]

        df_train_full = move_hash_first(df_train_full)
        df_test_full = move_hash_first(df_test_full)

        # Save to Parquet
        train_out = os.path.join(output_path, f"{year}_train.parquet")
        test_out = os.path.join(output_path, f"{year}_test.parquet")

        df_train_full.to_parquet(train_out, index=False)
        df_test_full.to_parquet(test_out, index=False)

        logging.info(
            f"{year} saved: {train_out}, {test_out} | "
            f"train shape: {df_train_full.shape}, test shape: {df_test_full.shape}"
        )
        return True

    except Exception as e:
        logging.error(f" Error processing {year}: {e}")
        return False

# === Run in parallel using 30 threads ===
max_workers = 30
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_year, year): year for year in years}
    for future in as_completed(futures):
        year = futures[future]
        try:
            future.result()
        except Exception as exc:
            logging.error(f" {year} failed unexpectedly: {exc}")

logging.info(" All years processed.")
 
