import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# === Load the CSV ===
csv_path = "/home/shared-datasets/Feature_extraction/all_hash_added_year_month_with_label_and_ym_and_family.csv"
df = pd.read_csv(csv_path)
df['sha256'] = df['sha256'].str.lower()
print("CSV loaded")

# Change the path to your extracted .data file and folder organization is
# year___ malware
#   |____ benign

data_path = '/home/shared-datasets/Feature_extraction/data_file_by_year'
output_path = 'Final_AZ_Data'
os.makedirs(output_path, exist_ok=True)

def get_features(file_list):
    features = []
    for f in file_list:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read().replace('\n', ',').strip(',')
                features.append(content.split(','))
        except FileNotFoundError:
            print(f"Missing file: {f}")
            continue
    return np.array(features, dtype=object)

def extract_save_features(df, data_path, test_size=0.2):
    df['sha256'] = df['sha256'].str.lower()
    all_years = sorted([y for y in os.listdir(data_path) if y.isdigit() and y != "2015"])

    for year in all_years:
        year_dir = os.path.join(data_path, year)
        malware_dir = os.path.join(year_dir, "malware")
        benign_dir = os.path.join(year_dir, "benign")

        hash_file_paths = {}
        for label_name, label_val in [("malware", 1), ("benign", 0)]:
            folder = os.path.join(year_dir, label_name)
            for fname in os.listdir(folder):
                if fname.endswith(".data"):
                    h = fname.replace(".data", "").lower()
                    hash_file_paths[h] = os.path.join(folder, fname)

        # Filter CSV to this year
        year_df = df[df['year'] == int(year)].copy()
        if len(year_df) == 0:
            continue

        year_df['sha256'] = year_df['sha256'].str.lower()
        year_df = year_df[year_df['sha256'].isin(hash_file_paths.keys())]

        # Drop rows with missing labels or families
        year_df = year_df.dropna(subset=["label", "family"])
        if len(year_df) == 0:
            print(f"No matching hashes with valid labels/families for year {year}")
            continue

        # Train/test split
        train_df, test_df = train_test_split(
            year_df, test_size=test_size, random_state=42, stratify=year_df['label']
        )

        def prepare_data(sub_df):
            paths = [hash_file_paths[h] for h in sub_df['sha256']]
            X = get_features(paths)
            Y = sub_df['label'].astype(int).values
            Y_family = sub_df['family'].astype(str).values
            vt = sub_df['vt_detection'].fillna(-1).values
            ym = sub_df['year_month'].fillna("unknown").values
            hashes = sub_df['sha256'].values
            return X, Y, Y_family, vt, ym, hashes

        print(f"Processing year {year} — train: {len(train_df)}, test: {len(test_df)}")

        X_tr, y_tr, fam_tr, vt_tr, ym_tr, hash_tr = prepare_data(train_df)
        X_te, y_te, fam_te, vt_te, ym_te, hash_te = prepare_data(test_df)

        np.savez(os.path.join(output_path, f"{year}_Domain_AZ_Train.npz"),
                 X_train=X_tr, Y_train=y_tr, Y_tr_family=fam_tr,
                 Y_tr_vt=vt_tr, Y_tr_ym=ym_tr, Y_tr_hash=hash_tr)

        np.savez(os.path.join(output_path, f"{year}_Domain_AZ_Test.npz"),
                 X_test=X_te, Y_test=y_te, Y_te_family=fam_te,
                 Y_te_vt=vt_te, Y_te_ym=ym_te, Y_te_hash=hash_te)

        print(f"Saved train/test .npz for year {year}\n")

# Run the extraction
extract_save_features(df, data_path)

