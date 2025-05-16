import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from collections import defaultdict

TRAIN_TEST = 'test'
VT = 0.0001
input_dir = f"/home/shared-datasets/Feature_extraction/npz_yearwise_Final_{VT}"
output_dir = f"/home/shared-datasets/Feature_extraction/npz_monthwise_Final_{TRAIN_TEST}_{VT}"
os.makedirs(output_dir, exist_ok=True)

years = [2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

summary_records = []
skipped_samples = 0

for year in years:
    print(f"\nProcessing year: {year}")
    try:
        # Load data
        X = load_npz(os.path.join(input_dir, f"{year}_X_{TRAIN_TEST}.npz"))
        meta = np.load(os.path.join(input_dir, f"{year}_meta_{TRAIN_TEST}.npz"), allow_pickle=True)

        y = meta["y"]
        family = meta["family"]
        vt_count = meta["vt_count"]
        year_month = meta["year_month"]

        assert len(y) == X.shape[0] == len(family) == len(vt_count) == len(year_month)

        # Group by year_month
        ym_indices = defaultdict(list)
        for idx, ym in enumerate(year_month):
            if ym != "unknown":
                ym_indices[ym].append(idx)
            else:
                skipped_samples += 1

        # Save each group
        for ym, indices in ym_indices.items():
            ym_X = X[indices]
            ym_y = y[indices]
            ym_family = family[indices]
            ym_vt = vt_count[indices]
            ym_ym = year_month[indices]

            # Save files
            save_npz(os.path.join(output_dir, f"{ym}_X_{TRAIN_TEST}.npz"), ym_X)
            # np.savez_compressed(os.path.join(output_dir, f"{ym}_meta.npz"), y=ym_y)
            np.savez_compressed(
                os.path.join(output_dir, f"{ym}_meta_{TRAIN_TEST}.npz"),
                y=ym_y, family=ym_family, vt_count=ym_vt, year_month=ym_ym
            )

            summary_records.append({
                "year_month": ym,
                "total": len(ym_y),
                "malware": int(np.sum(ym_y == 1)),
                "benign": int(np.sum(ym_y == 0))
            })

            print(f"[✔] Saved {ym} - Total: {len(ym_y)}, Malware: {np.sum(ym_y==1)}, Benign: {np.sum(ym_y==0)}")

    except Exception as e:
        print(f"[!] Skipping year {year} due to error: {e}")

# Save summary
summary_df = pd.DataFrame(summary_records)
summary_df = summary_df.sort_values(by="year_month")
summary_df.to_csv(os.path.join(output_dir, "year_month_split_summary.csv"), index=False)
print(f"\n[✔] Saved summary to: year_month_split_summary.csv")

# Log skipped
if skipped_samples > 0:
    print(f"[!] Skipped {skipped_samples} samples due to 'unknown' year_month.")
else:
    print("[✔] No skipped samples — all entries had valid year_month.")



# import os
# import numpy as np
# import pandas as pd
# from scipy.sparse import load_npz, save_npz
# from collections import defaultdict

# input_dir = "/home/shared-datasets/Feature_extraction/npz_yearwise_Final"
# output_dir = "/home/shared-datasets/Feature_extraction/npz_monthwise_Final"
# os.makedirs(output_dir, exist_ok=True)

# years = [2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# summary_records = []
# skipped_hashes = []

# for year in years:
#     print(f"\nProcessing year: {year}")

#     try:
#         # Load data
#         X = load_npz(os.path.join(input_dir, f"{year}_X_train.npz"))
#         # meta = np.load(os.path.join(input_dir, f"{year}_meta_.train.npz"), allow_pickle=True)
#         meta_full = np.load(os.path.join(input_dir, f"{year}_meta_train.npz"), allow_pickle=True)

#         y = meta_full["y"]
#         hash_arr = meta_full["hash"]
#         y_full = meta_full["y"]
#         family = meta_full["family"]
#         vt_count = meta_full["vt_count"]
#         year_month = meta_full["year_month"]

#         assert len(y) == len(hash_arr) == X.shape[0]

#         # Group by year_month
#         ym_indices = defaultdict(list)
#         for idx, ym in enumerate(year_month):
#             if ym != "unknown":
#                 ym_indices[ym].append(idx)
#             else:
#                 skipped_hashes.append(hash_arr[idx])

#         # Save each year_month group
#         for ym, indices in ym_indices.items():
#             ym_X = X[indices]
#             ym_y = y[indices]
#             ym_hash = hash_arr[indices]
#             ym_family = family[indices]
#             ym_vt = vt_count[indices]
#             ym_ym = year_month[indices]

#             # Save files
#             save_npz(os.path.join(output_dir, f"{ym}_X.npz"), ym_X)
#             np.savez_compressed(os.path.join(output_dir, f"{ym}_meta.npz"), y=ym_y)
#             np.savez_compressed(
#                 os.path.join(output_dir, f"{ym}_meta_with_family_vtcount_ym.npz"),
#                 hash=ym_hash,
#                 y=ym_y,
#                 family=ym_family,
#                 vt_count=ym_vt,
#                 year_month=ym_ym
#             )

#             # Record statistics
#             num_malware = int(np.sum(ym_y == 1))
#             num_benign = int(np.sum(ym_y == 0))
#             num_total = len(ym_y)
#             summary_records.append({
#                 "year_month": ym,
#                 "total": num_total,
#                 "malware": num_malware,
#                 "benign": num_benign
#             })

#             print(f"[✔] Saved {ym} - Total: {num_total}, Malware: {num_malware}, Benign: {num_benign}")

#     except Exception as e:
#         print(f"[!] Skipping {year} due to error: {e}")

# # Save summary statistics
# summary_df = pd.DataFrame(summary_records)
# summary_df = summary_df.sort_values(by="year_month")
# summary_df.to_csv(os.path.join(output_dir, "year_month_split_summary.csv"), index=False)
# print(f"\n[✔] Saved summary to: year_month_split_summary.csv")

# # Save skipped hashes (unknown year_month)
# if skipped_hashes:
#     with open(os.path.join(output_dir, "skipped_hashes_due_to_unknown_ym.txt"), "w") as f:
#         for h in skipped_hashes:
#             f.write(f"{h}\n")
#     print(f"[!] Skipped {len(skipped_hashes)} samples due to unknown year_month.")
#     print(f"[✔] Logged to: skipped_hashes_due_to_unknown_ym.txt")
# else:
#     print("[✔] No skipped samples — all entries had valid year_month.")






## Previously used to split the npz files by month using the 'added' date
# from latest_with_year_month.csv 


# import os
# import numpy as np
# import pandas as pd
# from scipy.sparse import load_npz, save_npz

# # SETTINGS
# years = [2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
# hashmap_csv = "/home/mhaque3/myDir/concept_drift_dataset/latest_with-year_month.csv"
# npz_path = "/home/shared-datasets/Feature_extraction/npz_yearwise"
# output_dir = "npz_monthwise"
# os.makedirs(output_dir, exist_ok=True)

# # LOAD HASH TO MONTH MAPPING (with 'sha256' and 'added' columns)
# df_map = pd.read_csv(hashmap_csv, usecols=['sha256', 'added', 'year_month', 'label'])
# # df_map = df_map.dropna(subset=['sha256', 'added'])
# # df_map['sha256'] = df_map['sha256'].str.lower()  # normalize case
# # df_map['added'] = pd.to_datetime(df_map['added'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
# # df_map = df_map.dropna(subset=['added'])
# # df_map['year_month'] = df_map['added'].dt.to_period('M').astype(str)
# hash_to_month = dict(zip(df_map['sha256'], df_map['year_month']))

# for year in years:
#     X_path = f"{npz_path}/{year}_X.npz"
#     meta_path = f"{npz_path}/{year}_meta.npz"

#     if not os.path.exists(X_path) or not os.path.exists(meta_path):
#         print(f"Skipping {year}: missing input files.")
#         continue

#     # LOAD DATA
#     X = load_npz(X_path)  # assuming key is 'X'
#     meta = np.load(meta_path, allow_pickle=True)
#     meta_hashes = meta['hash']
#     meta_labels = meta['y']
#     meta_families = meta['family']

#     # GROUP INDICES BY MONTH
#     month_indices = {}
#     for i, h in enumerate(meta_hashes):
#         h_lower = h.lower()  # normalize case for lookup
#         month = hash_to_month.get(h_lower)
#         if month:
#             month_indices.setdefault(month, []).append(i)

#     # SPLIT AND SAVE PER MONTH
#     for month, indices in month_indices.items():
#         X_month = X[indices]
#         hash_month = meta_hashes[indices]
#         label_month = meta_labels[indices]
#         family_month = meta_families[indices]

#         save_npz(os.path.join(output_dir, f"{month}_X.npz"), X_month)
#         np.savez_compressed(os.path.join(output_dir, f"{month}_meta.npz"),
#                             hash=hash_month, y=label_month, family=family_month)
#         print(f"{year}: Saved {month} with {len(indices)} samples.")
