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

            print(f"Saved {ym} - Total: {len(ym_y)}, Malware: {np.sum(ym_y==1)}, Benign: {np.sum(ym_y==0)}")

    except Exception as e:
        print(f"Skipping year {year} due to error: {e}")

# Save summary
summary_df = pd.DataFrame(summary_records)
summary_df = summary_df.sort_values(by="year_month")
summary_df.to_csv(os.path.join(output_dir, "year_month_split_summary.csv"), index=False)
print(f"\nSaved summary to: year_month_split_summary.csv")

# Log skipped
if skipped_samples > 0:
    print(f"Skipped {skipped_samples} samples due to 'unknown' year_month.")
else:
    print("No skipped samples — all entries had valid year_month.")


