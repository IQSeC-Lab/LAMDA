import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global font settings
plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "legend.frameon": False
})

# Load VirusTotal data
vt_df = pd.read_csv("LAMDA/metadata.csv")
vt_df['vt_detection'] = pd.to_numeric(vt_df['vt_detection'], errors='coerce')
vt_df['sha256'] = vt_df['sha256'].str.lower()

# Directory containing meta files
data_dir = "/LAMDA/Baseline_npz"
thresholds = range(1, 11)
results = {t: {'benign': 0, 'malware': 0} for t in thresholds}

# Aggregate counts across all years except 2015
for year in range(2013, 2026):
    if year == 2015:
        continue

    meta_file1 = os.path.join(data_dir, f"{year}_meta_train.npz")
    meta_file2 = os.path.join(data_dir, f"{year}_meta_test.npz")
    if not os.path.exists(meta_file1):
        continue

    
    meta1 = np.load(meta_file1, allow_pickle=True)
    meta2 = np.load(meta_file2, allow_pickle=True)
    
    for meta in [meta1, meta2]:
        hashes = meta['hash']
        hash_df = pd.DataFrame({'sha256': [h.lower() for h in hashes]})
        merged_df = pd.merge(hash_df, vt_df, on='sha256', how='inner')

        for t in thresholds:
            filtered = merged_df[(merged_df['vt_detection'] == 0) | (merged_df['vt_detection'] >= t)].copy()
            filtered['label'] = filtered['vt_detection'].apply(lambda x: 0 if x == 0 else 1)
            results[t]['benign'] += (filtered['label'] == 0).sum()
            results[t]['malware'] += (filtered['label'] == 1).sum()


# Reference at threshold 4
ref_benign = results[4]['benign']
ref_malware = results[4]['malware']
ref_total = ref_benign + ref_malware

# Calculate % change relative to threshold 4
summary = []
for t in thresholds:
    benign = results[t]['benign']
    malware = results[t]['malware']
    total = benign + malware

    benign_pct = 100 * (benign - ref_benign) / ref_benign if ref_benign > 0 else 0
    malware_pct = 100 * (malware - ref_malware) / ref_malware if ref_malware > 0 else 0
    total_pct = 100 * (total - ref_total) / ref_total if ref_total > 0 else 0

    summary.append({
        'threshold': t,
        'benign_samples': benign,
        'malware_samples': malware,
        'total_samples': total,
        'benign_%_change_vs_4': benign_pct,
        'malware_%_change_vs_4': malware_pct,
        'total_%_change_vs_4': total_pct
    })

df = pd.DataFrame(summary)
df.to_csv("vt_threshold_change_vs_4.csv", index=False)
print("Saved to vt_threshold_change_vs_4.csv")

# Plot % changes
plt.figure(figsize=(10, 6))
plt.plot(df['threshold'], df['benign_%_change_vs_4'], marker='o', label='Benign % change')
plt.plot(df['threshold'], df['malware_%_change_vs_4'], marker='o', label='Malware % change')
plt.plot(df['threshold'], df['total_%_change_vs_4'], marker='o', label='Total % change')
plt.axvline(4, color='gray', linestyle='--', label='Reference (VT=4)')
plt.axhline(0, linestyle='--', color='black')

# plt.title("Percentage Change in Sample Counts vs VT Detection Threshold (Relative to VT=4)")
plt.xlabel("VT Detection Threshold")
plt.ylabel("Change in percentage(%)  w.r.t VT=4")
plt.xticks(thresholds)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vt_threshold_pct_change_vs_4.png", dpi=300)
plt.show()


# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load VirusTotal data
# vt_df = pd.read_csv("/home/shared-datasets/Feature_extraction/all_hash_vtdetect_added.csv")
# vt_df['vt_detection'] = pd.to_numeric(vt_df['vt_detection'], errors='coerce')
# vt_df['sha256'] = vt_df['sha256'].str.lower()  # Normalize casing for matching

# data_dir = "/home/shared-datasets/Feature_extraction/npz_yearwise_Final"
# results = {threshold: {'benign': 0, 'malware': 0} for threshold in range(4, 11)}

# # Process each year
# for year in range(2013, 2026):
#     if year == 2015:
#         continue

#     meta_file = os.path.join(data_dir, f"{year}_meta_with_family.npz")
#     if not os.path.exists(meta_file):
#         print(f"[Warning] File not found for year {year}, skipping...")
#         continue

#     meta = np.load(meta_file, allow_pickle=True)
#     hashes = meta['hash']
#     hash_df = pd.DataFrame({'sha256': [h.lower() for h in hashes]})  # Match VT column

#     merged_df = pd.merge(hash_df, vt_df, on='sha256', how='inner')

#     for threshold in range(4, 11):
#         # Apply detection threshold filtering
#         filtered = merged_df[(merged_df['vt_detection'] == 0) | (merged_df['vt_detection'] >= threshold)].copy()
#         filtered['new_y'] = filtered['vt_detection'].apply(lambda x: 0 if x == 0 else 1)

#         results[threshold]['benign'] += (filtered['new_y'] == 0).sum()
#         results[threshold]['malware'] += (filtered['new_y'] == 1).sum()

# # Compile results
# summary = []
# for threshold in range(4, 11):
#     benign = results[threshold]['benign']
#     malware = results[threshold]['malware']
#     total = benign + malware
#     summary.append({
#         'vt_detection_threshold': threshold,
#         'benign_samples': benign,
#         'malware_samples': malware,
#         'total_samples': total
#     })

# results_df = pd.DataFrame(summary)
# results_df.to_csv("vt_threshold_summary.csv", index=False)
# print("Saved results to vt_threshold_summary.csv")

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(results_df['vt_detection_threshold'], results_df['benign_samples'], marker='o', label='Benign Samples')
# plt.plot(results_df['vt_detection_threshold'], results_df['malware_samples'], marker='o', label='Malware Samples')
# plt.plot(results_df['vt_detection_threshold'], results_df['total_samples'], marker='o', label='Total Samples')

# plt.title('VT Detection Threshold Impact (2013–2025, excluding 2015)')
# plt.xlabel('VT Detection Threshold')
# plt.ylabel('Number of Samples')
# plt.xticks(results_df['vt_detection_threshold'])
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("vt_detection_threshold_impact.png", dpi=300)
# plt.show()

