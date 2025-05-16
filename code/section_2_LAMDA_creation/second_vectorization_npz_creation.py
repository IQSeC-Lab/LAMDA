import os
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import lil_matrix, save_npz
from tqdm import tqdm
import joblib

raw_path = '/home/shared-datasets/Feature_extraction/Final_AZ_Data'
save_path = 'npz_yearwise_Final_rerun_0.001'
os.makedirs(save_path, exist_ok=True)

years = ['2013', '2014', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']

# Collect all data
all_X_tr, all_Y_tr, all_Y_tr_family, all_Y_tr_vt, all_Y_tr_ym, all_Y_tr_hash = [], [], [], [], [], []
all_X_te, all_Y_te, all_Y_te_family, all_Y_te_vt, all_Y_te_ym, all_Y_te_hash = [], [], [], [], [], []

for year in years:
    tr_file = os.path.join(raw_path, f"{year}_Domain_AZ_Train.npz")
    te_file = os.path.join(raw_path, f"{year}_Domain_AZ_Test.npz")

    tr = np.load(tr_file, allow_pickle=True)
    te = np.load(te_file, allow_pickle=True)

    all_X_tr.extend(tr['X_train'])
    all_Y_tr.extend(tr['Y_train'])
    all_Y_tr_family.extend(tr['Y_tr_family'])
    all_Y_tr_vt.extend(tr['Y_tr_vt'])
    all_Y_tr_ym.extend(tr['Y_tr_ym'])
    all_Y_tr_hash.extend(tr['Y_tr_hash'])

    all_X_te.extend(te['X_test'])
    all_Y_te.extend(te['Y_test'])
    all_Y_te_family.extend(te['Y_te_family'])
    all_Y_te_vt.extend(te['Y_te_vt'])
    all_Y_te_ym.extend(te['Y_te_ym'])
    all_Y_te_hash.extend(te['Y_te_hash'])

# --- Vocabulary and Vectorization ---
def build_vocabulary(data):
    vocab = set(word for sample in data for word in sample)
    return sorted(vocab)

def vectorize_samples_sparse(data, vocabulary):
    vocab_index = {word: idx for idx, word in enumerate(vocabulary)}
    mat = lil_matrix((len(data), len(vocabulary)), dtype=int)
    for i, sample in enumerate(data):
        for word in sample:
            if word in vocab_index:
                mat[i, vocab_index[word]] += 1
    return mat.tocsr()

# Build and vectorize training data
print("Building vocabulary and vectorizing training data...")
vocab = build_vocabulary(all_X_tr)
print(f"Vocabulary size: {len(vocab)}")

# Save vocabulary (as .npy and optionally as .txt for readability)
np.save(os.path.join(save_path, "vocabulary.npy"), np.array(vocab))

# Optional: Save as plain text (one word per line)
with open(os.path.join(save_path, "vocabulary.txt"), "w") as f:
    for word in vocab:
        f.write(f"{word}\n")


X_train_sparse = vectorize_samples_sparse(all_X_tr, vocab)
X_test_sparse = vectorize_samples_sparse(all_X_te, vocab)

# Save full raw features before selection
save_npz(os.path.join(save_path, "ALL_X_train_raw.npz"), X_train_sparse)
save_npz(os.path.join(save_path, "ALL_X_test_raw.npz"), X_test_sparse)

# Apply Variance Threshold
print("Applying VarianceThreshold...")
selector = VarianceThreshold(threshold=0.0001)
X_train_sel = selector.fit_transform(X_train_sparse)
X_test_sel = selector.transform(X_test_sparse)

# Save selected vocabulary (after variance threshold)
selected_mask = selector.get_support()
vocab_after_selection = np.array(vocab)[selected_mask]

np.save(os.path.join(save_path, "vocabulary_selected.npy"), vocab_after_selection)

# Optional: Save selected vocab as plain text
with open(os.path.join(save_path, "vocabulary_selected.txt"), "w") as f:
    for word in vocab_after_selection:
        f.write(f"{word}\n")

print(f"✅ Saved selected vocabulary with {len(vocab_after_selection)} features.")


# Save VarianceThreshold object
joblib.dump(selector, os.path.join(save_path, "variance_selector.joblib"))

# Save final selected features
save_npz(os.path.join(save_path, "ALL_X_train.npz"), X_train_sel)
save_npz(os.path.join(save_path, "ALL_X_test.npz"), X_test_sel)

# Save metadata
np.savez_compressed(os.path.join(save_path, "ALL_meta_train.npz"),
                    y=np.array(all_Y_tr),
                    family=np.array(all_Y_tr_family),
                    vt_count=np.array(all_Y_tr_vt),
                    year_month=np.array(all_Y_tr_ym),
                    hash=np.array(all_Y_tr_hash))

np.savez_compressed(os.path.join(save_path, "ALL_meta_test.npz"),
                    y=np.array(all_Y_te),
                    family=np.array(all_Y_te_family),
                    vt_count=np.array(all_Y_te_vt),
                    year_month=np.array(all_Y_te_ym),
                    hash=np.array(all_Y_te_hash))

# --- Save Per Year (raw and transformed) ---
print("Saving transformed data per year...")
for year in tqdm(years):
    tr = np.load(os.path.join(raw_path, f"{year}_Domain_AZ_Train.npz"), allow_pickle=True)
    te = np.load(os.path.join(raw_path, f"{year}_Domain_AZ_Test.npz"), allow_pickle=True)

    X_tr_raw = vectorize_samples_sparse(tr['X_train'], vocab)
    X_te_raw = vectorize_samples_sparse(te['X_test'], vocab)

    X_tr_sel = selector.transform(X_tr_raw)
    X_te_sel = selector.transform(X_te_raw)

    # Save raw
    # save_npz(os.path.join(save_path, f"{year}_X_train_raw.npz"), X_tr_raw)
    # save_npz(os.path.join(save_path, f"{year}_X_test_raw.npz"), X_te_raw)

    # Save selected
    save_npz(os.path.join(save_path, f"{year}_X_train.npz"), X_tr_sel)
    save_npz(os.path.join(save_path, f"{year}_X_test.npz"), X_te_sel)

    # Save metadata
    np.savez_compressed(os.path.join(save_path, f"{year}_meta_train.npz"),
                        y=tr['Y_train'],
                        family=tr['Y_tr_family'],
                        vt_count=tr['Y_tr_vt'],
                        year_month=tr['Y_tr_ym'],
                        hash=tr['Y_tr_hash'])

    np.savez_compressed(os.path.join(save_path, f"{year}_meta_test.npz"),
                        y=te['Y_test'],
                        family=te['Y_te_family'],
                        vt_count=te['Y_te_vt'],
                        year_month=te['Y_te_ym'],
                        hash=te['Y_te_hash'])

print("✅ All raw and transformed data saved yearwise. Variance selector saved.")


