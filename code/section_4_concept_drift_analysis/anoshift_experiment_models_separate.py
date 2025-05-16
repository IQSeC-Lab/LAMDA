import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import psutil
import random  # Add this import if not present

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# === MLP ===
class Ember_MLP_Net(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y):
    model = Ember_MLP_Net(X.shape[1]).to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                      torch.tensor(y, dtype=torch.float32)),
                        batch_size=512, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    for _ in range(20):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()
    return model


# === Model classes ===
class ChenEncoderMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 384), nn.ReLU(),
            nn.Linear(384, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(self.encoder(x))

# === Training functions ===
def train_chen_mlp(X, y):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = ChenEncoderMLP(X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    for _ in range(20):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()
    return model

# def train_lightgbm(X, y):
#     # Ensure numpy array and flatten labels
#     X = np.array(X)
#     y = np.array(y).ravel()

#     # Split into train/validation
#     X_tr, X_val, y_tr, y_val = train_test_split(
#         X, y, test_size=0.1, random_state=42, stratify=y
#     )

#     # Define model with optimized parameters
#     model = lgb.LGBMClassifier(
#         boosting_type="gbdt",
#         objective="binary",
#         n_estimators=3000,
#         learning_rate=0.05,
#         num_leaves=128,
#         max_depth=-1,
#         feature_fraction=0.8,
#         bagging_fraction=0.8,
#         bagging_freq=1,
#         lambda_l1=1.0,
#         lambda_l2=1.0,
#         min_child_samples=20,
#         max_bin=255,
#         verbose=-1,
#         n_jobs= 8,
#         metric="auc"
#     )

#     # Train with validation and early stopping
#     print("⚡ Training LightGBM with early stopping...")
#     model.fit(
#         X_tr, y_tr,
#         eval_set=[(X_val, y_val)],
#         eval_metric="auc",
#         callbacks=[
#             early_stopping(stopping_rounds=100),
#             log_evaluation(period=100)
#         ]
#     )

#     return model


def train_lightgbm(X, y):
    # Ensure numpy array and flatten labels
    X = np.array(X)
    y = np.array(y).ravel()

    # Split into train/validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # Define model with tuned parameters
    model = lgb.LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        n_estimators=5000, 
        learning_rate=0.02,    
        num_leaves=256,     
        max_depth=-1,
        min_child_samples=30,  
        subsample=0.8,         
        subsample_freq=1,
        colsample_bytree=0.8, 
        reg_alpha=1.5,           
        reg_lambda=1.5, 
        max_bin=255,
        n_jobs=8,
        verbose=-1,
        metric="auc"
    )

    # Train with validation and early stopping
    print("⚡ Training LightGBM with tuned parameters...")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=100)
        ]
    )

    return model


import xgboost as xgb

def train_xgboost(X, y):
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": 12, "eta": 0.05, "eval_metric": "logloss", "nthread": 8, "verbosity": 1, "tree_method": "gpu_hist"
    }
    return xgb.train(params, dtrain, num_boost_round=3000)


def train_svm(X, y):
    model = CalibratedClassifierCV(LinearSVC(max_iter=10000))
    model.fit(X, y)
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Final Resident memory (RAM) used: {memory_info.rss / 1024 ** 2} MB")
    return model


def evaluate(model, X, y, model_type):
    if model_type == "mlp":
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
    elif model_type == "xgboost":
        preds = model.predict(xgb.DMatrix(X))
    else:
        preds = model.predict_proba(X)[:, 1]

    pred_bin = (preds >= 0.5).astype(int)

    # Compute confusion matrix for FPR and FNR
    cm = confusion_matrix(y, pred_bin)
    fpr = fnr = 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "accuracy": accuracy_score(y, pred_bin),
        "precision": precision_score(y, pred_bin, zero_division=0),
        "recall": recall_score(y, pred_bin, zero_division=0),
        "f1_score": f1_score(y, pred_bin, zero_division=0),
        "roc_auc": roc_auc_score(y, preds),
        "pr_auc": average_precision_score(y, preds),
        "fpr": fpr,
        "fnr": fnr
    }



def _load_npz(prefix, folder, train_test="train"):
    X = load_npz(os.path.join(folder, f"{prefix}_X_{train_test}.npz")).toarray() #.astype(np.int8)
    y = np.load(os.path.join(folder, f"{prefix}_meta_{train_test}.npz"))['y']
    return X, y

# logging information
def run_model(model_name, train_dir, test_dir, run_id, output_file):
    exclude = {"2013-12", "2014-08"}
    train_months = sorted([f[:7] for f in os.listdir(train_dir) if f.endswith("_X_train.npz") and (f.startswith("2013") or f.startswith("2014"))  and f[:7] not in exclude])
    X_train, y_train = [], []
    # Load training data
    print(f"Training on {len(train_months)} months")
    for m in train_months:
        X, y = _load_npz(m, train_dir)
        X_train.append(X)
        y_train.append(y)
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Resident memory (RAM) used: {memory_info.rss / 1024 ** 2} MB")
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    # traning model
    print(f"Model: {model_name.upper()}")
    print(f"Run ID: {run_id}")
    print(f"Training on {X_train.shape[0]} samples")
    print(f"Training on {X_train.shape[1]} features")
    print(f"Training on {len(train_months)} months")
    if model_name == "mlp":
        model = train_mlp(X_train, y_train)
    elif model_name == "lightgbm":
        model = train_lightgbm(X_train, y_train)
    elif model_name == "xgboost":
        model = train_xgboost(X_train, y_train)
    elif model_name == "svm":
        model = train_svm(X_train, y_train)
    else:
        raise ValueError("Unsupported model")

    results = []
    splits = {
        "iid": ["2013-12", "2014-08"],
        "near": ["2016", "2017"],
        "far": ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    }
    # Load test data
    print(f"Testing on {len(splits)} splits")
    # testing on the model
   
    for split, items in splits.items():
        for entry in items:
            months = [f[:7] for f in os.listdir(test_dir) if f.startswith(entry) and f.endswith("_X_test.npz")]
            X_all, y_all = [], []
            for m in months:
                X, y = _load_npz(m, test_dir, train_test="test")
                X_all.append(X)
                y_all.append(y)
            if X_all:
                X_te, y_te = np.vstack(X_all), np.concatenate(y_all)
                metrics = evaluate(model, X_te, y_te, model_name)
                metrics.update({"model": model_name.upper(), "year": entry, "split": split, "run": run_id})
                results.append(metrics)

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Final Resident memory (RAM) used: {memory_info.rss / 1024 ** 2} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_model.py <model_name> <run_id>")
        sys.exit(1)

    model = sys.argv[1].lower()
    run = int(sys.argv[2])
    set_random_seed(run)
    VT = 0.0001
    train_dir = f"/home/shared-datasets/Feature_extraction/npz_monthwise_Final_train_{VT}"
    test_dir = f"/home/shared-datasets/Feature_extraction/npz_monthwise_Final_test_{VT}"
    os.makedirs(f"results_by_model_{VT}", exist_ok=True)
    out_path = f"results_by_model_{VT}/{model}_anoshift_run{run}.csv"
    run_model(model, train_dir, test_dir, run, out_path)
    












# # Previous code

# import os
# import sys
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from scipy.sparse import load_npz
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     roc_auc_score, confusion_matrix, average_precision_score
# )
# from sklearn.svm import LinearSVC
# from sklearn.calibration import CalibratedClassifierCV
# import xgboost as xgb
# import lightgbm as lgb

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # === Model classes ===
# class ChenEncoderMLP(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 512), nn.ReLU(),
#             nn.Linear(512, 384), nn.ReLU(),
#             nn.Linear(384, 256), nn.ReLU(),
#             nn.Linear(256, 128), nn.ReLU()
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(128, 100), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(100, 1), nn.Sigmoid()
#         )
#     def forward(self, x):
#         return self.classifier(self.encoder(x))

# class XGBoostWrapper:
#     def __init__(self, booster):
#         self.booster = booster
#     def predict_proba(self, X):
#         preds = self.booster.predict(xgb.DMatrix(X))
#         return np.vstack([1 - preds, preds]).T

# # === Training functions ===
# def train_chen_mlp(X, y):
#     dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
#                             torch.tensor(y, dtype=torch.float32))
#     loader = DataLoader(dataset, batch_size=64, shuffle=True)
#     model = ChenEncoderMLP(X.shape[1]).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.BCELoss()
#     for _ in range(20):
#         model.train()
#         for xb, yb in loader:
#             xb, yb = xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(xb).squeeze(), yb)
#             loss.backward()
#             optimizer.step()
#     return model

# def train_lightgbm_slow(X, y):
#     X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#     dtrain = lgb.Dataset(X_tr, label=y_tr)
#     dval = lgb.Dataset(X_val, label=y_val)
#     params = {
#         "boosting_type": "gbdt", "objective": "binary", "metric": ["binary_logloss", "auc"],
#         "learning_rate": 0.05, "num_leaves": 1024,
#         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5
#     }
#     return lgb.train(params, dtrain, 5000, valid_sets=[dval], callbacks=[lgb.early_stopping(100)])

# def train_lightgbm(X, y):
#     X = np.array(X)
#     y = np.array(y).ravel()  # Flatten y to 1D

#     X_tr, X_val, y_tr, y_val = train_test_split(
#         X, y, test_size=0.1, random_state=42, stratify=y
#     )
#     dtrain = lgb.Dataset(X_tr, label=y_tr)
#     dval = lgb.Dataset(X_val, label=y_val)

#     params = {
#         "boosting_type": "gbdt",
#         "objective": "binary",
#         "metric": ["binary_logloss", "auc"],
#         "learning_rate": 0.05,
#         "num_leaves": 128,
#         "max_depth": -1,
#         "feature_fraction": 0.8,
#         "bagging_fraction": 0.8,
#         "bagging_freq": 1,
#         "lambda_l1": 1.0,
#         "lambda_l2": 1.0,
#         "min_data_in_leaf": 20,
#         "max_bin": 255,
#         "verbosity": -1,
#         "num_threads": int(os.cpu_count()/4),
#     }

#     print("⚡ Training LightGBM...")
#     model = lgb.train(
#         params,
#         dtrain,
#         num_boost_round=3000,
#         valid_sets=[dtrain, dval],
#         valid_names=["train", "val"],
#         callbacks=[
#             lgb.early_stopping(stopping_rounds=100),
#             lgb.log_evaluation(period=100)
#         ]
#     )
#     return model



# def train_xgboost(X, y):
#     dtrain = xgb.DMatrix(X, label=y)
#     booster = xgb.train({"max_depth": 12, "eta": 0.05, "eval_metric": "error"},
#                         dtrain, num_boost_round=5000)
#     return XGBoostWrapper(booster)

# def train_svm(X, y):
#     model = CalibratedClassifierCV(LinearSVC(max_iter=10000))
#     model.fit(X, y)
#     return model

# # === Evaluation ===
# def compute_metrics(y_true, y_pred_probs, threshold=0.5):
#     y_pred = (y_pred_probs >= threshold).astype(int)
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, zero_division=0)
#     recall = recall_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     roc_auc = roc_auc_score(y_true, y_pred_probs)
#     pr_auc = average_precision_score(y_true, y_pred_probs)
#     cm = confusion_matrix(y_true, y_pred)
#     fpr = fnr = 0
#     if cm.shape == (2, 2):
#         tn, fp, fn, tp = cm.ravel()
#         fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
#         fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
#     return {
#         "accuracy": acc, "precision": prec, "recall": recall, "f1_score": f1,
#         "roc_auc": roc_auc, "pr_auc": pr_auc, "fpr": fpr, "fnr": fnr
#     }

# def evaluate_model(model, X, y, model_type):
#     if model_type == "mlp":
#         model.eval()
#         with torch.no_grad():
#             preds = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
#     elif model_type == "lightgbm":
#         preds = model.predict(X)
#     else:
#         preds = model.predict_proba(X)[:, 1]
#     return compute_metrics(y, preds)

# # === Load NPZ ===
# def load_npz_month(month, base_dir):
#     try:
#         X = load_npz(f"{base_dir}/{month}_X.npz").toarray()
#         y = np.load(f"{base_dir}/{month}_meta.npz")["y"]
#         return X, y
#     except:
#         return None, None

# # === Main function ===
# def run_model(model_name, base_dir, run_id, output_path):
#     all_results = []

#     train_months, test_months_split = [], []
#     for year in [2013, 2014]:
#         months = sorted([f[:-6] for f in os.listdir(base_dir)
#                          if f.startswith(str(year)) and f.endswith("_X.npz")])
#         if not months: continue
#         last_month = max(months)
#         train_months += [m for m in months if m != last_month]
#         test_months_split.append((last_month, "iid"))

#     X_train, y_train = [], []
#     for m in train_months:
#         X, y = load_npz_month(m, base_dir)
#         if X is not None:
#             X_train.append(X)
#             y_train.append(y)
#     X_train = np.vstack(X_train)
#     y_train = np.concatenate(y_train)
#     print(f"Training on {len(train_months)} months, {X_train.shape[0]} samples")
#     print(f"Model: {model_name.upper()}")
#     print(f"Run ID: {run_id}")
#     if model_name == "mlp":
#         model = train_chen_mlp(X_train, y_train)
#     elif model_name == "lightgbm":
#         model = train_lightgbm(X_train, y_train)
#     elif model_name == "xgboost":
#         model = train_xgboost(X_train, y_train)
#     elif model_name == "svm":
#         model = train_svm(X_train, y_train)
#     else:
#         raise ValueError("Unsupported model")

#     print(f"Testing on {len(test_months_split)} months")
#     for month, split in test_months_split:
#         X, y = load_npz_month(month, base_dir)
#         if X is not None:
#             result = evaluate_model(model, X, y, model_name)
#             result.update({"model": model_name.upper(), "year": month, "split": split, "run": run_id})
#             all_results.append(result)

#     for split, years in [("near", [2016, 2017]), ("far", list(range(2018, 2026)))]:
#         for year in years:
#             months = sorted([f[:-6] for f in os.listdir(base_dir)
#                              if f.startswith(str(year)) and f.endswith("_X.npz")])
#             X_all, y_all = [], []
#             for m in months:
#                 X, y = load_npz_month(m, base_dir)
#                 if X is not None:
#                     X_all.append(X)
#                     y_all.append(y)
#             if X_all:
#                 X_combined = np.vstack(X_all)
#                 y_combined = np.concatenate(y_all)
#                 result = evaluate_model(model, X_combined, y_combined, model_name)
#                 result.update({"model": model_name.upper(), "year": year, "split": split, "run": run_id})
#                 all_results.append(result)

#     df = pd.DataFrame(all_results)
#     df.to_csv(output_path, index=False)
#     print(f"✅ Results saved to {output_path}")

# # === Entry point ===
# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python run_anoshift_model.py <model_name> <run_number>")
#         sys.exit(1)

#     model = sys.argv[1].lower()
#     run = int(sys.argv[2])
#     base_dir = "/home/shared-datasets/Feature_extraction/npz_monthwise_Final"
#     output_dir = "results_by_model"
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = f"{model}_anoshift_run{run}.csv"
#     run_model(model, base_dir, run, os.path.join(output_dir, output_file))
