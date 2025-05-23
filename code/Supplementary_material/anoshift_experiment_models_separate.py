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
        n_estimators=5000,                # allow for longer training (early stopping will cut it)
        learning_rate=0.02,               # slower learning rate for better convergence
        num_leaves=256,                   # increased capacity
        max_depth=-1,
        min_child_samples=30,             # prevent overfitting on small noisy leaves
        subsample=0.8,                    # row sampling (replacement for bagging_fraction)
        subsample_freq=1,
        colsample_bytree=0.8,             # feature sampling (replacement for feature_fraction)
        reg_alpha=1.5,                    # stronger L1 regularization
        reg_lambda=1.5,                   # stronger L2 regularization
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
    train_dir = f"/var_thresh_{VT}_monthwise/train"
    test_dir = f"/var_thresh_{VT}_monthwise/test"
    os.makedirs(f"results_by_model_{VT}", exist_ok=True)
    out_path = f"results_by_model_{VT}/{model}_anoshift_run{run}.csv"
    run_model(model, train_dir, test_dir, run, out_path)
    

