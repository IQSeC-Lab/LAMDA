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
import random

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

def train_lightgbm(X, y):
    X = np.array(X)
    y = np.array(y).ravel()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
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

def run_model_monthwise(model_name, train_file, test_dir, run_id, output_file):
    # Load training data (2012)
    print(f"Training on {train_file}")
    data = np.load(train_file, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']

    print(f"Model: {model_name.upper()}")
    print(f"Run ID: {run_id}")
    print(f"Training on {X_train.shape[0]} samples")
    print(f"Training on {X_train.shape[1]} features")
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
    # Test on all available months (skip 2012, optionally skip 2015)
    for year in range(2013, 2019):
        if year == 2015:
            continue
        for month in range(1, 13):
            test_file = os.path.join(test_dir, f"{year}-{month:02d}.npz")
            if not os.path.exists(test_file):
                continue
            data = np.load(test_file, allow_pickle=True)
            X_te, y_te = data['X_train'], data['ytrain']
            metrics = evaluate(model, X_te, y_te, model_name)
            metrics.update({
                "model": model_name.upper(),
                "year": year,
                "month": month,
                "run": run_id
            })
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
    train_file = "/home/shared-datasets/gen_apigraph_drebin/2012-01to2012-12_selected.npz"
    test_dir = "/home/shared-datasets/gen_apigraph_drebin"
    os.makedirs("results_by_model_api_month", exist_ok=True)
    out_path = f"results_by_model_api_month/{model}_anoshift_monthwise_run{run}.csv"
    run_model_monthwise(model, train_file, test_dir, run, out_path)

