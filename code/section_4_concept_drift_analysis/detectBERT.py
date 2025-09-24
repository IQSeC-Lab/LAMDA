import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from nystrom_attention import NystromAttention


# ======================================================
# Utility Functions
# ======================================================
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_ds(X, y):
    return TensorDataset(torch.tensor(X, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.long))


def _load_npz(prefix, folder, train_test="train"):
    X = load_npz(os.path.join(folder, f"{prefix}_X_{train_test}.npz")).toarray()
    y = np.load(os.path.join(folder, f"{prefix}_meta_{train_test}.npz"))['y']
    return X, y


# ======================================================
# Model Definition
# ======================================================
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class DetectBERT(nn.Module):
    def __init__(self, cfg, n_classes, input_size=128, hidden_size=128):
        super(DetectBERT, self).__init__()
        self.cfg = cfg
        self._fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=hidden_size)
        self.layer2 = TransLayer(dim=hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self._fc2 = nn.Linear(hidden_size, self.n_classes)

    def forward(self, *args, **kwargs):
        data = args[0] if args else kwargs.get('data', kwargs.get('x', None))
        if data is None:
            raise KeyError("DetectBERT.forward expects input tensor")

        h = data.float()
        if h.dim() == 2:  # (B, D) → (B, 1, D)
            h = h.unsqueeze(1)

        h = self._fc1(h)
        agg = self.cfg['Model']['aggregation']

        if agg == "DetectBERT":
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
            h = torch.cat((cls_tokens, h), dim=1)
            h = self.layer1(h)
            h = self.layer2(h)
            h = self.norm(h)[:, 0]
        elif agg == "addition":
            h = h.sum(dim=1)
        elif agg == "average":
            h = h.mean(dim=1)
        elif agg == "random":
            random_index = torch.randint(0, h.size(1), (1,), device=h.device)
            h = h[:, random_index.item(), :]

        logits = self._fc2(h)
        return logits, h


# ======================================================
# Evaluation Function
# ======================================================
from sklearn.preprocessing import label_binarize
def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits, _ = model(data=tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    n_classes = probs.shape[1]
    y_arr = np.asarray(y)

    if n_classes == 2:
        y_true = (y_arr >= 1).astype(int) if set(np.unique(y_arr)) - {0, 1} else y_arr.astype(int)
        prob_pos = probs[:, 1]
        pred_bin = (prob_pos >= 0.5).astype(int)
        average = 'binary'
    else:
        y_true = y_arr.astype(int)
        pred_bin = probs.argmax(axis=1)
        average = 'macro'

    cm = confusion_matrix(y_true, pred_bin)
    fpr = fnr = 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    precision = precision_score(y_true, pred_bin, average=average, zero_division=0)
    recall = recall_score(y_true, pred_bin, average=average, zero_division=0)
    f1 = f1_score(y_true, pred_bin, average=average, zero_division=0)

    if n_classes == 2:
        roc_auc = roc_auc_score(y_true, prob_pos)
        pr_auc = average_precision_score(y_true, prob_pos)
    else:
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        roc_auc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
        pr_auc = average_precision_score(y_bin, probs, average='macro')

    return {
        "accuracy": accuracy_score(y_true, pred_bin),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "fnr": fnr
    }


# ======================================================
# Main Script
# ======================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DetectBERT Training and Evaluation")

    # dataset choice
    parser.add_argument(
        "--dataset",
        type=str,
        default="lamda",
        choices=["lamda", "apigraph"],
        help="Which dataset to use (LAMDA or APIGraph)"
    )

    args = parser.parse_args()
    set_random_seed(19)

    # -------------------------
    # Dataset Loading
    # -------------------------
    if args.dataset == "lamda":
        train_dir = "/LAMDA_dataset/Baseline_npz_monthwise/train"
        test_dir  = "/LAMDA_dataset/Baseline_npz_monthwise/test"

        exclude = {"2013-12", "2014-08"}
        train_months = sorted([
            f[:7] for f in os.listdir(train_dir)
            if f.endswith("_X_train.npz") and (f.startswith("2013") or f.startswith("2014")) and f[:7] not in exclude
        ])

        X_train, y_train = [], []
        for m in train_months:
            X, y = _load_npz(m, train_dir)
            if X.shape[0] == y.shape[0]:
                X_train.append(X)
                y_train.append(y)
        X = np.vstack(X_train)
        y = np.concatenate(y_train)

    elif args.dataset == "apigraph":
        train_file = "/home/shared-datasets/gen_apigraph_drebin/2012-01to2012-12_selected.npz"
        dataset_npz = np.load(train_file, allow_pickle=True)
        X, y = dataset_npz['X_train'], dataset_npz['y_train']
        y = (y >= 1).astype(np.int64)

    # Split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # -------------------------
    # DataLoaders
    # -------------------------
    batch_size = 256
    train_loader = DataLoader(to_ds(X_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(to_ds(X_valid, y_valid), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Model + Training Setup
    # -------------------------
    input_features = X_train.shape[1]
    config = {
        "Model": {
            "input_len": input_features,
            "hidden_len": 128,
            "catg_num": 2,
            "aggregation": "average"
        }
    }
    model = DetectBERT(config, n_classes=2, input_size=input_features, hidden_size=128).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # -------------------------
    # Training Loop
    # -------------------------
    num_epochs = 70
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(data=Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total

        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in valid_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits, _ = model(data=Xb)
                loss = criterion(logits, yb)
                running_loss += loss.item() * Xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch}/{num_epochs} | Train Acc: {train_acc:.4%} | Val Acc: {val_acc:.4%}")

    # -------------------------
    # Final Evaluation Example
    # -------------------------
    if args.dataset == "lamda":
        results = []
        splits = {
            "iid": ["2013-12", "2014-08"],
            "near": ["2016", "2017"],
            "far": ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
        }
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
                    metrics = evaluate(model, X_te, y_te, device)
                    metrics.update({"dataset": "LAMDA", "split": split, "year": entry})
                    results.append(metrics)
        pd.DataFrame(results).to_csv("lamda_results.csv", index=False)

    elif args.dataset == "apigraph":
        test_dir = "/home/shared-datasets/gen_apigraph_drebin"
        results = []
        for year in range(2013, 2019):
            for month in range(1, 13):
                test_file = os.path.join(test_dir, f"{year}-{month:02d}_selected.npz")
                if not os.path.exists(test_file):
                    continue
                data = np.load(test_file, allow_pickle=True)
                X_te, y_te = data['X_train'], data['y_train']
                y_te = (y_te >= 1).astype(np.int64)
                metrics = evaluate(model, X_te, y_te, device)
                metrics.update({"dataset": "APIGraph", "year": year, "month": month})
                results.append(metrics)
        pd.DataFrame(results).to_csv("apigraph_results.csv", index=False)
