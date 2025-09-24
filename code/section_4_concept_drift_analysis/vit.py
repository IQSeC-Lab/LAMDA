import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize


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
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super().__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = feats ** 0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)
        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        h = self.head
        d = f // h

        q = self.q(x).view(b, n, h, d).transpose(1, 2)
        k = self.k(x).view(b, n, h, d).transpose(1, 2)
        v = self.v(x).view(b, n, h, d).transpose(1, 2)

        attn = torch.softmax((q @ k.transpose(-2, -1)) / self.sqrt_d, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, f)
        return self.o(self.dropout(out))


class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super().__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class EmberTransformer(nn.Module):
    def __init__(
        self,
        in_feats: int,
        num_classes: int,
        hidden: int = 384,
        mlp_hidden: int = 384*4,
        num_layers: int = 7,
        nhead: int = 8,
        dropout: float = 0.1,
        use_cls_token: bool = True
    ):
        super().__init__()
        self.use_cls = use_cls_token
        self.feat_emb = nn.Linear(in_feats, hidden)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if use_cls_token else None
        n_tokens = 1 + (1 if use_cls_token else 0)
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, hidden))
        self.encoder = nn.Sequential(*[
            TransformerEncoder(hidden, mlp_hidden, head=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, *args, **kwargs):
        x = args[0] if args else kwargs.get('data', kwargs.get('x', None))
        if x is None:
            raise KeyError("EmberTransformer.forward expects input tensor")
        B = x.size(0)
        tokens = self.feat_emb(x).unsqueeze(1)
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_emb
        out = self.encoder(tokens)
        feat = out[:, 0]
        logits = self.classifier(feat)
        return logits, feat


# ======================================================
# Evaluation
# ======================================================
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
# Args
# ======================================================
def get_args():
    parser = argparse.ArgumentParser(description="ViT/EmberTransformer Training and Evaluation")
    parser.add_argument("--dataset", type=str, default="lamda", choices=["lamda", "apigraph"],
                        help="Which dataset to use")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-2)
    return parser.parse_args()


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

    # Dataset loading
    if args.dataset == "LAMDA":
        train_dir = "/LAMDA_dataset/Baseline_npz_monthwise/train"
        test_dir  = "/LAMDA_dataset/Baseline_npz_monthwise/test"
        exclude = {"2013-12", "2014-08"}
        train_months = sorted([f[:7] for f in os.listdir(train_dir)
                               if f.endswith("_X_train.npz") and (f.startswith("2013") or f.startswith("2014"))
                               and f[:7] not in exclude])
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
        test_dir = "/home/shared-datasets/gen_apigraph_drebin"

    # Split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=args.seed, stratify=y
    )

    # Dataloaders
    train_loader = DataLoader(to_ds(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(to_ds(X_valid, y_valid), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    input_features = X_train.shape[1]
    model = EmberTransformer(
        in_feats=input_features,
        num_classes=2,
        hidden=512,
        mlp_hidden=384*4,
        num_layers=10,
        nhead=128,
        dropout=0.1,
        use_cls_token=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Training Loop
    for epoch in range(1, args.epochs+1):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
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
        correct, total, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for Xb, yb in valid_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits, _ = model(data=Xb)
                loss = criterion(logits, yb)
                running_loss += loss.item() * Xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch}/{args.epochs} | Train Acc: {train_acc:.4%} | Val Acc: {val_acc:.4%}")

    # Evaluation Example
    results = []
    if args.dataset == "lamda":
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
        pd.DataFrame(results).to_csv("lamda_vit_results.csv", index=False)

    elif args.dataset == "apigraph":
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
        pd.DataFrame(results).to_csv("apigraph_vit_results.csv", index=False)
