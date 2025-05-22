import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

DOMAIN_YEARS: List[int] = [2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Class-Incremental Learning for Malware Detection")
    parser.add_argument("--strategy", choices=["naive", "cumulative", "expreplay"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--memory_size", type=int, default=200)
    parser.add_argument("--log_dir", type=Path, default=Path("logs_class"))
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


def _build_data_path(year: int, split: str) -> Path:
    return Path("Class-IL-Dataset") / f"{year}_{split}.parquet"


def load_dataset(year: int, split: str, device: torch.device) -> Tuple[torch.Tensor, List[str], List[str]]:
    path = _build_data_path(year, split)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    x = torch.tensor(df[feat_cols].values, dtype=torch.float32, device=device)
    families = df["family"].tolist()
    return x, families, df["family"].unique().tolist()


class ChenEncoderMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 384), nn.ReLU(),
            nn.Linear(384, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.classifier = nn.Linear(128, output_dim)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


class ReplayBuffer:
    def __init__(self, samples_per_experience: int):
        self.capacity_per_experience = samples_per_experience
        self.data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    def add_batch(self, year: int, x: torch.Tensor, y: torch.Tensor):
        x, y = x.cpu(), y.cpu()
        indices = np.random.choice(len(x), min(self.capacity_per_experience, len(x)), replace=False)
        self.data[year] = [(x[i], y[i]) for i in indices]

    def sample(self):
        x_list, y_list = [], []
        for year_samples in self.data.values():
            for xi, yi in year_samples:
                x_list.append(xi)
                y_list.append(yi)
        return torch.stack(x_list), torch.tensor([yi.item() for yi in y_list])

    def __len__(self):
        return sum(len(v) for v in self.data.values())


def create_loader(x, y, batch_size, shuffle):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def predict(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            logits_all.append(logits.cpu())
            labels_all.append(yb.cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    preds = torch.argmax(logits, dim=1)
    return preds.numpy(), labels.numpy()


def metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def save_results_to_csv(log: Dict, combined_path: Path):
    records = []
    for r in log["test_results"]:
        r_copy = r.copy()
        r_copy.setdefault("mode", "unknown")
        r_copy["experience"] = log.get("experience", "unknown")
        r_copy["strategy"] = log.get("strategy", "unknown")
        records.append(r_copy)

    df = pd.DataFrame(records)
    expected_columns = ["experience", "strategy", "test_year", "mode", "accuracy", "precision", "recall", "f1"]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected_columns]

    if combined_path.exists():
        df.to_csv(combined_path, mode='a', index=False, header=False)
    else:
        df.to_csv(combined_path, index=False)
    logging.info(f"Appended results to {combined_path}")


def main():
    args = get_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[
            logging.FileHandler(args.log_dir / "experiment.log"),
            logging.StreamHandler()
        ]
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    buffer = ReplayBuffer(args.memory_size) if args.strategy == "expreplay" else None
    global_families: Set[str] = set()
    label_encoder = LabelEncoder()
    all_family_names = []

    for year in DOMAIN_YEARS:
        _, _, fams = load_dataset(year, "train", device)
        all_family_names.extend(fams)
    label_encoder.fit(sorted(set(all_family_names)))

    model = None
    for idx, year in enumerate(DOMAIN_YEARS):
        x_train, fams_train, new_fams = load_dataset(year, "train", device)
        y_train = torch.tensor(label_encoder.transform(fams_train), dtype=torch.long, device=device)
        local_families = set(new_fams)
        new_families = local_families - global_families
        global_families.update(local_families)

        logging.info(f"[Year {year}] New families introduced: {len(new_families)} → {sorted(new_families)}")
        logging.info(f"[Year {year}] Total families in this experience: {len(local_families)}")

        if model is None:
            model = ChenEncoderMLP(x_train.size(1), len(label_encoder.classes_)).to(device)

        if args.strategy == "naive":
            pass  # use current experience only
        elif args.strategy == "cumulative":
            x_train_all, y_train_all = [x_train], [y_train]
            for y in DOMAIN_YEARS[:idx]:
                x_old, fams_old, _ = load_dataset(y, "train", device)
                y_old = torch.tensor(label_encoder.transform(fams_old), dtype=torch.long, device=device)
                x_train_all.append(x_old)
                y_train_all.append(y_old)
            x_train = torch.cat(x_train_all)
            y_train = torch.cat(y_train_all)
        elif args.strategy == "expreplay" and len(buffer):
            x_buf, y_buf = buffer.sample()
            x_train = torch.cat([x_train, x_buf.to(device)])
            y_train = torch.cat([y_train, y_buf.to(device)])

        loader = create_loader(x_train, y_train, args.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
        criterion = nn.CrossEntropyLoss()

        for _ in range(args.epochs):
            train_epoch(model, loader, optimizer, criterion, device)

        log = {
            "experience": year,
            "strategy": args.strategy,
            "test_results": [],
            "new_families": sorted(new_families),
            "local_family_count": len(local_families),
            "global_family_count": len(global_families),
        }

        for y in DOMAIN_YEARS:
            if y <= year:
                mode = "backward"
            else:
                mode = "forward"

            try:
                x_test, fams_test, _ = load_dataset(y, "test", device)
                y_test = torch.tensor(label_encoder.transform(fams_test), dtype=torch.long, device=device)
                logging.info(f"Testing on year {y} ({mode}) — samples: {x_test.size(0)}")
                test_loader = create_loader(x_test, y_test, args.batch_size, shuffle=False)
                preds, labels = predict(model, test_loader, device)
                result = {"test_year": y, "mode": mode, **metrics(preds, labels)}
                log["test_results"].append(result)
            except FileNotFoundError:
                logging.warning(f"Test set for year {y} not found. Skipping.")

        torch.save(model.state_dict(), args.log_dir / f"model_{year}.pth")
        (args.log_dir / f"log_{year}.json").write_text(json.dumps(log, indent=2))
        save_results_to_csv(log, args.log_dir / "results_all.csv")

        if buffer:
            buffer.add_batch(year, x_train.detach(), y_train.detach())

        logging.info(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
