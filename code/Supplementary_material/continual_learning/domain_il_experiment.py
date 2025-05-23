import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


# Years for domain incremental learning
DOMAIN_YEARS: List[int] = [2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
RNG = np.random.default_rng(42)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain-Incremental Learning for Malware Detection")
    parser.add_argument("--strategy", choices=["naive", "cumulative", "expreplay"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--memory_size", type=int, default=200)
    parser.add_argument("--log_dir", type=Path, default=Path("logs_domain"))
    parser.add_argument("--gpu_id", type=int, default=2)
    return parser.parse_args()

def _build_data_path(year: int, split: str) -> Path:
    return Path("Domain-IL-Baseline") / str(year) / f"{year}_{split}.parquet"

def load_dataset(year: int, split: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    path = _build_data_path(year, split)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    x = torch.tensor(df[feat_cols].values, dtype=torch.float32, device=device)
    y = torch.tensor(df["label"].values, dtype=torch.float32, device=device).unsqueeze(1)
    logging.info(f"Loaded {len(df)} samples from {split} split for year {year}")
    return x, y

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

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []
        self.n_seen = 0
        self.year_tags = []

    def add_batch(self, x: torch.Tensor, y: torch.Tensor, year: int):
        for xi, yi in zip(x, y):
            self.n_seen += 1
            if len(self.data) < self.capacity:
                self.data.append((xi.cpu(), yi.cpu()))
                self.year_tags.append(year)
            else:
                j = int(RNG.integers(0, self.n_seen))
                if j < self.capacity:
                    self.data[j] = (xi.cpu(), yi.cpu())
                    self.year_tags[j] = year

    def sample(self):
        x_list, y_list = zip(*self.data)
        return torch.stack(x_list), torch.stack(y_list)

    def years_in_buffer(self) -> List[int]:
        return sorted(set(self.year_tags))

    def __len__(self):
        return len(self.data)

def create_loader(x, y, batch_size, shuffle):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)

def train_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optim.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

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
    preds = (logits.squeeze(1) >= 0.5).long()
    labels = labels.long().squeeze(1)
    return preds.numpy(), labels.numpy()

def metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds,  zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

def save_results_to_csv(log: Dict, combined_path: Path):
    records = []
    for r in log["test_results"]:
        r_copy = r.copy()
        r_copy["experience"] = log["experience"]
        r_copy["strategy"] = log["strategy"]
        records.append(r_copy)
    df = pd.DataFrame(records)
    if combined_path.exists():
        df.to_csv(combined_path, mode='a', index=False, header=False)
    else:
        df.to_csv(combined_path, index=False)
    logging.info(f"Appended results to {combined_path}")

def main():
    args = get_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = args.log_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting experiment with strategy: %s", args.strategy)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    buffer = ReplayBuffer(args.memory_size) if args.strategy == "expreplay" else None
    model = None

    for idx, year in enumerate(DOMAIN_YEARS):
        x_curr, y_curr = load_dataset(year, "train", device)
        logging.info(f"Training samples for year {year}: {x_curr.size(0)}")

        if model is None:
            model = ChenEncoderMLP(x_curr.size(1)).to(device)

        # Explicit handling for each strategy
        if args.strategy == "naive":
            x_train, y_train = x_curr, y_curr

        elif args.strategy == "cumulative":
            x_train_list, y_train_list = [x_curr], [y_curr]
            for y in DOMAIN_YEARS[:idx]:
                x_p, y_p = load_dataset(y, "train", device)
                x_train_list.append(x_p)
                y_train_list.append(y_p)
            x_train = torch.cat(x_train_list)
            y_train = torch.cat(y_train_list)

        elif args.strategy == "expreplay":
            x_train, y_train = x_curr, y_curr
            if len(buffer):
                x_buf, y_buf = buffer.sample()
                x_train = torch.cat([x_train, x_buf.to(device)])
                y_train = torch.cat([y_train, y_buf.to(device)])

        else:
            raise ValueError(f"Unsupported strategy: {args.strategy}")

        loader = create_loader(x_train, y_train, args.batch_size, True)
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)

        for _ in range(args.epochs):
            train_epoch(model, loader, optim, criterion, device)

        log = {
            "experience": year,
            "trained_on_years": ([year] if args.strategy == "naive" else DOMAIN_YEARS[: idx + 1]),
            "strategy": args.strategy,
            "task_type": "domain",
            "test_results": [],
        }

        # backward evaluation
        for y in DOMAIN_YEARS[: idx + 1]:
            x_test, y_test = load_dataset(y, "test", device)
            logging.info(f"Testing (backward) for year {y}: {x_test.size(0)}")
            test_loader = create_loader(x_test, y_test, args.batch_size, False)
            preds, labels = predict(model, test_loader, device)
            log["test_results"].append({"test_year": y, "mode": "backward", **metrics(preds, labels)})
            if y == year:
                new_preds_current = preds

        # forward evaluation
        for y in DOMAIN_YEARS[idx + 1:]:
            x_test, y_test = load_dataset(y, "test", device)
            logging.info(f"Testing (forward) for year {y}: {x_test.size(0)}")
            test_loader = create_loader(x_test, y_test, args.batch_size, False)
            preds, labels = predict(model, test_loader, device)
            log["test_results"].append({"test_year": y, "mode": "forward", **metrics(preds, labels)})

        if buffer is not None:
            buffer.add_batch(x_curr.detach().cpu(), y_curr.detach().cpu(), year)

        ckpt_path = args.log_dir / f"model_{year}.pth"
        torch.save(model.state_dict(), ckpt_path)
        (args.log_dir / f"log_{year}.json").write_text(json.dumps(log, indent=2))
        save_results_to_csv(log, args.log_dir / "results_all.csv")

        logging.info(f"Finished training for year {year}")
        logging.info(json.dumps(log, indent=2))

if __name__ == "__main__":
    main()
