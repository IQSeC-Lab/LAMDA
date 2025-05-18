import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, average_precision_score
)
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import shap
import lime.lime_tabular
from captum.attr import IntegratedGradients
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns
import time
import concurrent.futures
import multiprocessing as mp
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class XGBoostWrapper:
    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X):
        dmatrix = xgb.DMatrix(X)
        preds = self.booster.predict(dmatrix)
        return np.vstack([1 - preds, preds]).T

class ChenEncoderMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(ChenEncoderMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 384), nn.ReLU(),
            nn.Linear(384, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, num_classes)  # logits for both classes
        )

    def forward(self, x, return_prob=False):
        x = self.encoder(x)
        logits = self.classifier(x)
        if return_prob:
            return F.softmax(logits, dim=1)
        return logits

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        fpr = fnr = 0
    return {
        "accuracy": acc, "precision": prec, "recall": recall, "f1_score": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "fpr": fpr, "fnr": fnr
    }

def train_lightgbm(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)
    params = {
        "boosting_type": "gbdt", "objective": "binary", "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05, "num_leaves": 2048, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "verbosity": -1
    }
    return lgb.train(params, dtrain, num_boost_round=5000, valid_sets=[dtrain, dval],
                     callbacks=[lgb.early_stopping(100)])

def train_xgboost(X, y):
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train({"max_depth": 12, "eta": 0.05, "eval_metric": "error"},
                        dtrain, num_boost_round=5000)
    return XGBoostWrapper(booster)

def train_svm(X, y):
    model = CalibratedClassifierCV(LinearSVC(max_iter=10000))
    model.fit(X, y)
    return model

def train_chen_mlp(X, y, epochs=20):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = ChenEncoderMLP(X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # loss = criterion(model(xb).squeeze(), yb)
            logits = model(xb)  # now returns [batch_size, 2]
            loss = criterion(logits, yb)  # no need for squeeze
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, X, y, model_type):
    if model_type == "mlp":
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
    elif model_type == "lightgbm":
        preds = model.predict(X)
    else:
        preds = model.predict_proba(X)[:, 1]
    return compute_metrics(y, preds)

def load_data(years, dir):
    X, y = [], []
    for year in years:
        X.append(load_npz(f"{dir}/{year}_X.npz").toarray())
        y.append(np.load(f"{dir}/{year}_meta.npz")["y"])
    return np.vstack(X), np.concatenate(y)

def shap_predict(model, x_numpy):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        print("probs: ", probs)
        return probs.detach().cpu().numpy()
    
def make_shap_predict(model):
    def shap_predict(x_numpy):
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
            logits = model(x_tensor, return_prob=True)  # shape: [batch_size, 2]
            return logits.detach().cpu().numpy()
    return shap_predict
    

def normalize(arr):
    return arr / np.sum(arr) if np.sum(arr) > 0 else arr

# Worker function for 1 run
def run_single_shap_experiment(base_url, year, month, run_id, data_dir, top_k, n_samples):

    # Load train data. features (X) yearwise
    data = np.load(f"{base_url}/train_{str(year)+'-'+month}_selected.npz")
    X_train = data['X_train']
    y_train = data['y_train']

    print("X_train.shape: ", X_train.shape)

    # load test data 
    test_data = np.load(f"{base_url}/test_{str(year)+'-'+month}_selected.npz")
    # X_test = np.load(f"/home/shared-datasets/Feature_extraction/npz_monthwise_Final_test/{str(year)+'-'+month}_X_test.npz")
    # X_test = csr_matrix((X_test['data'], X_test['indices'], X_test['indptr']), shape=X_test['shape']).toarray()
    X_test = test_data['X_test']
    y_test = test_data['y_test']

    print("X_test.shape: ", X_test.shape)
    print("y_train.shape: ", y_train.shape)

    top_indices = []
    top_importances = []

    # some months do not have enough data 
    # if X_train.shape[0] > n_samples and X_test.shape[0] > n_samples:
    model = train_chen_mlp(X_train, y_train)
    shap_predict_fn = make_shap_predict(model)

    
    # KernelExplainer
    explainer = shap.KernelExplainer(shap_predict_fn, X_train[:n_samples]) # X_train[:n_samples] background samples
    shap_values = explainer.shap_values(X_test[:n_samples], nsamples=n_samples) # X_test[:n_samples] explanation samples

    shap_importance = np.mean([np.abs(s[:, 1]) for s in shap_values], axis=0)
    top_indices = np.argsort(shap_importance)[-top_k:][::-1]
    top_importances = shap_importance[top_indices]

    return (year, month, run_id, top_indices, top_importances)

def run_explanation_parallely():
    # Constants
    top_k = 10
    runs_per_year = 3
    data_dir = '/home/shared-datasets/Feature_extraction/npz_monthwise_Final'
    train_years = [2013, 2014, 2016]
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    n_features = None
    n_samples = 100 # try passing 500, 1000 

    # for monthwise analysis 
    base_url = '/home/shared-datasets/Feature_extraction/npz_monthwise_Final'

    # List all files in the directory
    file_names = os.listdir(base_url)

    # Filter files if needed (e.g., only `.npz` files)
    X_npz_files = [file for file in file_names if file.endswith('X.npz')]
    meta_npz_files = [file for file in file_names if file.endswith('meta.npz')]

    print("X_npz_files")
    print(len(X_npz_files))

    print("meta_npz_files")
    print(len(meta_npz_files))

    
    shap_top_indices_by_year_month = {f"{year}-{month}": [] for year in train_years for month in months}
    shape_importance_by_year_month = {f"{year}-{month}": [] for year in train_years for month in months}

    start_time = time.time()

    # Run in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for year in train_years:
            for month in months:
                if str(year)+'-'+month+'_X.npz' not in X_npz_files:
                    print(f"file {str(year)+'-'+month} not in X files list")
                else:
                    print("year-month: ", (str(year)+'-'+month))
                    for run_id in range(runs_per_year):
                        futures.append(executor.submit(run_single_shap_experiment, base_url, year, month, run_id, data_dir, top_k, n_samples))

        for future in concurrent.futures.as_completed(futures):
            year, month, run_id, top_indices, top_importances = future.result()
            shap_top_indices_by_year_month[str(year)+'-'+month].append(top_indices)
            shape_importance_by_year_month[str(year)+'-'+month].append(top_importances)

    
    output_file = "shap_top_indices_by_month_nyears_3_top_k_10_nrun_3_X_train_100_X_test_100_parallel.txt"  # Specify the output file name

    # Write shap_top_indices_by_year to the file
    with open(output_file, "w") as file:
        file.write(str(shap_top_indices_by_year_month))  # Convert to string if necessary

    print(f"shap_top_indices_by_year_month has been written to {output_file}")


    output_file = "shape_importance_by_month_nyears_3_top_k_10_nrun_3_X_train_100_X_test_100_parallel.txt"  # Specify the output file name

    # Write shap_top_indices_by_year to the file
    with open(output_file, "w") as file:
        file.write(str(shape_importance_by_year_month))  # Convert to string if necessary

    print(f"shape_importance_by_year has been written to {output_file}")
    end_time = time.time()

    print(f"paralel execution took time: {end_time-start_time}s")



def run_monthly_shap_experiments(year, months, runs_per_year, base_url, data_dir, top_k, n_samples, X_npz_files):
    
    print("run_monthly_shap_experiments called...")
    shap_top_indices_by_year_month = defaultdict(list)
    shape_importance_by_year_month = defaultdict(list)

    for month in months:
        ym_key = f"{year}-{month}"
        if f"train_{ym_key}_selected.npz" not in X_npz_files or f"test_{ym_key}_selected.npz" not in X_npz_files:
            print(f"File {ym_key} not in data directory")
            continue

        for run_id in range(runs_per_year):
            top_indices, top_importances = run_single_shap_experiment(
                base_url, year, month, run_id, data_dir, top_k, n_samples
            )[3:]  # skipping year, month, run_id
            shap_top_indices_by_year_month[ym_key].append(top_indices)
            shape_importance_by_year_month[ym_key].append(top_importances)

    return shap_top_indices_by_year_month, shape_importance_by_year_month


def run_explanation_parallely_monthly(topk):
    print("run_explanation_parallely_monthly called...")
    # Constants
    top_k = topk
    runs_per_year = 5
    
    # our data dir 
    data_dir = '/home/malam10/projects/feature-evolution-android-malware/apigraph_dataset'
    base_url = '/home/malam10/projects/feature-evolution-android-malware/apigraph_dataset'

    # apigraph data dir 
    train_years = [2013, 2014, 2016, 2017, 2018]
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    n_features = None
    n_samples = 100 # try passing 500, 1000 

    # List all files in the directory
    file_names = os.listdir(base_url)
    print("file_names")
    print(file_names)

    # Filter files if needed (e.g., only `.npz` files)
    X_npz_files = [file for file in file_names if file.endswith('_selected.npz') and (file.startswith('train_') or file.startswith('test_'))]
    
    shap_top_indices_by_year_month = {f"{year}-{month}": [] for year in train_years for month in months}
    shape_importance_by_year_month = {f"{year}-{month}": [] for year in train_years for month in months}

    start_time = time.time()

    # Run in parallel, one process per year
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = {
            executor.submit(
                run_monthly_shap_experiments,
                year, months, runs_per_year, base_url, data_dir, top_k, n_samples, X_npz_files
            ): year for year in train_years
        }

        for future in as_completed(futures):
            year = futures[future]
            year_indices, year_importances = future.result()

            print("year_indices: ", year_indices)

            for ym_key in year_indices:
                shap_top_indices_by_year_month[ym_key].extend(year_indices[ym_key])
                shape_importance_by_year_month[ym_key].extend(year_importances[ym_key])

    
    output_file = "top_shap_indices_{top_k}_apigraph.txt"
    

    # Write shap_top_indices_by_year to the file
    with open(output_file, "w") as file:
        file.write(str(shap_top_indices_by_year_month))  # Convert to string if necessary

    print(f"shap_top_indices_by_year_month has been written to {output_file}")


    output_file = "top_shap_importance_{top_k}_apigraph.txt"

    # Write shap_top_indices_by_year to the file
    with open(output_file, "w") as file:
        file.write(str(shape_importance_by_year_month))  # Convert to string if necessary

    print(f"shape_importance_by_year_month has been written to {output_file}")
    end_time = time.time()

    print(f"paralel execution took time: {end_time-start_time}s")


def check_data_stats():
    # for monthwise analysis 
    base_url = '/home/shared-datasets/Feature_extraction/npz_monthwise_Final'

    # List all files in the directory
    file_names = os.listdir(base_url)

    # Filter files if needed (e.g., only `.npz` files)
    X_npz_files = [file for file in file_names if file.endswith('X.npz')]
    meta_npz_files = [file for file in file_names if file.endswith('meta.npz')]

    print("X_npz_files")
    print(len(X_npz_files))

    print("meta_npz_files")
    print(len(meta_npz_files))

    for filename in X_npz_files:
        print("filename: ", filename)

    print("meta_npz_files")
    print(len(meta_npz_files))

    train_years = [2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    d = {}
    for year in train_years:
        for month in months:
            # Load features (X) yearwise
            if f"{str(year)+'-'+month}_X.npz" in X_npz_files:
                

                X = np.load(f"{base_url}/{str(year)+'-'+month}_X.npz")
                X = csr_matrix((X['data'], X['indices'], X['indptr']), shape=X['shape']).toarray()

                # print("X.shape: ", X.shape)
                n_features = X.shape[1]

                # Load labels (y) yearwise
                # print(f"print url - {base_url}/{str(year)+'-'+month}_meta.npz")
                meta = np.load(f"{base_url}/{str(year)+'-'+month}_meta.npz")
                print(meta.keys())
                y = meta['label']

                random_state = random.randint(0, 1000)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
                # print("X_train.shape: ", X_train.shape)
                # print("X_test.shape: ", X_test.shape)

                d[f"{str(year)+'-'+month}"] = (X_train.shape, X_test.shape)
    print(d)
    import matplotlib.pyplot as plt

    # Your input data
    data = d

    # Extracting data
    months = list(data.keys())
    x_train_counts = [v[0][0] for v in data.values()]
    x_test_counts = [v[1][0] for v in data.values()]
    total_counts = [train + test for train, test in zip(x_train_counts, x_test_counts)]

    # Plotting
    plt.figure(figsize=(18, 6))
    bar_width = 0.25
    r1 = range(len(months))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, x_train_counts, width=bar_width, label='X_train')
    plt.bar(r2, x_test_counts, width=bar_width, label='X_test')
    plt.bar(r3, total_counts, width=bar_width, label='Total')

    # X-axis formatting
    plt.xticks([r + bar_width for r in range(len(months))], months, rotation=90, fontsize=10)

    # Labels and legend
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("Sample Count", fontsize=14)
    plt.title("Monthly Sample Distribution in X_train, X_test, and Total", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig("sample_counts_bars.png", dpi=300)
    plt.close()

    # Extract X_train and X_test sample counts
    x_train_counts = [v[0][0] for v in data.values()]
    x_test_counts = [v[1][0] for v in data.values()]

    # Compute min and max
    min_train = min(x_train_counts)
    max_train = max(x_train_counts)
    min_test = min(x_test_counts)
    max_test = max(x_test_counts)

    print(f"X_train - Min: {min_train}, Max: {max_train}")
    print(f"X_test  - Min: {min_test}, Max: {max_test}")


def check_n_features():
    # for monthwise analysis 
    X_train = np.load(f"/home/shared-datasets/gen_apigraph_drebin/2013-06_selected.npz")

    # Check the attributes (keys) in the .npz file
    print("Attributes in the dataset:", X_train.keys())
    print(X_train['X_train'].shape)
    print(X_train['y_train'].shape)

def apigraph_data_train_test_split():
    data_dir = '/home/shared-datasets/gen_apigraph_drebin'
    files = os.listdir(data_dir)

    # Ensure the directory exists
    output_dir = "apigraph_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    for file in files:
        if len(file) == len("2013-01_selected.npz"):
            # print(file)
            data = np.load(f"/home/shared-datasets/gen_apigraph_drebin/{file}")
            X = data['X_train']
            y = data['y_train']
            print("X.shape: ", X.shape)
            print("y.shape: ", y.shape)

            # Convert multi-label y to binary labels
            y_binary = (y != 0).astype(int)

            # Count the occurrences of 0 and 1
            counts = np.bincount(y_binary)

            print("y_binary.len: ", len(y_binary))

            # Print the counts
            # print("Count of 0s:", counts[0])
            # print("Count of 1s:", counts[1])

            # X_train, y_train, X_test, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
            X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

            assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
            assert y_train.shape[0] + y_test.shape[0] == y_binary.shape[0]


            # Flatten y_train
            y_train = np.array(y_train).ravel()

            print("X_train.shape: ", X_train.shape)
            print("y_train.shape: ", y_train.shape)



            print("X_test.shape: ", X_test.shape)
            print("y_test.shape: ", y_test.shape)

            print("Train label distribution:", np.bincount(y_train))
            print("Test label distribution:", np.bincount(y_test))
            dataset_dir = '/home/malam10/projects/feature-evolution-android-malware/apigraph_dataset'
            

            # Save train data to npz
            train_file = f"{dataset_dir}/train_{file}"
            np.savez(train_file, X_train=X_train, y_train=y_train)
            print(f"Train data saved to {train_file}")

            # Save test data to npz
            test_file = f"{dataset_dir}/test_{file}"
            np.savez(test_file, X_test=X_test, y_test=y_test)
            print(f"Test data saved to {test_file}")


def check_apigraph_dataset():
    base_url = '/home/malam10/projects/feature-evolution-android-malware/apigraph_dataset'

    
    files = os.listdir(base_url)
    # print(files)
    for file in files:
        print("filename: ", file)
        data = np.load(f"/home/malam10/projects/feature-evolution-android-malware/apigraph_dataset/{file}")
        if file.startswith("train"):
            X = data['X_train']
            y = data['y_train']
        # else:
        #     X = data['X_test']
        #     y = data['y_test']

        print("X.shape: ", X.shape)
        print("y.shape: ", y.shape)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # main()
    # what running combination I have to try 
    # yearly, monthly, top k 10, top k 50, top 100, 5 runs 
    
    topk = 100
    run_explanation_parallely_monthly(topk)

    topk = 1000
    run_explanation_parallely_monthly(topk)