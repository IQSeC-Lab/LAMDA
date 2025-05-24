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
import json
import ast
import re
from PIL import Image

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

def draw_graph(draw_for_lamda, topk, data_filename, graph_filename):

    # Step 1: read data (SHAP top indices) starts 
    shap_top_indices_by_year = {}
    data_str = None
    # Replace 'data.txt' with the path to your file
    with open(data_filename, 'r') as file:
        data_str = file.read()

    # Replace `array([ ... ])` with just `[ ... ]`
    data_str = re.sub(r'array\((\[[^\]]+\])\)', r'\1', data_str)

    # Now safely evaluate
    shap_top_indices_by_year = ast.literal_eval(data_str)
    # read data (SHAP top indices) ends

    # Step 2: Compute Jaccard and Kendall distances for each run across years
    jaccard_runs = []
    kendall_runs = []
    runs_per_year = 5
    
    if draw_for_lamda:
        # our dataset 
        n_features = 4561
    else:
        # apigraph dataset 
        n_features = 1159
    
    top_k = topk
    plot_years = []

    for key in shap_top_indices_by_year.keys():

        if len(shap_top_indices_by_year[key]) == 5:
            if key not in plot_years:
                plot_years.append(key)

    for run in range(runs_per_year):
        jaccard_distances = []
        kendall_distances = []
        prev = None
        for year_month in sorted(shap_top_indices_by_year.keys()):
            if len(shap_top_indices_by_year[year_month]) == 5:
                print(f"year_month: {year_month} has data")
                # plot_years.append(shap_top_indices_by_year[year_month])
                current = shap_top_indices_by_year[year_month][run]

                if prev is not None:
                    intersection = len(set(prev) & set(current))
                    union = len(set(prev) | set(current))
                    # jaccard = 1 - intersection / union
                    if union == 0:
                        jaccard = 1  # Max distance
                    else:
                        jaccard = 1 - intersection / union

                    # Convert to rank vector
                    def to_rank_vector(top_k_indices, total_features):
                        ranks = np.ones(total_features) * (top_k + 1)
                        for rank, idx in enumerate(top_k_indices):
                            ranks[idx] = rank
                        return ranks

                    ranks_prev = to_rank_vector(prev, n_features)
                    ranks_current = to_rank_vector(current, n_features)
                    kendall_score, _ = kendalltau(ranks_prev, ranks_current)
                    kendall_dist = (1 - kendall_score)/2 if kendall_score is not None else 1

                    jaccard_distances.append(jaccard)
                    kendall_distances.append(kendall_dist)
                prev = current

        jaccard_runs.append(jaccard_distances)
        kendall_runs.append(kendall_distances)

    # Step 3: Convert to arrays and compute mean ± std
    jaccard_array = np.array(jaccard_runs)
    kendall_array = np.array(kendall_runs)

    jaccard_mean = jaccard_array.mean(axis=0)
    jaccard_std = jaccard_array.std(axis=0)
    kendall_mean = kendall_array.mean(axis=0)
    kendall_std = kendall_array.std(axis=0)

    # Compute SEM (Standard Error of the Mean)
    jaccard_sem = jaccard_std / np.sqrt(runs_per_year)
    kendall_sem = kendall_std / np.sqrt(runs_per_year)

    # Step 4: Plot with error bars
    plot_years = plot_years[1:]

    # Create a single plot
    plt.figure(figsize=(10, 7))

    # --- Plot 1: Jaccard Distance ---
    plt.plot(plot_years, jaccard_mean, marker='o', color='blue', label='Jaccard Distance (mean)')
    plt.fill_between(plot_years, jaccard_mean - jaccard_sem, jaccard_mean + jaccard_sem,
                    color='blue', alpha=0.4)

    # --- Plot 2: Kendall Distance ---
    plt.plot(plot_years, kendall_mean, marker='x', color='orange', label='Kendall Distance (mean)')
    plt.fill_between(plot_years, kendall_mean - kendall_sem, kendall_mean + kendall_sem,
                    color='orange', alpha=0.4)

    # Shared x and y labels
    plt.xlabel("Year-Month")
    plt.ylabel("Distance")

    # Shared x-ticks
    plt.xticks(ticks=range(0, len(plot_years), 3),
            labels=[plot_years[i] for i in range(0, len(plot_years), 3)], rotation=90)

    # Add legend
    plt.legend()

    # Finalize and save the plot
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(graph_filename, format="png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    
    # graph 1
    draw_for_lamda = False
    topk = 100 
    data_filename = "shap_top_indices_by_month_nyears_12_top_k_100_nrun_5_X_train_100_X_test_100_parallel_new_newsplit_apigraph.txt"
    graph_filename = "shap_feature_drift_combined_distances_100_features_new_newsplit_apigraph-new.png"
    draw_graph(draw_for_lamda, topk, data_filename, graph_filename)

    # graph 2
    draw_for_lamda = False
    topk = 1000  
    data_filename = "shap_top_indices_by_month_nyears_12_top_k_1000_nrun_5_X_train_100_X_test_100_parallel_new_newsplit_apigraph.txt"
    graph_filename = "shap_feature_drift_combined_distances_1000_features_new_newsplit_apigraph-new.png"
    draw_graph(draw_for_lamda, topk, data_filename, graph_filename)

    # graph 3
    draw_for_lamda = True
    topk = 100 
    data_filename = "shap_top_indices_by_month_nyears_12_top_k_100_nrun_5_X_train_100_X_test_100_parallel_new_newsplit.txt"
    graph_filename = "shap_feature_drift_combined_distances_100_features_new_newsplit_lambda-new.png"
    draw_graph(draw_for_lamda, topk, data_filename, graph_filename)

    # graph 4
    draw_for_lamda = True
    topk = 1000 
    data_filename = "shap_top_indices_by_month_nyears_12_top_k_1000_nrun_5_X_train_100_X_test_100_parallel_new_newsplit.txt"
    graph_filename = "shap_feature_drift_combined_distances_1000_features_new_newsplit_lamda-new.png"
    draw_graph(draw_for_lamda, topk, data_filename, graph_filename)