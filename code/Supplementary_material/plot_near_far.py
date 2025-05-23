import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import glob
import argparse
import os

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

MODEL_LIST = ["LightGBM", "MLP", "SVM", "XGBoost"]
MODEL_COLORS = {
    "LightGBM": "#1f77b4",
    "MLP": "teal",
    "SVM": "#ff7f0e",
    "XGBoost": "#2ca02c"
}

def load_and_aggregate(csv_files):
    all_dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(all_dfs, ignore_index=True)
    df["model"] = df["model"].str.upper()
    df["year"] = df["year"].astype(str).str[:4]
    df = df[df["year"] != "2015"]

    # Only keep models of interest
    df = df[df["model"].isin([m.upper() for m in MODEL_LIST])]

    grouped = df.groupby(["model", "year"]).agg({
        "f1_score": ["mean", "std"]
    }).reset_index()

    grouped.columns = ["model", "year", "f1_mean", "f1_std"]
    return grouped.sort_values(["model", "year"])

def plot_f1_for_all_models(df, output_path="f1_by_model_with_errorbands.png"):
    years = sorted(df["year"].unique())
    x = list(range(len(years)))

    fig, ax = plt.subplots(figsize=(10, 7))

    for model in MODEL_LIST:
        model_upper = model.upper()
        sub = df[df["model"] == model_upper]
        # Align to all years (fill missing with NaN)
        sub = sub.set_index("year").reindex(years)
        f1_mean = sub["f1_mean"].values
        f1_std = sub["f1_std"].values
        color = MODEL_COLORS.get(model, None)
        ax.plot(x, f1_mean, label=model, color=color)
        ax.fill_between(x, f1_mean - f1_std, f1_mean + f1_std, color=color, alpha=0.18)

    # Annotate region bars
    region_map = {
        "IID": ["2013", "2014"],
        "NEAR": ["2016", "2017"],
        "FAR": ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    }
    region_colors = {"IID": "green", "NEAR": "orange", "FAR": "purple"}
    ymin = -0.02

    for label, years_range in region_map.items():
        x_pos = [i for i, y in enumerate(years) if y in years_range]
        if x_pos:
            start, end = min(x_pos), max(x_pos)
            ax.plot([start, end], [ymin, ymin], color=region_colors[label], linewidth=3, solid_capstyle="butt")
            ax.text((start + end)/2, ymin - 0.03, label, ha='center', va='top',
                    color=region_colors[label], fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylim(-0.1, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("Test Year")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.grid(
        True,
        which='both',  
        linestyle='--',
        linewidth=0.5,
        color='gray',
        alpha=0.3 
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot F1-score for all models from CSVs in a directory.")
    parser.add_argument('--dir', type=str, required=True, help='Directory containing CSV files')
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.dir, "*_anoshift_run*.csv")))
    if not csv_files:
        print(f"No CSV files found in {args.dir}")
        return

    df_summary = load_and_aggregate(csv_files)
    plot_f1_for_all_models(df_summary)


if __name__ == "__main__":
    main()

