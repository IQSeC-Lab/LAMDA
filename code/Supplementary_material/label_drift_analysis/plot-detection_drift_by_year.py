import pandas as pd
import matplotlib.pyplot as plt

# === Global Plot Settings ===
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

# === Load Data ===
df = pd.read_csv("vt_detections.csv")  

# === Categorize for Plot and Logging ===
def categorize(row):
    if row['vt_json_count'] == 0 and row['vt_androzoo_count'] > 0:
        return 'json_0_androzoo_gt0'
    elif row['vt_json_count'] == row['vt_androzoo_count']:
        return 'equal'
    elif row['vt_json_count'] > row['vt_androzoo_count']:
        return 'json_gt_androzoo'
    else:
        return 'json_lt_androzoo'

df['category'] = df.apply(categorize, axis=1)

# === Maps for Plot Styling ===
color_map = {
    'equal': 'blue',
    'json_gt_androzoo': 'red',
    'json_lt_androzoo': 'green',
    'json_0_androzoo_gt0': 'yellow'
}
marker_map = {k: 'o' for k in color_map}
label_map = {
    'equal': r'$\mathrm{D}_{\mathrm{Unchanged}}$',
    'json_gt_androzoo': r'$\mathrm{D}_{\mathrm{Improved}}$',
    'json_lt_androzoo': r'$\mathrm{D}_{\mathrm{Weakened}}$',
    'json_0_androzoo_gt0': r'$\mathrm{B}_{\mathrm{C}}$'
}

# === Logging Summary ===
print("=== Detection Drift Summary by Year ===")
years = sorted(df['year'].unique())
for year in years:
    year_data = df[df['year'] == year]
    improved = (year_data['category'] == 'json_gt_androzoo').sum()
    weakened = (year_data['category'] == 'json_lt_androzoo').sum()
    unchanged = (year_data['category'] == 'equal').sum()
    vt_zero = (year_data['category'] == 'json_0_androzoo_gt0').sum()

    print(f"\nYear: {year}")
    print(f"  Improved Detection (VT > AndroZoo): {improved}")
    print(f"  Weakened Detection (VT < AndroZoo): {weakened}")
    print(f"  Unchanged Detection (VT == AndroZoo): {unchanged}")
    print(f"  Became Zero in VT (VT=0, AndroZoo>0): {vt_zero}")

# === Plotting Setup ===
cols = 3
rows = (len(years) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 6 * rows), sharex=True, sharey=True)

# Plot data year-wise
for i, year in enumerate(years):
    ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
    subset = df[df['year'] == year]
    max_val = max(df['vt_json_count'].max(), df['vt_androzoo_count'].max()) + 2
    ax.plot([0, max_val], [0, max_val], linestyle='--', color='silver', linewidth=1)

    for category in df['category'].unique():
        cat_data = subset[subset['category'] == category]
        ax.scatter(
            cat_data['vt_androzoo_count'],
            cat_data['vt_json_count'],
            c=color_map[category],
            marker=marker_map[category],
            label=label_map[category],
            edgecolors='k',
            alpha=0.7
        )

    ax.set_title(f"{year}")
    ax.grid(True)

# Global axis labels
fig.supxlabel("Virustotal Current AV Detection Count", fontsize=16, weight='bold')
fig.supylabel("Androzoo AV Detection Count", fontsize=16, weight='bold')

# Global legend slightly higher to avoid overlap
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
fig.legend(
    unique.values(), unique.keys(),
    loc='upper center',
    bbox_to_anchor=(0.5, 1),  # Moved up
    ncol=len(label_map),
    title='Detection Change',
    title_fontsize=12
)


# plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.07, hspace=0.4, wspace=0.3)
# plt.show()


plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.07, hspace=0.4, wspace=0.3)

plt.savefig("detection_drift_by_year.png", dpi=600, bbox_inches='tight')  # You can also use .pdf

plt.show()
