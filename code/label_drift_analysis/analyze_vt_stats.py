import os
import sys
import pandas as pd
from collections import defaultdict

def analyze_vt_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if 'year' not in df.columns:
        print("input CSV must contain a 'year' column.")
        sys.exit(1)

    all_stats = []
    overall = defaultdict(int)

    for year, group in df.groupby('year'):
        total = len(group)
        now_benign = group[group['vt_json_count'] == 0]
        improved = group[group['vt_json_count'] > group['vt_androzoo_count']]
        weakened = group[(group['vt_json_count'] < group['vt_androzoo_count']) & (group['vt_json_count'] > 0)]
        unchanged = group[group['vt_json_count'] == group['vt_androzoo_count']]
        significant_drop = group[((group['vt_androzoo_count'] - group['vt_json_count']) / group['vt_androzoo_count']) > 0.5]
        significantly_increased = group[((group['vt_json_count'] - group['vt_androzoo_count']) / group['vt_androzoo_count']) > 0.5]

        year_stats = {
            'Year': year,
            'Total Samples': total,
            'Now Benign (vt_json==0)': len(now_benign),
            '% Now Benign': round(len(now_benign) / total * 100, 2),
            'Improved Detection': len(improved),
            'Weakened Detection': len(weakened),
            'Unchanged Detection': len(unchanged),
            'Significant Drop (>50%)': len(significant_drop),
            'Significantly Increased (>50%)': len(significantly_increased)
        }

        all_stats.append(year_stats)

        overall['Total Samples'] += total
        overall['Now Benign'] += len(now_benign)
        overall['Improved Detection'] += len(improved)
        overall['Weakened Detection'] += len(weakened)
        overall['Unchanged Detection'] += len(unchanged)
        overall['Significant Drop'] += len(significant_drop)
        overall['Significantly Increased'] += len(significantly_increased)
#cli
    print("\n--- Yearly Statistics ---")
    df_yearly = pd.DataFrame(all_stats).sort_values("Year")
    print(df_yearly.to_string(index=False))

    print("\n--- Overall Statistics ---")
    percent_now_benign = round(overall['Now Benign'] / overall['Total Samples'] * 100, 2)
    print(f"Total Samples: {overall['Total Samples']}")
    print(f"Now Benign (vt_json==0): {overall['Now Benign']} ({percent_now_benign}%)")
    print(f"Improved Detection: {overall['Improved Detection']}")
    print(f"Weakened Detection: {overall['Weakened Detection']}")
    print(f"Unchanged Detection: {overall['Unchanged Detection']}")
    print(f"Significant Drop (>50%): {overall['Significant Drop']}")
    print(f"Significantly Increased (>50%): {overall['Significantly Increased']}")


    base_dir = os.path.dirname(csv_path)
    yearly_path = os.path.join(base_dir, "vt_yearly_statistics.csv")
    overall_path = os.path.join(base_dir, "vt_overall_statistics.csv")

    df_yearly.to_csv(yearly_path, index=False)

    df_overall = pd.DataFrame([{
        'Total Samples': overall['Total Samples'],
        'Now Benign (vt_json==0)': overall['Now Benign'],
        '% Now Benign': percent_now_benign,
        'Improved Detection': overall['Improved Detection'],
        'Weakened Detection': overall['Weakened Detection'],
        'Unchanged Detection': overall['Unchanged Detection'],
        'Significant Drop (>50%)': overall['Significant Drop'],
        'Significantly Increased (>50%)': overall['Significantly Increased']
    }])
    df_overall.to_csv(overall_path, index=False)

    print("\nCSV files saved:")
    print(f" - {yearly_path}")
    print(f" - {overall_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_vt_stats.py <csv_file>")
        sys.exit(1)

    analyze_vt_data(sys.argv[1])
